import React, { useEffect, useId, useMemo, useRef, useState } from 'react';
import { timestamp } from './RecordedTrainingRanges';
import type { RecordedDrivingSession } from './recorded-driving-session';
import { buildRecordedTimelinePaths, recordedTimelineValues } from './recorded-timeline-telemetry';

const HOLD_MS = 350;
const DRAG_THRESHOLD = 5;
const THUMB_RADIUS = 8;
const MIN_WINDOW_SECONDS = 0.5;
const TICKS = [0, 0.25, 0.5, 0.75, 1];
const CHANNELS = ['throttle', 'brake', 'gear'] as const;
const CHANNEL_LABELS = { throttle: 'Throttle', brake: 'Brake', gear: 'Gear' };
const clamp = (value: number, start: number, end: number) => Math.max(start, Math.min(end, value));
const gearLabel = (gear: number) => gear === -1 ? 'R' : gear === 0 ? 'N' : String(gear);

interface ScrubGesture {
    pointerId: number;
    startX: number;
    lastX: number;
    trackStart: number;
    trackWidth: number;
    startSeconds: number;
    endSeconds: number;
    grabOffset: number;
    seconds: number;
    mode: 'holding' | 'coarse' | 'fine';
    timer?: ReturnType<typeof setTimeout>;
}

export default function RecordedTimeline({ time, duration, disabled, seek, session, telemetryStart }: {
    time: number; duration: number; disabled: boolean; seek: (seconds: number) => void;
    session: RecordedDrivingSession | null; telemetryStart: number;
}) {
    const hintId = useId();
    const input = useRef<HTMLInputElement>(null);
    const scrub = useRef<ScrubGesture | null>(null);
    const [fine, setFine] = useState(false);
    const [zoom, setZoom] = useState(1);
    const [windowStart, setWindowStart] = useState(0);
    const max = Math.max(0, duration - 0.001);
    const maxZoom = Math.max(2, 2 ** Math.floor(Math.log2(duration / MIN_WINDOW_SECONDS)));
    const span = duration / zoom;
    const start = clamp(windowStart, 0, duration - span);
    const end = start + span;
    const playhead = span ? clamp((time - start) / span, 0, 1) * 100 : 0;
    const paths = useMemo(() => session && span > 0 ? buildRecordedTimelinePaths(session, telemetryStart, start, end) : null,
        [session, telemetryStart, start, end, span]);
    const values = session ? recordedTimelineValues(session, telemetryStart, time) : null;

    useEffect(() => {
        // Keep the view still during playback until the playhead leaves it. The full recording stays seekable below.
        if (zoom > 1 && (time < start || time > end)) setWindowStart(time - span / 2);
    }, [time, start, end, span, zoom]);

    useEffect(() => {
        if (disabled) { scrub.current = null; setFine(false); }
        return () => clearTimeout(scrub.current?.timer);
    }, [disabled]);

    const changeZoom = (next: number) => {
        setWindowStart(time - duration / next / 2);
        setZoom(next);
    };

    const pickTime = (seconds: number, lower: number, upper: number) =>
        clamp(Math.round(seconds * 100) / 100, Math.min(lower, max), Math.min(upper, max));

    const begin = (event: React.PointerEvent<HTMLElement>, graph = false) => {
        if (disabled || event.button !== 0 || event.isPrimary === false || scrub.current) return;
        const surface = event.currentTarget;
        const { left, width } = surface.getBoundingClientRect();
        const inset = graph ? 0 : THUMB_RADIUS;
        if (width <= inset * 2) return;
        // Handle pointer input ourselves so the native slider cannot fight fine mode.
        event.preventDefault();
        input.current?.focus({ preventScroll: true });
        surface.setPointerCapture?.(event.pointerId);

        const trackStart = left + inset;
        const trackWidth = width - inset * 2;
        const startSeconds = graph ? start : 0;
        const endSeconds = graph ? end : max;
        const thumbX = trackStart + clamp(time, 0, max) / max * trackWidth;
        const x = event.clientX;
        const grabbingThumb = !graph && Math.abs(x - thumbX) <= THUMB_RADIUS;
        const seconds = grabbingThumb ? clamp(time, 0, max)
            : pickTime(startSeconds + (x - trackStart) / trackWidth * (endSeconds - startSeconds), startSeconds, endSeconds);
        const gesture: ScrubGesture = {
            pointerId: event.pointerId, startX: x, lastX: x, trackStart, trackWidth, startSeconds, endSeconds,
            grabOffset: grabbingThumb ? x - thumbX : 0, seconds, mode: 'holding',
        };
        scrub.current = gesture;
        seek(seconds);
        gesture.timer = setTimeout(() => { gesture.mode = 'fine'; setFine(true); }, HOLD_MS);
    };

    const move = (event: React.PointerEvent<HTMLElement>) => {
        const gesture = scrub.current;
        if (disabled || !gesture || gesture.pointerId !== event.pointerId) return;
        const x = event.clientX;
        const delta = x - gesture.lastX;
        gesture.lastX = x;
        if (gesture.mode === 'holding') {
            if (Math.abs(x - gesture.startX) < DRAG_THRESHOLD) return;
            clearTimeout(gesture.timer);
            gesture.mode = 'coarse';
        }
        const { startSeconds, endSeconds, trackStart, trackWidth, grabOffset } = gesture;
        const widthSeconds = endSeconds - startSeconds;
        if (gesture.mode === 'fine') {
            const secondsPerPixel = Math.min(0.005, widthSeconds / trackWidth / 10);
            gesture.seconds = clamp(gesture.seconds + delta * secondsPerPixel, startSeconds, Math.min(max, endSeconds));
        } else {
            gesture.seconds = startSeconds + (x - trackStart - grabOffset) / trackWidth * widthSeconds;
        }
        const next = pickTime(gesture.seconds, startSeconds, endSeconds);
        if (next !== pickTime(time, 0, max)) seek(next);
    };

    const finish = (event: React.PointerEvent<HTMLElement>) => {
        if (!scrub.current || scrub.current.pointerId !== event.pointerId) return;
        clearTimeout(scrub.current.timer);
        scrub.current = null;
        setFine(false);
    };

    return <section className="recorded-timeline" aria-label="Recorded telemetry timeline">
        <div className="recorded-timeline-toolbar">
            <div>
                <h4>Telemetry timeline</h4>
                <span className="recorded-timeline-position"><strong>{timestamp(time)}</strong> / {timestamp(duration)}</span>
            </div>
            <div className="recorded-timeline-zoom" role="group" aria-label="Telemetry timeline zoom">
                <button type="button" aria-label="Zoom out telemetry timeline" title="Zoom out" disabled={disabled || zoom <= 1}
                    onClick={() => changeZoom(zoom / 2)}>−</button>
                <span>{zoom}×</span>
                <button type="button" aria-label="Zoom in telemetry timeline" title="Zoom in" disabled={disabled || zoom >= maxZoom}
                    onClick={() => changeZoom(zoom * 2)}>+</button>
                <button type="button" aria-label="Fit full recording" disabled={disabled || zoom <= 1} onClick={() => changeZoom(1)}>Fit</button>
            </div>
        </div>
        <dl className="recorded-timeline-legend">
            {CHANNELS.map((channel) => {
                const value = values?.[channel];
                return <div key={channel} data-channel={channel}>
                    <dt>{CHANNEL_LABELS[channel]}</dt>
                    <dd aria-label={`${CHANNEL_LABELS[channel]} at playhead`}>
                        {value == null ? '—' : channel === 'gear' ? gearLabel(value) : `${Math.round(value * 100)}%`}
                    </dd>
                </div>;
            })}
        </dl>
        <div className="recorded-timeline-plot" data-testid="recorded-telemetry-plot" aria-hidden="true" data-disabled={disabled}
            onPointerDown={(event) => begin(event, true)} onPointerMove={move} onPointerUp={finish}
            onPointerCancel={finish} onLostPointerCapture={finish}
            onClick={(event) => {
                // Pointer interactions already seek on press. Handle clicks without a preceding pointer event, too.
                if (disabled || event.detail !== 0) return;
                const { left, width } = event.currentTarget.getBoundingClientRect();
                if (width > 0) seek(pickTime(start + (event.clientX - left) / width * span, start, end));
            }}>
            <div className="recorded-timeline-chart">
                <svg viewBox="0 0 1000 120" preserveAspectRatio="none">
                    <g className="recorded-timeline-grid">
                        {TICKS.slice(1, -1).map((tick) => <line key={tick} x1={tick * 1000} x2={tick * 1000} y1="0" y2="120" />)}
                        {[8, 39, 70, 82, 116].map((y) => <line key={y} x1="0" x2="1000" y1={y} y2={y} />)}
                    </g>
                    {CHANNELS.map((channel) => paths?.[channel] && <path key={channel} data-channel={channel}
                        data-testid={`recorded-telemetry-${channel}-trace`} d={paths[channel]} />)}
                </svg>
                <div className="recorded-timeline-scale">
                    <span>100%</span><span>0%</span>
                    {paths?.gear && <><span>{gearLabel(paths.maxGear)}</span><span>R</span></>}
                </div>
                {!CHANNELS.some((channel) => paths?.[channel]) && <span className="recorded-timeline-empty">
                    {session ? 'No telemetry in this view' : 'Load telemetry to see the driving inputs'}
                </span>}
                {duration > 0 && <span className="recorded-timeline-playhead" data-testid="recorded-telemetry-playhead"
                    style={{ left: `${playhead}%` }} />}
            </div>
            <div className="recorded-timeline-axis">
                {TICKS.map((tick) => <span key={tick} style={{ left: `${tick * 100}%` }}>{timestamp(start + tick * span)}</span>)}
            </div>
        </div>
        <label className="recorded-timeline-progress">Full recording progress
            <input ref={input} type="range" min="0" max={max} step="0.01" value={time} disabled={disabled} aria-label="Recorded frame"
                aria-valuetext={`${timestamp(time)} of ${timestamp(duration)}`} aria-describedby={hintId}
                onChange={(event) => seek(event.target.valueAsNumber)} onPointerDown={begin}
                onPointerMove={move} onPointerUp={finish} onPointerCancel={finish} onLostPointerCapture={finish} />
        </label>
        <small>
            {!fine && 'Click or drag the graph to seek. '}
            <span id={hintId} className={fine ? 'recorded-timeline-fine' : undefined}>
                {fine ? 'Fine adjustment active · drag left or right' : 'Hold the progress bar, then drag to fine-tune.'}
            </span>
        </small>
    </section>;
}
