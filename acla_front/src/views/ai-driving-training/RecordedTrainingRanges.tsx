import React, { useId, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { DesktopGame } from 'contexts/DesktopGameContext';
import trainingLabels from 'data/recorded-training-labels.json';
import type { RecordedGearShift } from './recorded-driving-session';

export interface TrainingRange { start: number; end: number; label?: string }
export interface TrainingRangeDraft {
    /** Null uses the full overlap. An empty list selects no footage. */
    ranges: TrainingRange[] | null;
    fullRecordingLabel?: string;
    /** Null clears a saved simulator override. */
    game?: DesktopGame | null;
    /** Manual alignment when the video has no usable telemetry start time. */
    offsetSeconds?: number;
}
interface SavedTrainingSelection extends TrainingRangeDraft {
    version: 2;
    videoFileName: string;
    telemetryFileName: string;
}
export type TrainingSelection = (TrainingRangeDraft & { version: 1 }) | SavedTrainingSelection;
export interface TrainingAnnotation {
    id: string;
    name: string;
    selection: SavedTrainingSelection;
    videoUrl: string;
    telemetryUrl: string;
    videoTelemetryStartSeconds?: number;
}

declare global {
    interface Window {
        recordedTrainingRanges?: {
            read(file: File): Promise<TrainingSelection | null>;
            open(file: File): Promise<TrainingAnnotation>;
            scan(files: File[]): Promise<Array<TrainingAnnotation | { error: string } | null>>;
            update(id: string, draft: TrainingRangeDraft): Promise<TrainingAnnotation>;
            write(video: File, telemetry: File, draft: TrainingRangeDraft): Promise<TrainingAnnotation>;
        };
    }
}

export const timestamp = (seconds: number) => {
    const centiseconds = Math.round(seconds * 100);
    return `${Math.floor(centiseconds / 6000)}:${((centiseconds % 6000) / 100).toFixed(2).padStart(5, '0')}`;
};

const MAX_TIMELINE_ZOOM = 32;
const SHIFT_HISTORY_SECONDS = 1;
const SHIFT_LOOKAHEAD_SECONDS = 1;
const LABEL_COLORS = ['#d3ba79', '#ada9f4', '#87b9ec', '#eabb7a', '#e6a8ce', '#7cc9d4',
    '#f3a18d', '#bfd88d', '#e5a2ad', '#b7a2e9', '#a3b9d8', '#79d3c2', '#f2da8c', '#a8c0f5'];
const rangeTime = ({ start, end }: TrainingRange) => `${timestamp(start)} – ${timestamp(end)}`;

function LabelTimeline({ ranges, fullLabel, overlap, duration, time, disabled, selectedIndices, select, selectMany }: {
    ranges: TrainingRange[] | null; fullLabel?: string; overlap: TrainingRange | null; duration: number; time: number;
    disabled: boolean; selectedIndices: Set<number>; select: (index: number) => void; selectMany: (indices: Set<number>) => void;
}) {
    const [zoom, setZoom] = useState(1);
    const viewport = useRef<HTMLDivElement>(null);
    const ruler = useRef<HTMLDivElement>(null);
    const zoomAnchor = useRef<{ fraction: number; viewX: number } | null>(null);
    const drag = useRef<{ pointerId: number; x: number; y: number } | null>(null);
    const [marquee, setMarquee] = useState<{ left: number; top: number; width: number; height: number }>();
    const canSelect = !disabled && Boolean(ranges?.length);
    const overlapStart = overlap?.start, overlapEnd = overlap?.end;
    const lanes = useMemo(() => {
        const lanes = new Map<string, { rowEnds: number[]; clips: { range: TrainingRange; index: number | null; row: number }[] }>();
        const fullRange = overlapStart === undefined || overlapEnd === undefined ? []
            : [{ start: overlapStart, end: overlapEnd, label: fullLabel }];
        (ranges ?? fullRange).forEach((range, index) => {
            const label = range.label || 'Unlabeled';
            const lane = lanes.get(label) ?? { rowEnds: [], clips: [] };
            // Keep overlapping clips with the same label individually clickable.
            const availableRow = lane.rowEnds.findIndex((end) => end <= range.start);
            const row = availableRow === -1 ? lane.rowEnds.length : availableRow;
            lane.rowEnds[row] = range.end;
            lane.clips.push({ range, index: ranges === null ? null : index, row });
            lanes.set(label, lane);
        });
        return lanes;
    }, [ranges, fullLabel, overlapStart, overlapEnd]);
    const clamp = (seconds: number) => Math.min(duration, Math.max(0, seconds));
    const percent = (seconds: number) => `${100 * seconds / duration}%`;
    const position = ({ start, end }: TrainingRange) => ({ left: percent(clamp(start)), width: percent(clamp(end) - clamp(start)) });
    const divisions = 4 * zoom;
    const changeZoom = (next: number) => {
        const scroll = viewport.current;
        const axis = ruler.current;
        if (scroll && axis?.offsetWidth) {
            // Keep the playhead in place when visible; otherwise keep the center of the visible time window.
            const visibleWidth = scroll.clientWidth - axis.offsetLeft;
            const playheadX = clamp(time) / duration * axis.offsetWidth - scroll.scrollLeft;
            const viewX = playheadX >= 0 && playheadX <= visibleWidth ? playheadX : visibleWidth / 2;
            zoomAnchor.current = { fraction: (scroll.scrollLeft + viewX) / axis.offsetWidth, viewX };
        }
        setZoom(next);
    };
    useLayoutEffect(() => {
        if (viewport.current && ruler.current && zoomAnchor.current) {
            const { fraction, viewX } = zoomAnchor.current;
            viewport.current.scrollLeft = zoom === 1 ? 0 : fraction * ruler.current.offsetWidth - viewX;
        }
        zoomAnchor.current = null;
    }, [zoom]);

    const beginSelection = (event: React.PointerEvent<HTMLDivElement>) => {
        if (!canSelect || drag.current || event.button !== 0 || (event.pointerType && event.pointerType !== 'mouse')) return;
        const target = event.target as HTMLElement;
        if (target.closest('.recorded-label-timeline-clip') || (target !== event.currentTarget
            && !target.closest('.recorded-label-timeline-lane, .recorded-label-timeline-ruler'))) return;
        const { left, top } = event.currentTarget.getBoundingClientRect();
        drag.current = { pointerId: event.pointerId, x: event.clientX - left, y: event.clientY - top };
        event.preventDefault();
        event.currentTarget.setPointerCapture?.(event.pointerId);
    };
    const selectionBox = (event: React.PointerEvent<HTMLDivElement>, bounds: DOMRect) => {
        const origin = drag.current;
        if (!origin || origin.pointerId !== event.pointerId || !canSelect) return;
        const scroll = viewport.current;
        const firstVisibleX = (scroll?.scrollLeft ?? 0) + (ruler.current?.offsetLeft ?? 0);
        const lastVisibleX = scroll?.clientWidth ? Math.min(bounds.width, scroll.scrollLeft + scroll.clientWidth) : bounds.width;
        // Anchor the start in the grid, but don't select off-screen clips by dragging past the viewport.
        const x = Math.min(lastVisibleX, Math.max(firstVisibleX, event.clientX - bounds.left));
        const y = Math.min(bounds.height, Math.max(0, event.clientY - bounds.top));
        const width = Math.abs(x - origin.x), height = Math.abs(y - origin.y);
        if (Math.max(width, height) < 4) return;
        return { left: Math.min(origin.x, x), top: Math.min(origin.y, y), width, height };
    };
    const cancelSelection = (event: React.PointerEvent<HTMLDivElement>) => {
        if (!drag.current || drag.current.pointerId !== event.pointerId) return;
        drag.current = null;
        setMarquee(undefined);
    };
    const finishSelection = (event: React.PointerEvent<HTMLDivElement>) => {
        if (!drag.current || drag.current.pointerId !== event.pointerId) return;
        const grid = event.currentTarget;
        const bounds = grid.getBoundingClientRect();
        const box = selectionBox(event, bounds);
        cancelSelection(event);
        if (!canSelect) return;
        const indices = new Set<number>();
        if (box) {
            const left = bounds.left + box.left, top = bounds.top + box.top;
            grid.querySelectorAll<HTMLButtonElement>('button[data-range-index]').forEach((clip) => {
                const rect = clip.getBoundingClientRect();
                if (rect.left < left + box.width && rect.right > left && rect.top < top + box.height && rect.bottom > top) {
                    indices.add(Number(clip.dataset.rangeIndex));
                }
            });
        }
        selectMany(indices);
    };

    return <>
        <div className="recorded-label-timeline-title">
            <h4>Label timeline <small>Video time</small></h4>
            <div className="recorded-label-timeline-zoom" role="group" aria-label="Label timeline zoom">
                <button type="button" aria-label="Zoom out timeline" title="Zoom out" disabled={zoom === 1} onClick={() => changeZoom(zoom / 2)}>−</button>
                <output aria-label="Timeline zoom level">{zoom}×</output>
                <button type="button" aria-label="Zoom in timeline" title="Zoom in" disabled={zoom === MAX_TIMELINE_ZOOM} onClick={() => changeZoom(zoom * 2)}>+</button>
                <button type="button" aria-label="Fit timeline to view" title="Fit the full recording" disabled={zoom === 1} onClick={() => changeZoom(1)}>Fit</button>
            </div>
        </div>
        <div className="recorded-label-timeline-scroll" ref={viewport} role="group" aria-label="Label timeline viewport" tabIndex={zoom > 1 ? 0 : undefined}>
            <div className="recorded-label-timeline-grid" style={{ '--timeline-zoom': zoom } as React.CSSProperties}
                data-selectable={canSelect || undefined} onPointerDown={beginSelection}
                onPointerMove={(event) => {
                    if (drag.current && drag.current.pointerId === event.pointerId) {
                        setMarquee(selectionBox(event, event.currentTarget.getBoundingClientRect()));
                    }
                }} onPointerUp={finishSelection} onPointerCancel={cancelSelection} onLostPointerCapture={cancelSelection}>
                <div className="recorded-label-timeline-heading">Label</div>
                <div className="recorded-label-timeline-ruler" ref={ruler} role="group" aria-label="Elapsed video time">
                    {Array.from({ length: divisions + 1 }, (_, tick) => <span key={tick} style={{ left: `${100 * tick / divisions}%` }}>
                        {timestamp(duration * tick / divisions)}
                    </span>)}
                </div>
                <div className="recorded-label-timeline-guide" style={{ backgroundSize: `${100 / divisions}% 100%` }} aria-hidden="true">
                    {overlap && <span className="recorded-label-timeline-overlap" style={position(overlap)} />}
                    <span className="recorded-label-timeline-playhead" style={{ left: percent(clamp(time)) }} />
                </div>
                {Array.from(lanes, ([label, { rowEnds, clips }]) => {
                    const color = LABEL_COLORS[trainingLabels.indexOf(label)] ?? '#a3adb7';
                    return <React.Fragment key={label}>
                        <div className="recorded-label-timeline-label" title={label}>
                            <span aria-hidden="true" style={{ backgroundColor: color }} />{label}
                        </div>
                        <div className="recorded-label-timeline-lane" style={{ '--clip-color': color, '--clip-rows': rowEnds.length } as React.CSSProperties}>
                            {clips.map(({ range, index, row }) => index === null
                                ? <span className="recorded-label-timeline-clip" key="full" style={position(range)} role="img"
                                    aria-label={`Full aligned recording: ${label}, ${rangeTime(range)}`} title={`${label} · ${rangeTime(range)}`} />
                                : <button className="recorded-label-timeline-clip" type="button" key={index}
                                    style={{ ...position(range), '--clip-row': row } as React.CSSProperties} data-at-end={range.start >= duration || undefined}
                                    data-range-index={index}
                                    aria-label={`Range ${index + 1}: ${label}, ${rangeTime(range)}`} title={`${label} · ${rangeTime(range)}`}
                                    disabled={disabled} aria-pressed={selectedIndices.has(index)} onClick={() => select(index)} />)}
                        </div>
                    </React.Fragment>;
                })}
                {!lanes.size && <div className="recorded-label-timeline-empty">
                    {ranges === null ? 'No aligned footage to show.' : 'No clips on the timeline yet.'}
                </div>}
                {canSelect && marquee && <div className="recorded-label-timeline-marquee" style={marquee} aria-hidden="true" />}
            </div>
        </div>
    </>;
}

function TrainingLabelOptions({ value }: { value?: string }) {
    return <>
        <option value="">Unlabeled</option>
        {value && !trainingLabels.includes(value) && <option value={value} disabled>{value} (legacy)</option>}
        {trainingLabels.map((label) => <option key={label} value={label}>{label}</option>)}
    </>;
}

interface Props {
    selection: TrainingSelection;
    duration: number;
    time: number;
    step: number;
    overlap: TrainingRange | null;
    gearShifts: RecordedGearShift[];
    shiftVideoOffset: number;
    disabled: boolean;
    dirty: boolean;
    canSave: boolean;
    playhead(): number;
    seek(seconds: number): void;
    change(draft: TrainingRangeDraft): void;
    changeOffset?(offsetSeconds: number): void;
    save(): void;
}

export default function RecordedTrainingRanges({ selection, duration, time, step, overlap, gearShifts, shiftVideoOffset, disabled, dirty, canSave,
    playhead, seek, change, changeOffset, save }: Props) {
    const ranges = selection.ranges ?? [];
    const [start, setStart] = useState('');
    const [end, setEnd] = useState('');
    const [label, setLabel] = useState('');
    const [editing, setEditing] = useState<number | null>(null);
    const [selectedIndices, setSelectedIndices] = useState<Set<number>>(() => new Set());
    const selectedIndex = selectedIndices.size === 1 ? Array.from(selectedIndices)[0] : null;
    const selected = selectedIndex === null ? undefined : ranges[selectedIndex];
    const [error, setError] = useState('');
    const [offsetInput, setOffsetInput] = useState<string>();
    const offsetHelpId = useId();
    const savedOffsetMs = selection.offsetSeconds === undefined ? '' : selection.offsetSeconds * 1000;
    const reset = () => { setStart(''); setEnd(''); setLabel(''); setEditing(null); setError(''); };
    const removeSelection = () => {
        change({ ranges: ranges.filter((_, index) => !selectedIndices.has(index)) });
        setSelectedIndices(new Set());
        reset();
    };
    const mark = (set: (value: string) => void) => {
        set(String(Math.min(duration, Math.round(playhead() * 100) / 100)));
        setError('');
    };
    const commitRange = (range: TrainingRange) => {
        const next = [...ranges];
        if (editing === null) next.push(range);
        else next[editing] = range;
        next.sort((a, b) => a.start - b.start);
        change({ ranges: next });
        setSelectedIndices(new Set([next.indexOf(range)]));
        setError('');
    };
    const commit = () => {
        const name = label.trim();
        const range = { start: Number(start), end: Number(end), ...(name ? { label: name } : {}) };
        if (!start || !end || !Number.isFinite(range.start) || !Number.isFinite(range.end)
            || range.start < 0 || range.end > duration || range.end - range.start < step - 1e-7) {
            setError(`Choose a start and end inside the video, at least ${step} s apart.`);
            return;
        }
        commitRange(range);
        reset();
    };
    const addShift = (direction: RecordedGearShift['direction']) => {
        if (!overlap || overlap.end - overlap.start < step - 1e-7) {
            setError(`Adding a shift label needs at least ${step} s of aligned video and telemetry.`);
            return;
        }
        const at = playhead();
        const shiftRanges = (selection.ranges ?? [{ ...overlap, label: selection.fullRecordingLabel }]).filter(({ label }) => {
            const name = label?.trim().toLowerCase();
            return name === 'shift up' || name === 'shift down';
        });
        let nearest: { at: number; range: TrainingRange } | undefined;
        for (let index = 0; index < gearShifts.length; index++) {
            const shift = gearShifts[index];
            if (shift.direction !== direction) continue;
            const before = shift.before + shiftVideoOffset, after = shift.after + shiftVideoOffset;
            if (before < overlap.start - 1e-7 || after > overlap.end + 1e-7) continue;
            if (nearest && Math.abs(at - after) >= Math.abs(at - nearest.at)) continue;

            // Fit the clip between existing shift ranges, including when adding history or lookahead.
            let gapStart = overlap.start, gapEnd = overlap.end;
            for (const marked of shiftRanges) {
                if (marked.end <= before + 1e-7) gapStart = Math.max(gapStart, marked.end);
                else gapEnd = Math.min(gapEnd, marked.start);
            }
            if (after > gapEnd + 1e-7 || gapEnd - gapStart < step - 1e-7) continue;

            // Look only as far as replay can reach while still sampling the selected shift's old gear.
            const availableSteps = Math.floor((gapEnd - gapStart + 1e-7) / step);
            const latestEnd = Math.min(gapEnd, before + availableSteps * step);
            const lookaheadEnd = Math.min(after + SHIFT_LOOKAHEAD_SECONDS, latestEnd);
            let lastShiftTime = after;
            for (let next = index + 1; next < gearShifts.length; next++) {
                const future = gearShifts[next];
                const futureTime = future.after + shiftVideoOffset;
                if (futureTime > lookaheadEnd + 1e-7) break;
                if (future.direction === direction) lastShiftTime = futureTime;
            }

            let start = Math.max(gapStart, before - SHIFT_HISTORY_SECONDS);
            // End on a replay step that sees the last included shift's new gear.
            const steps = Math.max(1, Math.ceil((lastShiftTime - start - 1e-7) / step));
            let end = start + steps * step;
            if (end > gapEnd) {
                const length = Math.min(steps, availableSteps) * step;
                end = Math.min(latestEnd, before + length);
                start = Math.max(gapStart, end - length);
            }
            if (end < lastShiftTime - 1e-7 || end - start < step - 1e-7) continue;
            const range = { start, end, label: direction === 'up' ? 'Shift up' : 'Shift down' };
            nearest = { at: after, range };
        }
        if (!nearest) {
            setError(`No unmarked shift ${direction} found in the aligned recording.`);
            return;
        }
        commitRange(nearest.range);
        seek(nearest.at);
    };

    return <section className="recorded-training-ranges" aria-label="Training ranges">
        <h3>Training ranges</h3>
        {duration > 0 && <section className="recorded-label-timeline" aria-label="Label timeline">
            <LabelTimeline ranges={selection.ranges} fullLabel={selection.fullRecordingLabel} overlap={overlap}
                duration={duration} time={time} disabled={disabled} selectedIndices={selectedIndices} selectMany={setSelectedIndices}
                select={(index) => { setSelectedIndices(new Set([index])); seek(ranges[index].start); }} />
            {selected && selectedIndex !== null ? <div className="recorded-label-timeline-actions" role="group" aria-label={`Range ${selectedIndex + 1} controls`}>
                <div><strong>{selected.label || 'Unlabeled'}</strong><small>{rangeTime(selected)}</small></div>
                <button disabled={disabled} aria-label={`Jump to start of range ${selectedIndex + 1}`} onClick={() => seek(selected.start)}>Start</button>
                <button disabled={disabled} aria-label={`Jump to end of range ${selectedIndex + 1}`} onClick={() => seek(selected.end)}>End</button>
                <button disabled={disabled} aria-label={`Edit range ${selectedIndex + 1}`} onClick={() => {
                    setStart(String(selected.start)); setEnd(String(selected.end)); setLabel(selected.label ?? ''); setEditing(selectedIndex); setError('');
                }}>Edit</button>
                <button disabled={disabled} aria-label={`Remove range ${selectedIndex + 1}`} onClick={removeSelection}>Remove</button>
            </div> : selectedIndices.size > 1 ? <div className="recorded-label-timeline-actions" role="group" aria-label="Selected clips controls">
                <div><strong>{selectedIndices.size} clips selected</strong></div>
                <button disabled={disabled} aria-label={`Remove ${selectedIndices.size} selected ranges`} onClick={removeSelection}>Remove</button>
            </div> : ranges.length > 0 && <p className="driving-note recorded-label-timeline-hint">
                Click a clip to jump to its start, or left-click and drag from empty timeline space to select multiple clips.
            </p>}
        </section>}
        <p className="driving-note">
            {overlap ? `Video and telemetry overlap from ${timestamp(overlap.start)} to ${timestamp(overlap.end)}. ` : ''}
            {selection.ranges === null ? 'Full aligned recording'
                : ranges.length ? `${ranges.length} selected ${ranges.length === 1 ? 'range' : 'ranges'} · ${timestamp(ranges.reduce((total, range) => total + range.end - range.start, 0))} of footage`
                    : 'No footage selected.'}
        </p>
        {changeOffset && <>
            <label className="recorded-setting">Telemetry offset (ms)
                <input type="number" step="any" placeholder="0" value={offsetInput ?? savedOffsetMs} disabled={disabled}
                    aria-describedby={offsetHelpId} onChange={({ currentTarget }) => {
                        setOffsetInput(currentTarget.value);
                        if (Number.isFinite(currentTarget.valueAsNumber)) changeOffset(currentTarget.valueAsNumber / 1000);
                    }} onBlur={() => setOffsetInput(undefined)} />
            </label>
            <p className="driving-note" id={offsetHelpId}>Positive values read earlier telemetry.</p>
        </>}
        {selection.ranges === null && <label className="recorded-setting">Full recording label (optional)
            <select value={selection.fullRecordingLabel ?? ''} disabled={disabled}
                onChange={(event) => change({ ranges: null, fullRecordingLabel: event.target.value })}>
                <TrainingLabelOptions value={selection.fullRecordingLabel} />
            </select>
        </label>}
        <div className="recorded-settings">
            <label>Range start (seconds)<input type="number" min="0" max={duration} step="0.01" value={start}
                disabled={disabled} onChange={(event) => { setStart(event.target.value); setError(''); }} /></label>
            <label>Range end (seconds)<input type="number" min="0" max={duration} step="0.01" value={end}
                disabled={disabled} onChange={(event) => { setEnd(event.target.value); setError(''); }} /></label>
            <label>Clip label (optional)<select value={label} disabled={disabled} onChange={(event) => setLabel(event.target.value)}>
                <TrainingLabelOptions value={label} />
            </select></label>
        </div>
        <div className="driving-buttons">
            <button disabled={disabled} onClick={() => mark(setStart)}>Mark start here</button>
            <button disabled={disabled} onClick={() => mark(setEnd)}>Mark end here</button>
            <button disabled={disabled || !start || !end} onClick={commit}>{editing === null ? 'Add range' : 'Update range'}</button>
            {editing !== null && <button disabled={disabled} onClick={reset}>Cancel edit</button>}
        </div>
        <div className="driving-buttons" role="group" aria-label="Quick shift labels">
            <button type="button" disabled={disabled || editing !== null} onClick={() => addShift('up')}>Add shift up</button>
            <button type="button" disabled={disabled || editing !== null} onClick={() => addShift('down')}>Add shift down</button>
        </div>
        <p className="driving-note">Each button finds the nearest gear change outside existing shift clips, adds up to 1,000 ms of history,
            and looks ahead 1,000 ms for shifts in the same direction.
            {selection.ranges === null && ' Adding a range switches training from the full recording to selected ranges.'}</p>
        {error && <p className="driving-error" role="alert">{error}</p>}
        <div className="driving-buttons">
            {selection.ranges !== null && <button disabled={disabled} onClick={() => { change({ ranges: null }); setSelectedIndices(new Set()); reset(); }}>Use full recording</button>}
            <button className="driving-primary" disabled={disabled || !canSave}
                onClick={save}>Save training annotation</button>
            {dirty && <span role="status">Unsaved annotation changes</span>}
        </div>
    </section>;
}
