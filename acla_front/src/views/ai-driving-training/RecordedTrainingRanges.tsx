import React, { useId, useState } from 'react';
import type { DesktopGame } from 'contexts/DesktopGameContext';
import trainingLabels from 'data/recorded-training-labels.json';

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
    disabled: boolean;
    dirty: boolean;
    canSave: boolean;
    playhead(): number;
    seek(seconds: number): void;
    change(draft: TrainingRangeDraft): void;
    changeOffset?(offsetSeconds: number): void;
    save(): void;
}

export default function RecordedTrainingRanges({ selection, duration, time, step, overlap, disabled, dirty, canSave,
    playhead, seek, change, changeOffset, save }: Props) {
    const ranges = selection.ranges ?? [];
    const [start, setStart] = useState('');
    const [end, setEnd] = useState('');
    const [label, setLabel] = useState('');
    const [editing, setEditing] = useState<number | null>(null);
    const [error, setError] = useState('');
    const [offsetInput, setOffsetInput] = useState<string>();
    const offsetHelpId = useId();
    const savedOffsetMs = selection.offsetSeconds === undefined ? '' : selection.offsetSeconds * 1000;
    const reset = () => { setStart(''); setEnd(''); setLabel(''); setEditing(null); setError(''); };
    const mark = (set: (value: string) => void) => {
        set(String(Math.min(duration, Math.round(playhead() * 100) / 100)));
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
        const others = ranges.filter((_, index) => index !== editing);
        if (others.some((other) => range.start <= other.end && range.end >= other.start)) {
            setError('This range touches another range. Edit that range or leave a gap between them.');
            return;
        }
        change({ ranges: [...others, range].sort((a, b) => a.start - b.start) });
        reset();
    };

    return <section className="recorded-training-ranges" aria-label="Training ranges">
        <h3>Training ranges</h3>
        {duration > 0 && <div className="recorded-range-track" aria-hidden="true">
            {overlap && <span className="recorded-range-overlap"
                style={{ left: `${100 * overlap.start / duration}%`, width: `${100 * (overlap.end - overlap.start) / duration}%` }} />}
            {ranges.map((range) => <span className="recorded-range-marker" key={range.start}
                style={{ left: `${100 * range.start / duration}%`, width: `${100 * (range.end - range.start) / duration}%` }} />)}
            <span className="recorded-range-playhead" style={{ left: `${Math.min(100, 100 * time / duration)}%` }} />
        </div>}
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
        {error && <p className="driving-error" role="alert">{error}</p>}
        {ranges.length > 0 && <ol className="recorded-range-list">
            {ranges.map((range, index) => <li key={range.start}>
                <span>{timestamp(range.start)} – {timestamp(range.end)}{range.label && ` · ${range.label}`}</span>
                <button disabled={disabled} aria-label={`Jump to start of range ${index + 1}`} onClick={() => seek(range.start)}>Start</button>
                <button disabled={disabled} aria-label={`Jump to end of range ${index + 1}`} onClick={() => seek(range.end)}>End</button>
                <button disabled={disabled} aria-label={`Edit range ${index + 1}`} onClick={() => {
                    setStart(String(range.start)); setEnd(String(range.end)); setLabel(range.label ?? ''); setEditing(index); setError('');
                }}>Edit</button>
                <button disabled={disabled} aria-label={`Remove range ${index + 1}`} onClick={() => {
                    change({ ranges: ranges.filter((_, item) => item !== index) }); reset();
                }}>Remove</button>
            </li>)}
        </ol>}
        <div className="driving-buttons">
            {selection.ranges !== null && <button disabled={disabled} onClick={() => { change({ ranges: null }); reset(); }}>Use full recording</button>}
            <button className="driving-primary" disabled={disabled || !canSave}
                onClick={save}>Save training annotation</button>
            {dirty && <span role="status">Unsaved annotation changes</span>}
        </div>
    </section>;
}
