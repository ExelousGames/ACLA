import { DEFAULT_DRIVING_CONFIG } from './driving-model-config';
import { findRecordedTelemetry, type RecordedDrivingRow, type RecordedDrivingSession } from './recorded-driving-session';

const VIEW_WIDTH = 1000;
const CLOCK_EPSILON = 1e-9;
const MAX_GAP_SECONDS = DEFAULT_DRIVING_CONFIG.runtime.telemetryMaxAgeMs / 1000;
const FIELDS = { throttle: 'Physics_gas', brake: 'Physics_brake', gear: 'Physics_gear' } as const;
type Channel = keyof typeof FIELDS;

function channelValue(row: RecordedDrivingRow | undefined, channel: Channel): number | null {
    const value = row?.sample?.[FIELDS[channel]];
    if (typeof value !== 'number' || !Number.isFinite(value)) return null;
    // Telemetry stores reverse as 0 and neutral as 1 for both ACC and normalized iRacing data.
    if (channel === 'gear') return Number.isInteger(value) && value >= 0 ? value - 1 : null;
    return Math.min(1, Math.max(0, value));
}

/** Pedals are fractions and gear is -1 for reverse, 0 for neutral, or the forward gear. */
export function recordedTimelineValues(session: RecordedDrivingSession, telemetryAtStart: number, time: number) {
    const telemetrySeconds = telemetryAtStart + time;
    const elapsed = telemetrySeconds - session.startTimeSeconds;
    const row = elapsed <= session.durationSeconds + CLOCK_EPSILON
        ? findRecordedTelemetry(session, telemetrySeconds) ?? undefined : undefined;
    return { throttle: channelValue(row, 'throttle'), brake: channelValue(row, 'brake'), gear: channelValue(row, 'gear') };
}

function firstRowAfter(rows: readonly RecordedDrivingRow[], time: number) {
    let low = 0, high = rows.length;
    while (low < high) {
        const middle = Math.floor((low + high) / 2);
        if (rows[middle].timeSeconds <= time) low = middle + 1;
        else high = middle;
    }
    return low;
}

interface PlotPoint { index: number; x: number; value: number; run: number }
interface PlotBucket { column: number; first: PlotPoint; min: PlotPoint; max: PlotPoint; last: PlotPoint }

function plotChannel(rows: readonly RecordedDrivingRow[], first: number, last: number,
    start: number, widthSeconds: number, channel: Channel, maxGear: number) {
    const commands: string[] = [];
    let bucket: PlotBucket | undefined;
    let lastPlotted: PlotPoint | undefined;
    let singlePoint = false;
    const stepped = channel === 'gear';
    const y = (value: number) => (stepped ? 116 - (value + 1) * 34 / (maxGear + 1) : 70 - value * 62).toFixed(2);
    const moveTo = (point: PlotPoint) => `M${point.x.toFixed(2)} ${y(point.value)}`;
    const plotIsolatedPoint = () => {
        if (singlePoint && lastPlotted && lastPlotted.x >= -0.005 && lastPlotted.x <= VIEW_WIDTH + 0.005) {
            commands.push(moveTo(lastPlotted), 'h0');
        }
    };

    const flush = () => {
        if (!bucket) return;
        // Keep the extrema as well as the edges of each column to retain brief pedal and gear changes.
        for (const point of [bucket.first, bucket.min, bucket.max, bucket.last].sort((a, b) => a.index - b.index)) {
            if (point.index === lastPlotted?.index) continue;
            if (!lastPlotted || point.run !== lastPlotted.run) {
                plotIsolatedPoint();
                singlePoint = true;
            } else {
                if (singlePoint) commands.push(moveTo(lastPlotted));
                const x = point.x.toFixed(2);
                commands.push(stepped ? `H${x}V${y(point.value)}` : `L${x} ${y(point.value)}`);
                singlePoint = false;
            }
            lastPlotted = point;
        }
    };

    let run = 0;
    let previous: RecordedDrivingRow | undefined;
    for (let index = first; index <= last; index++) {
        const row = rows[index];
        const value = channelValue(row, channel);
        if (value === null) {
            previous = undefined;
            continue;
        }
        if (!previous || row.segment !== previous.segment || row.timeSeconds - previous.timeSeconds > MAX_GAP_SECONDS + CLOCK_EPSILON) run++;
        previous = row;

        const x = (row.timeSeconds - start) / widthSeconds * VIEW_WIDTH;
        const point = { index, x, value, run };
        const column = Math.floor(x);
        if (!bucket || column !== bucket.column) {
            flush();
            bucket = { column, first: point, min: point, max: point, last: point };
        } else {
            if (value < bucket.min.value) bucket.min = point;
            if (value > bucket.max.value) bucket.max = point;
            bucket.last = point;
        }
    }
    flush();
    plotIsolatedPoint();
    return commands.join(' ');
}

const gearScales = new WeakMap<RecordedDrivingSession, number>();

/** Builds visible traces on the video clock. Each path is bounded by the chart resolution. */
export function buildRecordedTimelinePaths(session: RecordedDrivingSession, telemetryAtStart: number, start: number, end: number) {
    const { rows } = session;
    const cachedMaxGear = gearScales.get(session);
    const maxGear = cachedMaxGear ?? rows.reduce((max, row) => Math.max(max, channelValue(row, 'gear') ?? 0), 1);
    if (cachedMaxGear === undefined) gearScales.set(session, maxGear);

    const offset = telemetryAtStart - session.startTimeSeconds;
    const from = start + offset;
    const to = end + offset;
    if (!(end > start) || to < -CLOCK_EPSILON || from > session.durationSeconds + CLOCK_EPSILON) {
        return { throttle: '', brake: '', gear: '', maxGear };
    }
    // Include the rows just outside the viewport so a trace can continue through either edge.
    const first = Math.max(0, firstRowAfter(rows, from) - 1);
    const last = Math.min(rows.length - 1, firstRowAfter(rows, to));
    const plot = (channel: Channel) => plotChannel(rows, first, last, from, end - start, channel, maxGear);
    return { throttle: plot('throttle'), brake: plot('brake'), gear: plot('gear'), maxGear };
}
