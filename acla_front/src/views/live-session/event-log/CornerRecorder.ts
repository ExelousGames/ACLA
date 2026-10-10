import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import { getCircuitMapCenterlineSegments } from 'views/circuit-maps/centerline-segments';
import type { CircuitMapCenterlineSegment, CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import type { StandardTelemetrySample } from '../live-session-types';

export interface CornerTimingPoint {
    timeMs: number;
    position: number;
    sampleIndex: number;
}

export interface DriverCornerRecord {
    id: string;
    carId: string;
    isPlayer: boolean;
    cornerId: string;
    cornerName: string;
    entry: CornerTimingPoint | null;
    exit: CornerTimingPoint;
    decelerationStart: CornerTimingPoint | null;
    accelerationStart: CornerTimingPoint | null;
    cornerTimeMs: number | null;
    decelerationToExitMs: number | null;
}

interface MotionPoint extends CornerTimingPoint {
    progress: number;
    rate?: number; // Lap fraction per second, calculated independently for each car.
}

const normalized = (value: unknown): value is number => typeof value === 'number'
    && Number.isFinite(value) && value >= 0 && value <= 1;
const forwardDistance = (from: number, to: number) => (to - from + 1) % 1;
const MAX_GAP_MS = 2000;
const HISTORY_MS = 30000;
const RATE_WINDOW_MS = 150;
const RATE_CHANGE = 0.02;

export function parseCornerTime(value: unknown): number | null {
    if (typeof value !== 'string') return null;
    const match = /^(\d+):([0-5]\d)[:.](\d{3})$/.exec(value.trim());
    if (!match) return null;
    const time = Number(match[1]) * 60000 + Number(match[2]) * 1000 + Number(match[3]);
    return Number.isSafeInteger(time) ? time : null;
}

export function formatCornerTime(timeMs: number): string {
    const time = Math.max(0, Math.round(timeMs));
    return `${String(Math.floor(time / 60000)).padStart(2, '0')}:${String(Math.floor(time / 1000) % 60).padStart(2, '0')}:${String(time % 1000).padStart(3, '0')}`;
}

const timingPoint = (point: MotionPoint): CornerTimingPoint => ({
    timeMs: point.timeMs, position: point.position, sampleIndex: point.sampleIndex,
});

function crossing(history: MotionPoint[], progress: number, position: number): CornerTimingPoint | null {
    for (let index = history.length - 1; index > 0; index -= 1) {
        const before = history[index - 1];
        const after = history[index];
        if (before.progress <= progress + 1e-9 && after.progress >= progress - 1e-9) {
            const fraction = Math.max(0, Math.min(1, (progress - before.progress) / (after.progress - before.progress)));
            return { timeMs: Math.round(before.timeMs + fraction * (after.timeMs - before.timeMs)),
                position, sampleIndex: after.sampleIndex };
        }
    }
    return null;
}

/** Records completed mapped corners. No player physics or wall clock is shared with opponents. */
export class CornerRecorder {
    private corners: CircuitMapCenterlineSegment[] = [];
    private cars = new Map<string, MotionPoint[]>();
    private previousTime: number | null = null;

    setMap(map: CircuitMapDto | null): number {
        this.reset();
        this.corners = getCircuitMapCenterlineSegments(map)
            .filter((corner) => normalized(corner.start_position) && normalized(corner.end_position)
                && forwardDistance(corner.start_position, corner.end_position) > 0
                && corner.tags.some((tag) => ['corner', 'slow corner', 'fast corner'].includes(tag.trim().toLowerCase()))
                && !corner.tags.some((tag) => tag.trim().toLowerCase() === 'consecutive corners'))
            .sort((a, b) => a.start_position - b.start_position)
            .filter((corner, index, all) => !all.slice(0, index).some((other) =>
                other.start_position === corner.start_position && other.end_position === corner.end_position));
        return this.corners.length;
    }

    reset(): void {
        this.cars.clear();
        this.previousTime = null;
    }

    tick(sample: StandardTelemetrySample, sampleIndex: number): DriverCornerRecord[] {
        const timeMs = parseCornerTime(sample.Graphics_current_time_str);
        if (timeMs === null || (sample.Graphics_status !== undefined && sample.Graphics_status !== ACC_STATUS.ACC_LIVE)) {
            this.reset();
            return [];
        }
        if (this.previousTime !== null && timeMs < this.previousTime) this.reset();
        if (this.previousTime === timeMs) return [];
        this.previousTime = timeMs;
        if (this.corners.length === 0) return [];

        const positions = new Map(Object.entries(sample.Graphics_normalized_positions ?? {}).filter(([, value]) => normalized(value)));
        const playerId = sample.Graphics_player_car_id;
        const playerKey = typeof playerId === 'number' && Number.isSafeInteger(playerId) && playerId >= 0 ? String(playerId) : 'player';
        if (normalized(sample.Graphics_normalized_car_position)) positions.set(playerKey, sample.Graphics_normalized_car_position);
        this.cars.forEach((_, id) => { if (!positions.has(id)) this.cars.delete(id); });
        const records: DriverCornerRecord[] = [];

        positions.forEach((position, carId) => {
            let history = this.cars.get(carId) ?? [];
            const previous = history[history.length - 1];
            const delta = previous ? forwardDistance(previous.position, position) : 0;
            const elapsed = previous ? timeMs - previous.timeMs : 0;
            if (previous && elapsed <= MAX_GAP_MS && delta === 0) return; // Repeated broadcast packets are not a stop.
            if (previous && (elapsed > MAX_GAP_MS || delta > 0.1)) history = [];
            const before = history[history.length - 1];
            const point: MotionPoint = { timeMs, position, sampleIndex, progress: before ? before.progress + delta : position };
            // A short window limits packet quantization noise while retaining the original onset sample.
            const rateStart = history.slice().reverse().find((item) => timeMs - item.timeMs >= RATE_WINDOW_MS);
            if (rateStart) point.rate = (point.progress - rateStart.progress) * 1000 / (timeMs - rateStart.timeMs);
            history.push(point);
            this.cars.set(carId, history);

            if (before) this.corners.forEach((corner, index) => {
                const distanceToEnd = forwardDistance(before.position, corner.end_position);
                if (distanceToEnd === 0 || distanceToEnd > delta + 1e-9) return;
                const end = before.progress + distanceToEnd;
                records.push(this.record(history, corner, index, end, carId, carId === playerKey));
            });
            // Only a bounded approach history is needed; completed records live in the event log.
            while (history.length > 2 && timeMs - history[0].timeMs > HISTORY_MS) history.shift();
        });
        return records;
    }

    private record(history: MotionPoint[], corner: CircuitMapCenterlineSegment, index: number,
        end: number, carId: string, isPlayer: boolean): DriverCornerRecord {
        const start = end - forwardDistance(corner.start_position, corner.end_position);
        const entry = crossing(history, start, corner.start_position);
        const exit = crossing(history, end, corner.end_position)!;
        // Do not backtrack into another corner's braking phase.
        const previousEnds = this.corners.filter((other) => other !== corner)
            .map((other) => forwardDistance(other.end_position, corner.start_position));
        const approachStart = start - Math.min(0.25, ...previousEnds);
        const points = history.filter((point) => point.progress >= approachStart && point.progress <= end);
        let slowest = -1;
        points.forEach((point, pointIndex) => {
            if (point.progress >= start && point.rate !== undefined
                && (slowest < 0 || point.rate <= points[slowest].rate! * (1 + 1e-6))) slowest = pointIndex;
        });
        let decelerationStart: CornerTimingPoint | null = null;
        let accelerationStart: CornerTimingPoint | null = null;
        if (slowest >= 0) {
            let peak = slowest;
            for (let i = slowest - 1; i >= 0 && points[i].rate !== undefined; i -= 1) {
                if (points[i].rate! < points[peak].rate! * (1 - RATE_CHANGE)) break;
                if (points[i].rate! > points[peak].rate! * (1 + 1e-6)) peak = i;
            }
            if (peak > 0 && peak < slowest && points[peak - 1].rate !== undefined
                && points[peak - 1].rate! <= points[peak].rate! * (1 + RATE_CHANGE)
                && points[peak].rate! > points[slowest].rate! * (1 + RATE_CHANGE)) {
                decelerationStart = timingPoint(points[peak]);
            }
            const accelerating = points.slice(slowest + 1).some((point) =>
                point.rate !== undefined && point.rate > points[slowest].rate! * (1 + RATE_CHANGE));
            if (accelerating) accelerationStart = timingPoint(points[slowest]);
        }
        return {
            id: `${carId}:${corner.id}:${exit.sampleIndex}`, carId, isPlayer, cornerId: corner.id,
            cornerName: `Corner ${index + 1}`, entry, exit, decelerationStart, accelerationStart,
            cornerTimeMs: entry ? exit.timeMs - entry.timeMs : null,
            decelerationToExitMs: decelerationStart ? exit.timeMs - decelerationStart.timeMs : null,
        };
    }
}
