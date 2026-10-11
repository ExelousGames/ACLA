import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import type { StandardTelemetrySample } from '../live-session-types';
import { parseCornerTime, type CornerTimingPoint } from './CornerRecorder';

export interface DriverLapRecord {
    id: string;
    carId: string;
    isPlayer: boolean;
    start: CornerTimingPoint;
    end: CornerTimingPoint;
    lapTimeMs: number;
}

interface LapProgress {
    previous: CornerTimingPoint;
    start: CornerTimingPoint | null;
}

const normalized = (value: unknown): value is number => typeof value === 'number'
    && Number.isFinite(value) && value >= 0 && value <= 1;

/** Only consecutive observed finish-line crossings form a full lap; no alternate clock. */
export class LapRecorder {
    private cars = new Map<string, LapProgress>();
    private previousTime: number | null = null;

    reset(): void {
        this.cars.clear();
        this.previousTime = null;
    }

    tick(sample: StandardTelemetrySample, sampleIndex: number): DriverLapRecord[] {
        const timeMs = parseCornerTime(sample.Graphics_current_time_str);
        if (timeMs === null || (sample.Graphics_status !== undefined && sample.Graphics_status !== ACC_STATUS.ACC_LIVE)) {
            this.reset();
            return [];
        }
        if (this.previousTime !== null && (timeMs < this.previousTime || timeMs - this.previousTime > 2000)) {
            this.reset();
        }
        const positions = new Map(Object.entries(sample.Graphics_normalized_positions ?? {}).filter(([, value]) => normalized(value)));
        const playerId = sample.Graphics_player_car_id;
        const playerKey = typeof playerId === 'number' && Number.isSafeInteger(playerId) && playerId >= 0 ? String(playerId) : 'player';
        if (normalized(sample.Graphics_normalized_car_position)) positions.set(playerKey, sample.Graphics_normalized_car_position);
        this.cars.forEach((_, id) => { if (!positions.has(id)) this.cars.delete(id); });
        if (this.previousTime === timeMs) return [];
        this.previousTime = timeMs;
        const records: DriverLapRecord[] = [];

        positions.forEach((value, carId) => {
            const position = value % 1;
            const point: CornerTimingPoint = { timeMs, position, sampleIndex };
            const state = this.cars.get(carId);
            if (!state) {
                this.cars.set(carId, { previous: point, start: null });
                return;
            }
            const before = state.previous;
            let delta = position - before.position;
            if (delta < -0.5) delta += 1;
            if (delta > 0.5) delta -= 1;
            // Small numerical error at exactly 0.1 lap is not a jump.
            if (delta < 0 || delta > 0.1 + 1e-9 || timeMs - before.timeMs > 2000) {
                this.cars.set(carId, { previous: point, start: null });
                return;
            }
            if (delta > 0 && position < before.position) {
                const crossing: CornerTimingPoint = {
                    timeMs: Math.round(before.timeMs + (1 - before.position) / delta * (timeMs - before.timeMs)),
                    position: 0,
                    sampleIndex,
                };
                if (state.start && crossing.timeMs > state.start.timeMs) records.push({
                    id: `${carId}:lap:${sampleIndex}`, carId, isPlayer: carId === playerKey,
                    start: state.start, end: crossing, lapTimeMs: crossing.timeMs - state.start.timeMs,
                });
                state.start = crossing;
            }
            state.previous = point;
        });
        return records;
    }
}
