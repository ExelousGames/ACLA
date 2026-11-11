import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import type { StandardTelemetrySample } from '../live-session-types';
import { referencePositionAhead, referenceTravelDistance, REFERENCE_LOOKAHEAD_MS } from './phrase-reference-guidance';

const MIN_SAMPLE_INTERVAL_MS = 100;
const PREDICTION_FRAME_INTERVAL_MS = 100;
const boundedTime = (time: number) => Math.max(0, Math.min(REFERENCE_LOOKAHEAD_MS, time));
export const BRAKE_RELEASE_RATE_THRESHOLD = 0.01;
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const wrap = (value: number, period: number) => value >= 0 && value < period ? value : ((value % period) + period) % period;
const difference = (from: number, to: number, period?: number) => period === undefined ? to - from
    : wrap(to - from + period / 2, period) - period / 2;
const motionFields = [
    'Physics_steer_angle', 'Physics_speed_kmh',
    'Physics_velocity_x', 'Physics_velocity_y', 'Physics_velocity_z',
    'Physics_local_velocity_x', 'Physics_local_velocity_y', 'Physics_local_velocity_z',
    'Physics_heading', 'Physics_pitch', 'Physics_roll',
] as const;
const predictionFields = [...motionFields, 'Physics_brake', 'Physics_gas', 'Graphics_normalized_car_position', 'x', 'y', 'z'] as const;
type PredictionField = typeof predictionFields[number];
type TimedSample = { sample: StandardTelemetrySample; at: number };

export interface PhrasePredictionFrame {
    readonly offsetMs: number;
    readonly intendedTelemetry: StandardTelemetrySample;
    readonly deceleration: number | undefined;
    readonly brakeReleasing: boolean | undefined;
}

function playerPosition(sample: StandardTelemetrySample) {
    const id = sample.Graphics_player_car_id;
    const index = id === undefined ? -1 : sample.Graphics_car_id?.indexOf(id) ?? -1;
    return index < 0 ? undefined : sample.Graphics_car_coordinates?.[index];
}

function readField(sample: StandardTelemetrySample, field: PredictionField, playerId?: number): number | undefined {
    if (field === 'x' || field === 'y' || field === 'z') {
        return sample.Graphics_player_car_id === playerId ? playerPosition(sample)?.[field] : undefined;
    }
    const value = sample[field];
    if (!finite(value)) return undefined;
    if (field === 'Physics_speed_kmh' && value < 0) return undefined;
    if ((field === 'Physics_brake' || field === 'Physics_gas' || field === 'Graphics_normalized_car_position')
        && (value < 0 || value > 1)) return undefined;
    return value;
}

/** A second-order telemetry preview shared by closure entry, children and exit. */
export class PhraseLookahead {
    private history: TimedSample[] = [];
    private rates = new Map<PredictionField, number>();
    private curvatures = new Map<PredictionField, number>();
    private at = 0;
    private forecast?: { sample: StandardTelemetrySample; lap: TopLapReferenceEntry | null;
        frames: readonly PhrasePredictionFrame[]; byTime: Map<number, PhrasePredictionFrame> };

    reset() { this.history = []; this.rates.clear(); this.curvatures.clear(); this.forecast = undefined; }

    update(sample: StandardTelemetrySample, now: number, continuous: boolean, lap: TopLapReferenceEntry | null = null) {
        this.forecast = undefined;
        const elapsed = now - this.at;
        if (!continuous || elapsed <= 0) this.reset();
        this.at = now;
        let previous: TimedSample | undefined = this.history[this.history.length - 1];
        const from = previous?.sample.Graphics_normalized_car_position, to = sample.Graphics_normalized_car_position;
        const speed = sample.Physics_speed_kmh, previousSpeed = previous?.sample.Physics_speed_kmh;
        if (previous && finite(from) && finite(to) && finite(speed) && finite(previousSpeed)) {
            const travelled = referenceTravelDistance(lap, from, to);
            const expected = (previousSpeed + speed) / 7.2 * (now - previous.at) / 1000;
            // Frozen graphics or a teleport cannot supply derivatives for live motion.
            if ((from === to && expected > 0 && now - previous.at >= MIN_SAMPLE_INTERVAL_MS) || (travelled !== undefined
                && Math.abs(travelled - expected) > Math.max(3, expected * 0.5))) {
                this.reset();
                previous = undefined;
            }
        }
        // Separate derivative anchors so physics/graphics packet jitter is not amplified.
        if (!previous || now - previous.at >= MIN_SAMPLE_INTERVAL_MS) {
            this.history.push({ sample: { ...sample,
                Graphics_car_id: sample.Graphics_car_id?.slice(),
                Graphics_car_coordinates: sample.Graphics_car_coordinates?.map((point) => ({ ...point })),
            }, at: now });
            if (this.history.length > 3) this.history.shift();
        }
        const anchors = this.history.filter(({ at }) => now - at >= MIN_SAMPLE_INTERVAL_MS);
        const anchor = anchors[anchors.length - 1], older = anchors[anchors.length - 2];
        predictionFields.forEach((field) => {
            const current = readField(sample, field, sample.Graphics_player_car_id);
            const before = anchor && readField(anchor.sample, field, sample.Graphics_player_car_id);
            if (!anchor || !finite(current) || !finite(before)) {
                this.rates.delete(field);
                this.curvatures.delete(field);
                return;
            }
            const atLimit = ((field === 'Physics_brake' || field === 'Physics_gas') && (current === 0 || current === 1))
                || (field === 'Physics_steer_angle' && Math.abs(current) === 1)
                || (field === 'Physics_speed_kmh' && current === 0);
            if (current === before && atLimit) {
                // A held physical limit must not turn prior curvature into an invented rebound.
                this.rates.set(field, 0);
                this.curvatures.set(field, 0);
                return;
            }
            const period = field === 'Graphics_normalized_car_position' ? 1
                : field === 'Physics_heading' || field === 'Physics_pitch' || field === 'Physics_roll' ? 2 * Math.PI : undefined;
            const interval = (now - anchor.at) / 1000;
            let rate = difference(before, current, period) / interval;
            const oldest = older && readField(older.sample, field, sample.Graphics_player_car_id);
            let curvature: number | undefined;
            if (older && finite(oldest)) {
                const previousInterval = (anchor.at - older.at) / 1000;
                const previousRate = difference(oldest, before, period) / previousInterval;
                // Secants describe interval midpoints, even when the intervals differ.
                const secondDerivative = 2 * (rate - previousRate) / (interval + previousInterval);
                if (finite(secondDerivative)) {
                    curvature = secondDerivative;
                    rate += curvature * interval / 2;
                }
            }
            if (!finite(rate)) { this.rates.delete(field); this.curvatures.delete(field); return; }
            this.rates.set(field, rate);
            if (curvature === undefined) this.curvatures.delete(field);
            else this.curvatures.set(field, curvature);
        });
    }

    /** Retain the complete three-second preview until telemetry or the reference changes. */
    getFrames(sample: StandardTelemetrySample, lap: TopLapReferenceEntry | null): readonly PhrasePredictionFrame[] {
        if (this.forecast?.sample !== sample || this.forecast.lap !== lap) {
            const frames = Array.from({ length: REFERENCE_LOOKAHEAD_MS / PREDICTION_FRAME_INTERVAL_MS + 1 }, (_, index) =>
                this.createFrame(sample, lap, index * PREDICTION_FRAME_INTERVAL_MS));
            this.forecast = { sample, lap, frames, byTime: new Map(frames.map((frame) => [frame.offsetMs, frame])) };
        }
        return this.forecast.frames;
    }

    /** Exact preparation boundaries share the buffer without rounding to its sampling interval. */
    getFrame(sample: StandardTelemetrySample, lap: TopLapReferenceEntry | null, offsetMs: number): PhrasePredictionFrame {
        this.getFrames(sample, lap);
        const time = boundedTime(offsetMs), byTime = this.forecast!.byTime;
        let frame = byTime.get(time);
        if (!frame) {
            frame = this.createFrame(sample, lap, time);
            byTime.set(time, frame);
        }
        return frame;
    }

    private createFrame(sample: StandardTelemetrySample, lap: TopLapReferenceEntry | null, offsetMs: number): PhrasePredictionFrame {
        return { offsetMs, intendedTelemetry: this.project(sample, lap, offsetMs),
            deceleration: this.projectDeceleration(sample, offsetMs), brakeReleasing: this.projectBrakeReleasing(sample, offsetMs) };
    }

    private prediction(sample: StandardTelemetrySample, field: PredictionField, secondsAhead: number) {
        const current = readField(sample, field, sample.Graphics_player_car_id), rate = this.rates.get(field);
        if (!finite(current) || rate === undefined) return undefined;
        const curvature = this.curvatures.get(field) ?? 0;
        const value = current + rate * secondsAhead + curvature * secondsAhead ** 2 / 2;
        const futureRate = rate + curvature * secondsAhead;
        return finite(value) && finite(futureRate) ? { value, rate: futureRate } : undefined;
    }

    /** Speed's derivative at the shared horizon, in m/s², positive while slowing. */
    projectDeceleration(sample: StandardTelemetrySample, lookaheadMs = REFERENCE_LOOKAHEAD_MS): number | undefined {
        const motion = this.prediction(sample, 'Physics_speed_kmh', boundedTime(lookaheadMs) / 1000);
        if (!motion) return undefined;
        // Match the speed preview's zero floor: a stopped car cannot keep decelerating.
        return motion.value <= 0 ? 0 : -motion.rate / 3.6;
    }

    /** Detect pedal release from its derivative at the same horizon as intended pressure. */
    projectBrakeReleasing(sample: StandardTelemetrySample, lookaheadMs = REFERENCE_LOOKAHEAD_MS): boolean | undefined {
        const motion = this.prediction(sample, 'Physics_brake', boundedTime(lookaheadMs) / 1000);
        if (!motion) return undefined;
        // Clamped pressure cannot keep falling below zero or change while above full pressure.
        return motion.value > 0 && motion.value <= 1 && -motion.rate > BRAKE_RELEASE_RATE_THRESHOLD;
    }

    project(sample: StandardTelemetrySample, lap: TopLapReferenceEntry | null, lookaheadMs = REFERENCE_LOOKAHEAD_MS): StandardTelemetrySample {
        const secondsAhead = boundedTime(lookaheadMs) / 1000;
        const intended = { ...sample };
        const predict = (field: PredictionField) => this.prediction(sample, field, secondsAhead)?.value;
        motionFields.forEach((field) => {
            const value = predict(field);
            if (value !== undefined) intended[field] = field === 'Physics_speed_kmh' ? Math.max(0, value)
                : field === 'Physics_steer_angle' ? Math.max(-1, Math.min(1, value)) : value;
        });
        (['Physics_brake', 'Physics_gas'] as const).forEach((field) => {
            const value = predict(field);
            if (value !== undefined) intended[field] = Math.max(0, Math.min(1, value));
        });
        const position = predict('Graphics_normalized_car_position');
        const validSpeed = finite(sample.Physics_speed_kmh) && sample.Physics_speed_kmh >= 0;
        intended.Graphics_normalized_car_position = !validSpeed ? undefined
            : sample.Physics_speed_kmh === 0 ? sample.Graphics_normalized_car_position
                : position === undefined ? referencePositionAhead(sample, lap, lookaheadMs) : wrap(position, 1);
        const player = playerPosition(sample);
        if (player && sample.Graphics_car_coordinates) {
            const projected = { ...player };
            (['x', 'y', 'z'] as const).forEach((axis) => {
                const value = predict(axis);
                if (value !== undefined) projected[axis] = value;
            });
            intended.Graphics_car_coordinates = sample.Graphics_car_coordinates.map((point) => point === player ? projected : { ...point });
        }
        return intended;
    }
}
