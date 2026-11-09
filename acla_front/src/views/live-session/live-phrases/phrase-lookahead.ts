import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import type { StandardTelemetrySample } from '../live-session-types';
import { referencePositionAhead, referenceTravelDistance, REFERENCE_LOOKAHEAD_MS } from './phrase-reference-guidance';

const MIN_SAMPLE_INTERVAL_MS = 100;
export const BRAKE_RELEASE_RATE_THRESHOLD = 0.01;
const secondsAhead = REFERENCE_LOOKAHEAD_MS / 1000;
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
type TimedSample = { sample: StandardTelemetrySample; at: number };

function playerPosition(sample: StandardTelemetrySample) {
    const id = sample.Graphics_player_car_id;
    const index = id === undefined ? -1 : sample.Graphics_car_id?.indexOf(id) ?? -1;
    return index < 0 ? undefined : sample.Graphics_car_coordinates?.[index];
}

/** A first-order telemetry preview shared by closure entry, children and exit. */
export class PhraseLookahead {
    private history: TimedSample[] = [];
    private at = 0;

    reset() { this.history = []; }

    update(sample: StandardTelemetrySample, now: number, continuous: boolean, lap: TopLapReferenceEntry | null = null) {
        if (!continuous) this.reset();
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
            if (this.history.length > 2) this.history.shift();
        }
    }

    private derivative(sample: StandardTelemetrySample, read: (value: StandardTelemetrySample) => number | undefined, period?: number) {
        const [previous] = this.history.filter(({ at }) => this.at - at >= MIN_SAMPLE_INTERVAL_MS).slice(-1);
        const current = read(sample), before = previous && read(previous.sample);
        if (!previous || !finite(current) || !finite(before)) return undefined;
        const elapsed = (this.at - previous.at) / 1000;
        return { current, rate: difference(before, current, period) / elapsed };
    }

    /** Speed's derivative at the shared horizon, in m/s², positive while slowing. */
    projectDeceleration(sample: StandardTelemetrySample): number | undefined {
        const motion = this.derivative(sample, (value) => {
            const speed = value.Physics_speed_kmh;
            return finite(speed) && speed >= 0 ? speed : undefined;
        });
        if (!motion) return undefined;
        const { current, rate } = motion;
        const speed = current + rate * secondsAhead;
        const deceleration = -rate / 3.6;
        if (!finite(speed) || !finite(deceleration)) return undefined;
        // Match the speed preview's zero floor: a stopped car cannot keep decelerating.
        return speed <= 0 ? 0 : deceleration;
    }

    /** Detect pedal release from its derivative at the same horizon as intended pressure. */
    projectBrakeReleasing(sample: StandardTelemetrySample): boolean | undefined {
        const motion = this.derivative(sample, (value) => {
            const pressure = value.Physics_brake;
            return finite(pressure) && pressure >= 0 && pressure <= 1 ? pressure : undefined;
        });
        if (!motion) return undefined;
        const { current, rate } = motion;
        const pressure = current + rate * secondsAhead;
        const releaseRate = -rate;
        if (!finite(pressure) || !finite(releaseRate)) return undefined;
        // Clamped pressure cannot keep falling below zero or change while above full pressure.
        return pressure > 0 && pressure <= 1 && releaseRate > BRAKE_RELEASE_RATE_THRESHOLD;
    }

    project(sample: StandardTelemetrySample, lap: TopLapReferenceEntry | null): StandardTelemetrySample {
        const intended = { ...sample };
        const predict = (read: (value: StandardTelemetrySample) => number | undefined, period?: number): number | undefined => {
            const motion = this.derivative(sample, read, period);
            if (!motion) return undefined;
            const { current, rate } = motion;
            const value = current + rate * secondsAhead;
            return finite(value) ? value : undefined;
        };
        motionFields.forEach((field) => {
            const angle = field === 'Physics_heading' || field === 'Physics_pitch' || field === 'Physics_roll';
            const value = predict((value) => value[field], angle ? 2 * Math.PI : undefined);
            if (value !== undefined) intended[field] = field === 'Physics_speed_kmh' ? Math.max(0, value)
                : field === 'Physics_steer_angle' ? Math.max(-1, Math.min(1, value)) : value;
        });
        (['Physics_brake', 'Physics_gas'] as const).forEach((field) => {
            const value = predict((sample) => {
                const pressure = sample[field];
                return finite(pressure) && pressure >= 0 && pressure <= 1 ? pressure : undefined;
            });
            if (value !== undefined) intended[field] = Math.max(0, Math.min(1, value));
        });
        const position = predict((value) => {
            const position = value.Graphics_normalized_car_position;
            return finite(position) && position >= 0 && position <= 1 ? position : undefined;
        }, 1);
        const validSpeed = finite(sample.Physics_speed_kmh) && sample.Physics_speed_kmh >= 0;
        intended.Graphics_normalized_car_position = !validSpeed ? undefined
            : sample.Physics_speed_kmh === 0 ? sample.Graphics_normalized_car_position
                : position === undefined ? referencePositionAhead(sample, lap) : wrap(position, 1);
        const player = playerPosition(sample);
        if (player && sample.Graphics_car_coordinates) {
            const projected = { ...player };
            (['x', 'y', 'z'] as const).forEach((axis) => {
                const value = predict((value) => value.Graphics_player_car_id === sample.Graphics_player_car_id
                    ? playerPosition(value)?.[axis] : undefined);
                if (value !== undefined) projected[axis] = value;
            });
            intended.Graphics_car_coordinates = sample.Graphics_car_coordinates.map((point) => point === player ? projected : { ...point });
        }
        return intended;
    }
}
