import { SessionEvent, TelemetrySample } from 'views/lap-analysis/session-intelligence/types';

const SPEED_DROP_THRESHOLD = 40;
const GFORCE_THRESHOLD = 4.5;
const LOOKBACK_SAMPLES = 30;
const LOOKAHEAD_SAMPLES = 20;
const COOLDOWN_SAMPLES = 100;

export class CrashSensor {
    private prevSpeed: number = -1;
    private cooldown: number = 0;

    tick(sample: TelemetrySample, sampleIdx: number): SessionEvent | null {
        if (this.cooldown > 0) {
            this.cooldown -= 1;
            return null;
        }

        const speed: number = sample.Physics_speed_kmh ?? 0;
        const gx: number = Math.abs(sample.Physics_g_force_x ?? 0);
        const gy: number = Math.abs(sample.Physics_g_force_y ?? 0);
        const speedDrop = this.prevSpeed >= 0 ? this.prevSpeed - speed : 0;
        const isCrash = speedDrop >= SPEED_DROP_THRESHOLD || gx >= GFORCE_THRESHOLD || gy >= GFORCE_THRESHOLD;

        this.prevSpeed = speed;
        if (!isCrash) return null;

        this.cooldown = COOLDOWN_SAMPLES;
        return {
            id: `crash-${sampleIdx}`,
            type: 'CRASHED',
            startSampleIdx: Math.max(0, sampleIdx - LOOKBACK_SAMPLES),
            endSampleIdx: sampleIdx + LOOKAHEAD_SAMPLES,
            lap: sample.Graphics_completed_laps ?? 0,
            trackPosition: sample.Graphics_normalized_car_position ?? 0,
            timestamp: Date.now(),
            metadata: { speedDrop, gx, gy },
        };
    }

    reset(): void {
        this.prevSpeed = -1;
        this.cooldown = 0;
    }
}
