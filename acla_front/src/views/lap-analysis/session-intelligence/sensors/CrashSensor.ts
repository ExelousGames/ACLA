import { SessionEvent, TelemetrySample } from '../types';

const SPEED_DROP_THRESHOLD = 40;   // km/h lost in one tick to qualify as crash
const GFORCE_THRESHOLD = 4.5;      // lateral G to qualify as crash impact
const LOOKBACK_SAMPLES = 30;       // samples before detection to include in range
const LOOKAHEAD_SAMPLES = 20;      // samples after detection to include in range
const COOLDOWN_SAMPLES = 100;      // ignore further crashes for N samples after one

export class CrashSensor {
    private prevSpeed: number = -1;
    private cooldown: number = 0;
    private crashStartIdx: number = 0;
    private crashLap: number = 0;
    private crashPos: number = 0;

    tick(sample: TelemetrySample, sampleIdx: number): SessionEvent | null {
        if (this.cooldown > 0) {
            this.cooldown--;
            return null;
        }

        const speed: number = sample['Physics_speed_kmh'] ?? 0;
        const gx: number = Math.abs(sample['Physics_g_force_x'] ?? 0);
        const gy: number = Math.abs(sample['Physics_g_force_y'] ?? 0);
        const lap: number = sample['Graphics_completed_laps'] ?? 0;
        const pos: number = sample['Graphics_normalized_car_position'] ?? 0;

        const speedDrop = this.prevSpeed >= 0 ? this.prevSpeed - speed : 0;
        const isCrash = speedDrop >= SPEED_DROP_THRESHOLD || gx >= GFORCE_THRESHOLD || gy >= GFORCE_THRESHOLD;

        this.prevSpeed = speed;

        if (isCrash) {
            this.cooldown = COOLDOWN_SAMPLES;
            const event: SessionEvent = {
                id: `crash-${sampleIdx}`,
                type: 'CRASHED',
                startSampleIdx: Math.max(0, sampleIdx - LOOKBACK_SAMPLES),
                endSampleIdx: sampleIdx + LOOKAHEAD_SAMPLES,
                lap,
                trackPosition: pos,
                timestamp: Date.now(),
                metadata: { speedDrop, gx, gy },
            };
            return event;
        }

        return null;
    }

    reset(): void {
        this.prevSpeed = -1;
        this.cooldown = 0;
    }
}
