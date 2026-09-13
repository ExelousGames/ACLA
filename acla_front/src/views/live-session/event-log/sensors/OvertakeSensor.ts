import { SessionEvent, TelemetrySample } from 'views/lap-analysis/session-intelligence/types';

const DEBOUNCE_TICKS = 10;
const LOOKBACK_SAMPLES = 100;
const LOOKAHEAD_SAMPLES = 60;

export class OvertakeSensor {
    private prevPosition: number = -1;
    private pendingPosition: number = -1;
    private pendingTicks: number = 0;
    private pendingLap: number = 0;
    private pendingTrackPosition: number = 0;
    private pendingStartIdx: number = 0;

    tick(sample: TelemetrySample, sampleIdx: number): SessionEvent | null {
        const racePosition: number = sample.Graphics_position ?? -1;
        const lap: number = sample.Graphics_completed_laps ?? 0;
        const trackPosition: number = sample.Graphics_normalized_car_position ?? 0;

        if (racePosition < 0 || this.prevPosition < 0) {
            this.prevPosition = racePosition;
            return null;
        }

        if (racePosition < this.prevPosition) {
            if (this.pendingPosition !== racePosition) {
                this.pendingPosition = racePosition;
                this.pendingTicks = 1;
                this.pendingLap = lap;
                this.pendingTrackPosition = trackPosition;
                this.pendingStartIdx = sampleIdx;
            } else {
                this.pendingTicks += 1;
            }

            if (this.pendingTicks >= DEBOUNCE_TICKS) {
                const event: SessionEvent = {
                    id: `overtake-${sampleIdx}`,
                    type: 'OVERTAKE',
                    startSampleIdx: Math.max(0, this.pendingStartIdx - LOOKBACK_SAMPLES),
                    endSampleIdx: sampleIdx + LOOKAHEAD_SAMPLES,
                    lap: this.pendingLap,
                    trackPosition: this.pendingTrackPosition,
                    timestamp: Date.now(),
                    metadata: { positionBefore: this.prevPosition, positionAfter: this.pendingPosition },
                };
                this.prevPosition = this.pendingPosition;
                this.pendingPosition = -1;
                this.pendingTicks = 0;
                return event;
            }
        } else {
            if (racePosition !== this.pendingPosition) {
                this.pendingPosition = -1;
                this.pendingTicks = 0;
            }
            this.prevPosition = racePosition;
        }

        return null;
    }

    reset(): void {
        this.prevPosition = -1;
        this.pendingPosition = -1;
        this.pendingTicks = 0;
    }
}
