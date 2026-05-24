import { SessionEvent, TelemetrySample } from '../types';

const DEBOUNCE_TICKS = 10;      // position must hold for N ticks before confirming
const LOOKBACK_SAMPLES = 100;   // ~5 sec at 20Hz — captures the approach
const LOOKAHEAD_SAMPLES = 60;   // ~3 sec — captures the pull-away

export class OvertakeSensor {
    private prevPosition: number = -1;        // race position integer
    private pendingPosition: number = -1;
    private pendingTicks: number = 0;
    private pendingLap: number = 0;
    private pendingTrackPos: number = 0;
    private pendingStartIdx: number = 0;

    tick(sample: TelemetrySample, sampleIdx: number): SessionEvent | null {
        const racePos: number = sample['Graphics_position'] ?? -1;
        const lap: number = sample['Graphics_completed_laps'] ?? 0;
        const trackPos: number = sample['Graphics_normalized_car_position'] ?? 0;

        if (racePos < 0 || this.prevPosition < 0) {
            this.prevPosition = racePos;
            return null;
        }

        // Position number decreased → gained a place (overtake)
        if (racePos < this.prevPosition) {
            if (this.pendingPosition !== racePos) {
                // New candidate
                this.pendingPosition = racePos;
                this.pendingTicks = 1;
                this.pendingLap = lap;
                this.pendingTrackPos = trackPos;
                this.pendingStartIdx = sampleIdx;
            } else {
                this.pendingTicks++;
            }

            if (this.pendingTicks >= DEBOUNCE_TICKS) {
                const confirmed = this.pendingPosition;
                const event: SessionEvent = {
                    id: `overtake-${sampleIdx}`,
                    type: 'OVERTAKE',
                    startSampleIdx: Math.max(0, this.pendingStartIdx - LOOKBACK_SAMPLES),
                    endSampleIdx: sampleIdx + LOOKAHEAD_SAMPLES,
                    lap: this.pendingLap,
                    trackPosition: this.pendingTrackPos,
                    timestamp: Date.now(),
                    metadata: { positionBefore: this.prevPosition, positionAfter: confirmed },
                };
                this.prevPosition = confirmed;
                this.pendingPosition = -1;
                this.pendingTicks = 0;
                return event;
            }
        } else {
            // Position stabilised or went back — reset pending
            if (racePos !== this.pendingPosition) {
                this.pendingPosition = -1;
                this.pendingTicks = 0;
            }
            this.prevPosition = racePos;
        }

        return null;
    }

    reset(): void {
        this.prevPosition = -1;
        this.pendingPosition = -1;
        this.pendingTicks = 0;
    }
}
