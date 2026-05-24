import { SessionEvent, TelemetrySample } from '../types';
import { CornerDefinition, getCornerAtPosition } from '../track-corners';

export class CornerSensor {
    private corners: CornerDefinition[] = [];
    private activeCorner: CornerDefinition | null = null;
    private enterSampleIdx: number = 0;
    private enterLap: number = 0;
    private enterPosition: number = 0;

    setCorners(corners: CornerDefinition[]): void {
        this.corners = corners;
    }

    tick(sample: TelemetrySample, sampleIdx: number): SessionEvent | null {
        const pos: number = sample['Graphics_normalized_car_position'] ?? 0;
        const lap: number = sample['Graphics_completed_laps'] ?? 0;
        const now = Date.now();

        const corner = getCornerAtPosition(this.corners, pos);

        if (!this.activeCorner && corner) {
            // Entered a corner
            this.activeCorner = corner;
            this.enterSampleIdx = sampleIdx;
            this.enterLap = lap;
            this.enterPosition = pos;
            return null;
        }

        if (this.activeCorner && !corner) {
            // Exited the corner — emit the range event
            const event: SessionEvent = {
                id: `corner-${sampleIdx}`,
                type: 'CORNER',
                startSampleIdx: this.enterSampleIdx,
                endSampleIdx: sampleIdx,
                lap: this.enterLap,
                trackPosition: this.enterPosition,
                timestamp: now,
                metadata: { name: this.activeCorner.name },
            };
            this.activeCorner = null;
            return event;
        }

        return null;
    }

    reset(): void {
        this.activeCorner = null;
    }
}
