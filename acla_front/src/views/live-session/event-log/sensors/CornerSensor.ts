import { CornerDefinition, SessionEvent, TelemetrySample } from 'views/lap-analysis/session-intelligence/types';
import { getCornerAtPosition } from 'views/lap-analysis/session-intelligence/track-corners';

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
        const position: number = sample.Graphics_normalized_car_position ?? 0;
        const lap: number = sample.Graphics_completed_laps ?? 0;
        const corner = getCornerAtPosition(this.corners, position);

        if (!this.activeCorner && corner) {
            this.activeCorner = corner;
            this.enterSampleIdx = sampleIdx;
            this.enterLap = lap;
            this.enterPosition = position;
            return null;
        }

        if (this.activeCorner && !corner) {
            const event: SessionEvent = {
                id: `corner-${sampleIdx}`,
                type: 'CORNER',
                startSampleIdx: this.enterSampleIdx,
                endSampleIdx: sampleIdx,
                lap: this.enterLap,
                trackPosition: this.enterPosition,
                timestamp: Date.now(),
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
