import { SessionEvent, TelemetrySample } from './types';
import { EventLog } from './EventLog';
import { CornerSensor } from './sensors/CornerSensor';
import { CrashSensor } from './sensors/CrashSensor';
import { OvertakeSensor } from './sensors/OvertakeSensor';
import { getCornersForTrack } from './track-corners';

export class SensorManager {
    private cornerSensor = new CornerSensor();
    private crashSensor = new CrashSensor();
    private overtakeSensor = new OvertakeSensor();
    private onEvent: ((event: SessionEvent) => void) | null = null;

    setTrack(trackName: string): void {
        this.cornerSensor.setCorners(getCornersForTrack(trackName));
    }

    // Optional callback — fired for every event (e.g. to send an observation over WS).
    onEventEmitted(cb: (event: SessionEvent) => void): void {
        this.onEvent = cb;
    }

    tick(sample: TelemetrySample, sampleIdx: number, log: EventLog): void {
        const candidates = [
            this.cornerSensor.tick(sample, sampleIdx),
            this.crashSensor.tick(sample, sampleIdx),
            this.overtakeSensor.tick(sample, sampleIdx),
        ];

        for (const event of candidates) {
            if (event) {
                log.push(event);
                this.onEvent?.(event);
            }
        }
    }

    reset(): void {
        this.cornerSensor.reset();
        this.crashSensor.reset();
        this.overtakeSensor.reset();
    }
}
