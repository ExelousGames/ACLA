import { TelemetrySample } from './types';
import { EventLog } from './EventLog';
import { CornerSensor } from './sensors/CornerSensor';
import { CrashSensor } from './sensors/CrashSensor';
import { OvertakeSensor } from './sensors/OvertakeSensor';
import { getCornersForTrack } from './track-corners';

export class SensorManager {
    private cornerSensor = new CornerSensor();
    private crashSensor = new CrashSensor();
    private overtakeSensor = new OvertakeSensor();

    setTrack(trackName: string): void {
        this.cornerSensor.setCorners(getCornersForTrack(trackName));
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
            }
        }
    }

    reset(): void {
        this.cornerSensor.reset();
        this.crashSensor.reset();
        this.overtakeSensor.reset();
    }
}
