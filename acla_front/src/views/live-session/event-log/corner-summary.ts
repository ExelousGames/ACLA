import type { DriverCornerRecord } from './CornerRecorder';
import { summarizeMeasurements, summarizeTimes } from './timing-summary';

export interface DriverCornerSummary {
    carId: string;
    isPlayer: boolean;
    cornerId: string;
    cornerName: string;
    sampleCount: number;
    includedCount: number;
    averageTimeMs: number | null;
    /** Lap fraction per second; trimmed independently from corner times. */
    averageExitSpeed: number | null;
    exitSpeedSampleCount: number;
    exitSpeedIncludedCount: number;
}

/** Average each driver's corner times and exit speeds, trimming each independently. */
export function summarizeDriverCorners(records: DriverCornerRecord[]): DriverCornerSummary[] {
    const cars = new Map<string, Map<string, { summary: DriverCornerSummary; times: number[]; exitSpeeds: number[] }>>();
    records.forEach((record) => {
        let corners = cars.get(record.carId);
        if (!corners) {
            corners = new Map();
            cars.set(record.carId, corners);
        }
        let corner = corners.get(record.cornerId);
        if (!corner) {
            corner = { summary: {
                carId: record.carId, isPlayer: record.isPlayer,
                cornerId: record.cornerId, cornerName: record.cornerName,
                sampleCount: 0, includedCount: 0, averageTimeMs: null,
                averageExitSpeed: null, exitSpeedSampleCount: 0, exitSpeedIncludedCount: 0,
            }, times: [], exitSpeeds: [] };
            corners.set(record.cornerId, corner);
        }
        corner.summary.isPlayer ||= record.isPlayer;
        if (record.cornerTimeMs !== null && Number.isFinite(record.cornerTimeMs) && record.cornerTimeMs > 0) {
            corner.times.push(record.cornerTimeMs);
        }
        if (record.exitSpeed !== null && Number.isFinite(record.exitSpeed) && record.exitSpeed > 0) {
            corner.exitSpeeds.push(record.exitSpeed);
        }
    });

    return Array.from(cars.values()).flatMap((corners) => Array.from(corners.values(), ({ summary, times, exitSpeeds }) => {
        const speed = summarizeMeasurements(exitSpeeds);
        return { ...summary, ...summarizeTimes(times), averageExitSpeed: speed.average,
            exitSpeedSampleCount: speed.sampleCount, exitSpeedIncludedCount: speed.includedCount };
    }));
}
