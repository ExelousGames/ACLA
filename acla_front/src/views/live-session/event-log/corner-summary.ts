import type { DriverCornerRecord } from './CornerRecorder';

export interface DriverCornerSummary {
    carId: string;
    isPlayer: boolean;
    cornerId: string;
    cornerName: string;
    sampleCount: number;
    includedCount: number;
    averageTimeMs: number | null;
}

/** Average each driver's corner times after trimming 12.5% from each end. */
export function summarizeDriverCorners(records: DriverCornerRecord[]): DriverCornerSummary[] {
    const cars = new Map<string, Map<string, { summary: DriverCornerSummary; times: number[] }>>();
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
            }, times: [] };
            corners.set(record.cornerId, corner);
        }
        corner.summary.isPlayer ||= record.isPlayer;
        if (record.cornerTimeMs !== null && Number.isFinite(record.cornerTimeMs) && record.cornerTimeMs > 0) {
            corner.times.push(record.cornerTimeMs);
        }
    });

    return Array.from(cars.values()).flatMap((corners) => Array.from(corners.values(), ({ summary, times }) => {
        summary.sampleCount = times.length;
        // Whole samples: round down, but always exclude both extremes. Three passes are the minimum.
        if (times.length >= 3) {
            times.sort((a, b) => a - b);
            const trimCount = Math.max(1, Math.floor(times.length * 0.125));
            const middle = times.slice(trimCount, times.length - trimCount);
            summary.includedCount = middle.length;
            summary.averageTimeMs = middle.reduce((sum, time) => sum + time, 0) / middle.length;
        }
        return summary;
    }));
}
