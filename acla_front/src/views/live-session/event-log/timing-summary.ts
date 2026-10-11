export interface TimingSummary {
    sampleCount: number;
    includedCount: number;
    averageTimeMs: number | null;
}

/** Trim 12.5% from each end, rounding down but always excluding both extremes. */
export function summarizeMeasurements(values: Array<number | null>) {
    const times = values.filter((time): time is number => time !== null && Number.isFinite(time) && time > 0)
        .sort((a, b) => a - b);
    if (times.length < 3) return { sampleCount: times.length, includedCount: 0, average: null };
    const trimCount = Math.max(1, Math.floor(times.length * 0.125));
    const middle = times.slice(trimCount, times.length - trimCount);
    return {
        sampleCount: times.length,
        includedCount: middle.length,
        average: middle.reduce((sum, time) => sum + time, 0) / middle.length,
    };
}

export function summarizeTimes(values: Array<number | null>): TimingSummary {
    const { average, ...counts } = summarizeMeasurements(values);
    return { ...counts, averageTimeMs: average };
}
