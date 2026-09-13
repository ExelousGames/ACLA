const FINISH_LINE_BACKWARD_JUMP = 0.5;
const MINIMUM_TIME_STEP_MS = 1e-6;

export interface UnwrappedLapTelemetrySequence {
    timesMs: number[];
    positions: number[];
}

/**
 * Unwraps validated track positions and lap-relative clocks at finish-line
 * crossings, preserving source order and clock values elsewhere.
 */
export const unwrapLapTelemetrySequence = (
    timesMs: readonly number[],
    normalizedPositions: readonly number[],
): UnwrappedLapTelemetrySequence => {
    const unwrappedTimesMs: number[] = [];
    const unwrappedPositions: number[] = [];
    let lapOffset = 0;
    let timeOffsetMs = 0;
    let previousRawTimeMs: number | undefined;
    let previousUnwrappedTimeMs: number | undefined;
    let previousNormalizedPosition: number | undefined;

    for (let index = 0; index < timesMs.length; index += 1) {
        const rawTimeMs = timesMs[index];
        const normalizedPosition = normalizedPositions[index];
        const crossedFinishLine = (
            previousNormalizedPosition !== undefined
            && previousNormalizedPosition - normalizedPosition > FINISH_LINE_BACKWARD_JUMP
        );
        if (crossedFinishLine) lapOffset += 1;

        let unwrappedTimeMs = rawTimeMs + timeOffsetMs;
        if (
            crossedFinishLine
            && previousRawTimeMs !== undefined
            && rawTimeMs <= previousRawTimeMs
            && previousUnwrappedTimeMs !== undefined
        ) {
            // The source clock is lap-relative. Continue after the last observed
            // pre-line sample while retaining time already elapsed in the new lap.
            timeOffsetMs = previousUnwrappedTimeMs;
            unwrappedTimeMs = rawTimeMs + timeOffsetMs;
            if (unwrappedTimeMs <= previousUnwrappedTimeMs) {
                unwrappedTimeMs = previousUnwrappedTimeMs + MINIMUM_TIME_STEP_MS;
                timeOffsetMs = unwrappedTimeMs - rawTimeMs;
            }
        }

        unwrappedTimesMs.push(unwrappedTimeMs);
        unwrappedPositions.push(normalizedPosition + lapOffset);
        previousRawTimeMs = rawTimeMs;
        previousUnwrappedTimeMs = unwrappedTimeMs;
        previousNormalizedPosition = normalizedPosition;
    }

    return { timesMs: unwrappedTimesMs, positions: unwrappedPositions };
};
