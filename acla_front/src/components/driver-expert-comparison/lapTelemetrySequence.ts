const FINISH_LINE_BACKWARD_JUMP = 0.5;
const MINIMUM_TIME_STEP_MS = 1e-6;

export interface UnwrappedLapTelemetrySequence {
    timesMs: number[];
    positions: number[];
}

/**
 * Unwraps normalized track positions and lap-relative clocks into one continuous
 * segment timeline. A clock may reset only on the same sample that crosses the
 * finish line; other repeated or decreasing timestamps remain invalid.
 */
export const unwrapLapTelemetrySequence = (
    timesMs: readonly number[],
    normalizedPositions: readonly number[],
): UnwrappedLapTelemetrySequence | undefined => {
    if (!timesMs.length || timesMs.length !== normalizedPositions.length) return undefined;

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
        if (
            !Number.isFinite(rawTimeMs)
            || rawTimeMs < 0
            || !Number.isFinite(normalizedPosition)
            || normalizedPosition < 0
            || normalizedPosition > 1
        ) {
            return undefined;
        }

        let crossedFinishLine = false;
        if (
            previousNormalizedPosition !== undefined
            && normalizedPosition < previousNormalizedPosition
        ) {
            if (previousNormalizedPosition - normalizedPosition <= FINISH_LINE_BACKWARD_JUMP) {
                return undefined;
            }
            crossedFinishLine = true;
            lapOffset += 1;
        }

        let unwrappedTimeMs = rawTimeMs + timeOffsetMs;
        if (previousRawTimeMs !== undefined && rawTimeMs <= previousRawTimeMs) {
            if (!crossedFinishLine || previousUnwrappedTimeMs === undefined) return undefined;

            // The source clock is lap-relative. Continue after the last observed
            // pre-line sample while retaining time already elapsed in the new lap.
            timeOffsetMs = previousUnwrappedTimeMs;
            unwrappedTimeMs = rawTimeMs + timeOffsetMs;
            if (unwrappedTimeMs <= previousUnwrappedTimeMs) {
                unwrappedTimeMs = previousUnwrappedTimeMs + MINIMUM_TIME_STEP_MS;
                timeOffsetMs = unwrappedTimeMs - rawTimeMs;
            }
        }

        if (
            previousUnwrappedTimeMs !== undefined
            && unwrappedTimeMs <= previousUnwrappedTimeMs
        ) {
            return undefined;
        }

        unwrappedTimesMs.push(unwrappedTimeMs);
        unwrappedPositions.push(normalizedPosition + lapOffset);
        previousRawTimeMs = rawTimeMs;
        previousUnwrappedTimeMs = unwrappedTimeMs;
        previousNormalizedPosition = normalizedPosition;
    }

    return { timesMs: unwrappedTimesMs, positions: unwrappedPositions };
};
