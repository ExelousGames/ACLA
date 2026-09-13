export const ACC_CONTINUITY_FIELDS = [
    'Static_track',
    'Graphics_session_time_left',
    'Static_car_model',
    'Graphics_completed_lap',
    'Graphics_current_time',
    'Graphics_distance_traveled',
    'Graphics_used_fuel',
] as const;

type AccContinuityField = typeof ACC_CONTINUITY_FIELDS[number];

type AccContinuitySample = {
    Static_track: string;
    Graphics_session_time_left: number;
    Static_car_model: string;
    Graphics_completed_lap: number;
    Graphics_current_time: number;
    Graphics_distance_traveled: number;
    Graphics_used_fuel: number;
};

export type AccSessionContinuityResult = {
    continuityBroken: boolean;
    reason: string;
};

const identityFields = ['Static_track', 'Static_car_model'] as const;
const numericFields = [
    'Graphics_session_time_left',
    'Graphics_completed_lap',
    'Graphics_current_time',
    'Graphics_distance_traveled',
    'Graphics_used_fuel',
] as const;

const parseSample = (
    value: unknown,
    sampleName: 'previous' | 'current',
): { sample: AccContinuitySample } | { reason: string } => {
    const record = value && typeof value === 'object'
        ? value as Record<string, unknown>
        : {};

    for (const field of identityFields) {
        const fieldValue = record[field];
        if (typeof fieldValue !== 'string' || fieldValue.trim().length === 0) {
            return { reason: `${sampleName}.${field} is missing or invalid` };
        }
    }

    for (const field of numericFields) {
        const fieldValue = record[field];
        if (typeof fieldValue !== 'number' || !Number.isFinite(fieldValue)) {
            return { reason: `${sampleName}.${field} is missing or invalid` };
        }
    }

    return { sample: record as AccContinuitySample };
};

const broken = (reason: string): AccSessionContinuityResult => ({
    continuityBroken: true,
    reason,
});

export const classifyAccSessionContinuity = (
    previousValue: unknown,
    currentValue: unknown,
): AccSessionContinuityResult => {
    const previousResult = parseSample(previousValue, 'previous');
    if ('reason' in previousResult) {
        return broken(previousResult.reason);
    }

    const currentResult = parseSample(currentValue, 'current');
    if ('reason' in currentResult) {
        return broken(currentResult.reason);
    }

    const previous = previousResult.sample;
    const current = currentResult.sample;

    if (current.Static_track !== previous.Static_track) {
        return broken('Static_track changed');
    }
    if (current.Graphics_session_time_left > previous.Graphics_session_time_left) {
        return broken('Graphics_session_time_left increased');
    }
    if (current.Static_car_model !== previous.Static_car_model) {
        return broken('Static_car_model changed');
    }
    if (current.Graphics_completed_lap < previous.Graphics_completed_lap) {
        return broken('Graphics_completed_lap decreased');
    }
    if (
        current.Graphics_completed_lap === previous.Graphics_completed_lap
        && current.Graphics_current_time < previous.Graphics_current_time
    ) {
        return broken('Graphics_current_time decreased without a completed lap');
    }
    if (current.Graphics_distance_traveled < previous.Graphics_distance_traveled) {
        return broken('Graphics_distance_traveled decreased');
    }
    if (current.Graphics_used_fuel < previous.Graphics_used_fuel) {
        return broken('Graphics_used_fuel decreased');
    }

    return {
        continuityBroken: false,
        reason: 'continuity preserved',
    };
};
