import {
    ACC_CONTINUITY_FIELDS,
    classifyAccSessionContinuity,
} from './acc-session-continuity';

const previousSample = {
    Static_track: 'Monza',
    Graphics_session_time_left: 900,
    Static_car_model: 'Ferrari 296 GT3',
    Graphics_completed_lap: 3,
    Graphics_current_time: 45_000,
    Graphics_distance_traveled: 12_000,
    Graphics_used_fuel: 18,
};

const continuingSample = {
    ...previousSample,
    Graphics_session_time_left: 899,
    Graphics_current_time: 45_016,
    Graphics_distance_traveled: 12_003,
    Graphics_used_fuel: 18.01,
};

describe('ACC session continuity classification', () => {
    it('preserves continuity when identity matches and progression remains monotonic', () => {
        expect(classifyAccSessionContinuity(previousSample, continuingSample)).toEqual({
            continuityBroken: false,
            reason: 'continuity preserved',
        });
    });

    it('allows current lap time to reset when the completed-lap count increases', () => {
        expect(classifyAccSessionContinuity(previousSample, {
            ...continuingSample,
            Graphics_completed_lap: 4,
            Graphics_current_time: 100,
        }).continuityBroken).toBe(false);
    });

    it.each([
        ['Static_track', 'Spa'],
        ['Graphics_session_time_left', 901],
        ['Static_car_model', 'Porsche 992 GT3 R'],
        ['Graphics_completed_lap', 2],
        ['Graphics_current_time', 44_999],
        ['Graphics_distance_traveled', 11_999],
        ['Graphics_used_fuel', 17.99],
    ] as const)('splits when %s independently breaks continuity', (field, value) => {
        const result = classifyAccSessionContinuity(previousSample, {
            ...continuingSample,
            [field]: value,
        });

        expect(result.continuityBroken).toBe(true);
        expect(result.reason).toContain(field);
    });

    it.each(['previous', 'current'] as const)(
        'splits when any required field is missing from the %s sample',
        (sampleName) => {
            for (const field of ACC_CONTINUITY_FIELDS) {
                const previous = { ...previousSample } as Record<string, unknown>;
                const current = { ...continuingSample } as Record<string, unknown>;
                delete (sampleName === 'previous' ? previous : current)[field];

                const result = classifyAccSessionContinuity(previous, current);

                expect(result.continuityBroken).toBe(true);
                expect(result.reason).toContain(`${sampleName}.${field}`);
            }
        },
    );

    it.each(['previous', 'current'] as const)(
        'splits on invalid track or car identity in the %s sample',
        (sampleName) => {
            for (const field of ['Static_track', 'Static_car_model'] as const) {
                for (const invalidValue of ['', '   ', 123]) {
                    const previous = { ...previousSample } as Record<string, unknown>;
                    const current = { ...continuingSample } as Record<string, unknown>;
                    (sampleName === 'previous' ? previous : current)[field] = invalidValue;

                    expect(classifyAccSessionContinuity(previous, current).continuityBroken).toBe(true);
                }
            }
        },
    );

    it.each(['previous', 'current'] as const)(
        'splits on nonnumeric or non-finite progression values in the %s sample',
        (sampleName) => {
            const numericFields = ACC_CONTINUITY_FIELDS.filter(
                (field) => field !== 'Static_track' && field !== 'Static_car_model',
            );

            for (const field of numericFields) {
                for (const invalidValue of ['1', Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
                    const previous = { ...previousSample } as Record<string, unknown>;
                    const current = { ...continuingSample } as Record<string, unknown>;
                    (sampleName === 'previous' ? previous : current)[field] = invalidValue;

                    expect(classifyAccSessionContinuity(previous, current).continuityBroken).toBe(true);
                }
            }
        },
    );

    it.each([
        ['reset', { Physics_packed_id: 1 }],
        ['missing', {}],
    ])('ignores a %s Physics_packed_id', (_label, packetFields) => {
        const result = classifyAccSessionContinuity(
            { ...previousSample, Physics_packed_id: 500 },
            { ...continuingSample, ...packetFields },
        );

        expect(result.continuityBroken).toBe(false);
    });
});
