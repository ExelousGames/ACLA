import {
    adaptAnalysisResultsComparison,
    resolveAnalysisResultsComparison,
} from './analysisResultsComparisonAdapter';

const driverRow = (
    position: number,
    timeMs: number,
    player: { x: number; y?: number; z?: number },
    overrides: Record<string, unknown> = {},
) => ({
    Graphics_current_time: timeMs,
    Graphics_normalized_car_position: position,
    Graphics_car_id: [7, 42],
    Graphics_player_car_id: 42,
    Graphics_car_coordinates: [
        { x: -100, y: -200, z: -300 },
        player,
    ],
    Physics_gas: 0.5,
    Physics_brake: 0.1,
    Physics_gear: 3,
    ...overrides,
});

const expertRow = (
    position: number,
    timeMs: number,
    overrides: Record<string, unknown> = {},
) => ({
    expert_optimal_time: timeMs,
    Graphics_normalized_car_position: position,
    expert_optimal_player_pos_x: timeMs / 10,
    expert_optimal_player_pos_y: timeMs / 20,
    expert_optimal_player_pos_z: timeMs / -10,
    expert_optimal_throttle: 0.6,
    expert_optimal_brake: 0.2,
    expert_optimal_gear: 4,
    ...overrides,
});

describe('adaptAnalysisResultsComparison', () => {
    it('uses Expert source order and normalized positions when raw indexes differ or are absent', () => {
        const result = adaptAnalysisResultsComparison({
            baselineRecords: [
                driverRow(0.1, 100, { x: 10, y: 11, z: 12 }, { raw_index: 900 }),
                driverRow(0.2, 200, { x: 20, y: 21, z: 22 }),
                driverRow(0.3, 300, { x: 30, y: 31, z: 32 }, { raw_index: -1 }),
            ],
            expertReferenceData: [
                expertRow(0.1, 1_000, { raw_index: 80 }),
                expertRow(0.2, 1_100),
                expertRow(0.3, 1_200, { raw_index: 2 }),
            ],
        });

        expect(result.samples).toHaveLength(3);
        expect(result.samples.map((sample) => sample.driverTimeMs)).toEqual([100, 200, 300]);
        expect(result.samples.map((sample) => sample.expertTimeMs)).toEqual([1_000, 1_100, 1_200]);
        expect(result.samples.map((sample) => sample.driverTrackPosition)).toEqual([0.1, 0.2, 0.3]);
        expect(result.samples.map((sample) => sample.expertTrackPosition)).toEqual([0.1, 0.2, 0.3]);
        expect(result.samples[0]).toMatchObject({
            driverTrajectory: { x: 10, y: 11, z: 12 },
            expertTrajectory: { x: 100, y: 50, z: -100 },
            driverGas: 0.5,
            expertGas: 0.6,
        });
    });

    it('interpolates the Driver at an Expert start between samples', () => {
        const result = adaptAnalysisResultsComparison({
            baselineRecords: [
                driverRow(0.1, 100, { x: 10, y: 20, z: 30 }, {
                    Physics_gas: 0.2,
                    Physics_brake: 0.1,
                    Physics_gear: 2,
                }),
                driverRow(0.3, 300, { x: 30, y: 40, z: 50 }, {
                    Physics_gas: 0.8,
                    Physics_brake: 0.5,
                    Physics_gear: 4,
                }),
            ],
            expertReferenceData: [expertRow(0.2, 1_000)],
        });

        expect(result.samples).toEqual([expect.objectContaining({
            expertTimeMs: 1_000,
            driverTrackPosition: 0.2,
            expertTrackPosition: 0.2,
            driverTrajectory: { x: 20, y: 30, z: 40 },
            driverGear: 2,
        })]);
        expect(result.samples[0].driverTimeMs).toBeCloseTo(200);
        expect(result.samples[0].driverGas).toBeCloseTo(0.5);
        expect(result.samples[0].driverBrake).toBeCloseTo(0.3);
    });

    it('preserves optional channels only when their interpolation inputs are available', () => {
        const result = adaptAnalysisResultsComparison({
            baselineRecords: [
                driverRow(0.1, 100, { x: 10, y: 20 }, {
                    Physics_gas: undefined,
                    Physics_brake: 0.1,
                }),
                driverRow(0.3, 300, { x: 30, y: 40 }, {
                    Physics_gas: 0.8,
                    Physics_brake: 0.5,
                }),
            ],
            expertReferenceData: [expertRow(0.2, 1_000, {
                expert_optimal_player_pos_x: 50,
                expert_optimal_player_pos_y: 60,
                expert_optimal_player_pos_z: undefined,
                expert_optimal_throttle: Number.POSITIVE_INFINITY,
                expert_optimal_brake: 0.8,
            })],
        });

        expect(result.samples[0]).toMatchObject({
            driverTrajectory: { x: 20, y: 30 },
            expertTrajectory: { x: 50, y: 60 },
            expertBrake: 0.8,
        });
        expect(result.samples[0].driverBrake).toBeCloseTo(0.3);
        expect(result.samples[0]).not.toHaveProperty('driverGas');
        expect(result.samples[0]).not.toHaveProperty('expertGas');
    });

    it('skips isolated invalid Driver rows while retaining valid comparison coverage', () => {
        const result = adaptAnalysisResultsComparison({
            baselineRecords: [
                driverRow(0.1, 100, { x: 10, y: 10, z: 10 }),
                driverRow(0.2, 200, { x: 20, y: 20, z: 20 }),
                driverRow(0.25, 225, { x: 25, y: 25, z: 25 }, {
                    Graphics_current_time: undefined,
                }),
                driverRow(0.27, 200, { x: 27, y: 27, z: 27 }),
                driverRow(0.15, 250, { x: 15, y: 15, z: 15 }),
                driverRow(0.3, 300, { x: 30, y: 30, z: 30 }),
            ],
            expertReferenceData: [
                expertRow(0.1, 1_000),
                expertRow(0.2, 1_100),
                expertRow(0.3, 1_200),
            ],
        });

        expect(result.samples).toHaveLength(3);
        expect(result.samples.map((sample) => sample.driverTimeMs)).toEqual([100, 200, 300]);
        expect(result.samples.map((sample) => sample.driverTrackPosition)).toEqual([0.1, 0.2, 0.3]);
    });

    it('steps gear from the preceding Driver sample and advances repeated positions in source order', () => {
        const result = adaptAnalysisResultsComparison({
            baselineRecords: [
                driverRow(0.1, 100, { x: 10, y: 10, z: 10 }, {
                    Physics_gas: 0,
                    Physics_gear: 1,
                }),
                driverRow(0.2, 200, { x: 20, y: 20, z: 20 }, {
                    Physics_gas: 0.2,
                    Physics_gear: 2,
                }),
                driverRow(0.2, 250, { x: 25, y: 25, z: 25 }, {
                    Physics_gas: 0.4,
                    Physics_gear: 3,
                }),
                driverRow(0.3, 300, { x: 30, y: 30, z: 30 }, {
                    Physics_gas: 0.6,
                    Physics_gear: 4,
                }),
            ],
            expertReferenceData: [
                expertRow(0.2, 1_000),
                expertRow(0.2, 1_100),
                expertRow(0.25, 1_200),
            ],
        });

        expect(result.samples.map((sample) => sample.driverTimeMs)).toEqual([200, 250, 275]);
        expect(result.samples.map((sample) => sample.driverGas)).toEqual([0.2, 0.4, 0.5]);
        expect(result.samples.map((sample) => sample.driverGear)).toEqual([1, 2, 3]);
    });

    it('unwraps finish-line crossings before interpolating both streams', () => {
        const result = adaptAnalysisResultsComparison({
            baselineRecords: [
                driverRow(0.95, 100, { x: 10, y: 10, z: 10 }),
                driverRow(0.99, 200, { x: 20, y: 20, z: 20 }),
                driverRow(0.01, 300, { x: 30, y: 30, z: 30 }),
                driverRow(0.05, 400, { x: 40, y: 40, z: 40 }),
            ],
            expertReferenceData: [
                expertRow(0.98, 1_000),
                expertRow(0, 1_100),
                expertRow(0.04, 1_200),
            ],
        });

        expect(result.samples.map((sample) => sample.driverTimeMs)).toEqual([175, 250, 375]);
        expect(result.samples.map((sample) => sample.driverTrackPosition)).toEqual([0.98, 0, 0.04]);
        expect(result.samples.map((sample) => sample.expertTrackPosition)).toEqual([0.98, 0, 0.04]);
    });

    it('keeps Driver telemetry when the lap timer resets across the finish line', () => {
        const result = adaptAnalysisResultsComparison({
            baselineRecords: [
                driverRow(0.95, 99_800, { x: 10, y: 10, z: 10 }),
                driverRow(0.99, 99_900, { x: 20, y: 20, z: 20 }),
                driverRow(0.01, 50, { x: 30, y: 30, z: 30 }),
                driverRow(0.05, 150, { x: 40, y: 40, z: 40 }),
            ],
            expertReferenceData: [
                expertRow(0.99, 1_000),
                expertRow(0.01, 1_100),
                expertRow(0.05, 1_200),
            ],
        });

        expect(result.samples).toHaveLength(3);
        expect(result.samples.map((sample) => sample.driverTimeMs)).toEqual([
            99_900,
            99_950,
            100_050,
        ]);
        expect(result.samples.map((sample) => sample.driverTrackPosition)).toEqual([
            0.99,
            0.01,
            0.05,
        ]);
    });

    it('chooses the earliest complete Driver lap when more than one lap covers the range', () => {
        const result = adaptAnalysisResultsComparison({
            baselineRecords: [
                driverRow(0.1, 100, { x: 10, y: 10, z: 10 }),
                driverRow(0.3, 200, { x: 20, y: 20, z: 20 }),
                driverRow(0.9, 300, { x: 30, y: 30, z: 30 }),
                driverRow(0.1, 400, { x: 40, y: 40, z: 40 }),
                driverRow(0.3, 500, { x: 50, y: 50, z: 50 }),
            ],
            expertReferenceData: [
                expertRow(0.15, 1_000),
                expertRow(0.25, 1_100),
            ],
        });

        expect(result.samples.map((sample) => sample.driverTimeMs)).toEqual([125, 175]);
    });

    it.each([
        ['missing the start', [0.2, 0.4], [0.1, 0.3]],
        ['missing the end', [0.1, 0.3], [0.2, 0.4]],
    ])('rejects incomplete Driver coverage when %s of the Expert range', (
        _case,
        driverPositions,
        expertPositions,
    ) => {
        const result = adaptAnalysisResultsComparison({
            baselineRecords: driverPositions.map((position, index) => driverRow(
                position,
                100 + (index * 100),
                { x: index + 1, y: index + 1, z: index + 1 },
            )),
            expertReferenceData: expertPositions.map((position, index) => expertRow(
                position,
                1_000 + (index * 100),
            )),
        });

        expect(result.samples).toEqual([]);
    });

    it.each([
        ['missing Driver position', [undefined, 0.3], [0.1, 0.2]],
        ['out-of-range Driver position', [0.1, 1.1], [0.1, 0.2]],
        ['small Driver reversal', [0.4, 0.3], [0.35, 0.4]],
        ['missing Expert position', [0.1, 0.3], [undefined, 0.2]],
        ['out-of-range Expert position', [0.1, 0.3], [0.1, 1.1]],
        ['small Expert reversal', [0.1, 0.5], [0.4, 0.3]],
        ['half-lap backward Expert jump', [0.1, 0.9], [0.75, 0.25]],
    ])('rejects a comparison with a %s', (_case, driverPositions, expertPositions) => {
        const result = adaptAnalysisResultsComparison({
            baselineRecords: driverPositions.map((position, index) => driverRow(
                position as number,
                100 + (index * 100),
                { x: index + 1, y: index + 1, z: index + 1 },
                position === undefined ? { Graphics_normalized_car_position: undefined } : {},
            )),
            expertReferenceData: expertPositions.map((position, index) => expertRow(
                position as number,
                1_000 + (index * 100),
                position === undefined ? { Graphics_normalized_car_position: undefined } : {},
            )),
        });

        expect(result.samples).toEqual([]);
    });

    it.each([
        ['missing Driver clock', [undefined, 200], [1_000, 1_100]],
        ['repeated Driver clock', [100, 100], [1_000, 1_100]],
        ['decreasing Driver clock', [200, 100], [1_000, 1_100]],
        ['non-finite Driver clock', [100, Number.POSITIVE_INFINITY], [1_000, 1_100]],
        ['missing Expert clock', [100, 200], [undefined, 1_100]],
        ['repeated Expert clock', [100, 200], [1_000, 1_000]],
        ['decreasing Expert clock', [100, 200], [1_100, 1_000]],
        ['non-finite Expert clock', [100, 200], [1_000, Number.NaN]],
    ])('rejects a comparison with a %s', (_case, driverTimes, expertTimes) => {
        const driverPositions = [0.1, 0.3];
        const expertPositions = [0.1, 0.3];
        const result = adaptAnalysisResultsComparison({
            baselineRecords: driverPositions.map((position, index) => driverRow(
                position,
                driverTimes[index] as number,
                { x: index + 1, y: index + 1, z: index + 1 },
                driverTimes[index] === undefined ? { Graphics_current_time: undefined } : {},
            )),
            expertReferenceData: expertPositions.map((position, index) => expertRow(
                position,
                expertTimes[index] as number,
                expertTimes[index] === undefined ? { expert_optimal_time: undefined } : {},
            )),
        });

        expect(result.samples).toEqual([]);
    });

    it.each([
        ['an empty reference sequence', []],
        ['a non-object reference row', [expertRow(0.2, 1_000), null]],
        ['an array reference row', [[0.2, 1_000]]],
    ])('rejects %s', (_case, expertReferenceData) => {
        const result = adaptAnalysisResultsComparison({
            baselineRecords: [
                driverRow(0.1, 100, { x: 1, y: 1, z: 1 }),
                driverRow(0.3, 300, { x: 3, y: 3, z: 3 }),
            ],
            expertReferenceData,
        });

        expect(result.samples).toEqual([]);
    });
});

describe('resolveAnalysisResultsComparison diagnostics', () => {
    const reasonCodes = (resolution: ReturnType<typeof resolveAnalysisResultsComparison>) => (
        resolution.diagnostics.map((diagnostic) => diagnostic.code)
    );

    it('reports both missing telemetry sources', () => {
        const resolution = resolveAnalysisResultsComparison({
            baselineRecords: [],
            expertReferenceData: [],
        });

        expect(resolution.comparison).toBeUndefined();
        expect(reasonCodes(resolution)).toEqual([
            'driver_records_missing',
            'expert_reference_missing',
        ]);
    });

    it('reports every invalid Expert field category found in the reference rows', () => {
        const resolution = resolveAnalysisResultsComparison({
            baselineRecords: [
                driverRow(0.1, 100, { x: 1, y: 1, z: 1 }),
                driverRow(0.3, 300, { x: 3, y: 3, z: 3 }),
            ],
            expertReferenceData: [
                expertRow(0.1, 1_000, { expert_optimal_time: undefined }),
                expertRow(0.3, 1_200, { Graphics_normalized_car_position: 2 }),
                null,
            ],
        });

        expect(reasonCodes(resolution)).toEqual(expect.arrayContaining([
            'expert_time_missing_or_invalid',
            'expert_position_missing_or_invalid',
            'expert_row_invalid',
            'expert_reference_invalid',
        ]));
    });

    it('distinguishes incomplete coverage from a position interpolation failure', () => {
        const incomplete = resolveAnalysisResultsComparison({
            baselineRecords: [
                driverRow(0.2, 200, { x: 2, y: 2, z: 2 }),
                driverRow(0.4, 400, { x: 4, y: 4, z: 4 }),
            ],
            expertReferenceData: [expertRow(0.1, 1_000), expertRow(0.3, 1_200)],
        });
        const interpolation = resolveAnalysisResultsComparison({
            baselineRecords: [
                driverRow(0.1, 100, { x: 1, y: 1, z: 1 }),
                driverRow(0.3, 300, { x: 3, y: 3, z: 3 }),
            ],
            expertReferenceData: [expertRow(0.2, 1_000), expertRow(0.2, 1_100)],
        });

        expect(reasonCodes(incomplete)).toContain('driver_coverage_incomplete');
        expect(reasonCodes(interpolation)).toContain('driver_interpolation_failed');
    });

    it('returns no diagnostics when a comparison is displayable', () => {
        const resolution = resolveAnalysisResultsComparison({
            baselineRecords: [
                driverRow(0.1, 100, { x: 1, y: 1, z: 1 }),
                driverRow(0.3, 300, { x: 3, y: 3, z: 3 }),
            ],
            expertReferenceData: [expertRow(0.1, 1_000), expertRow(0.3, 1_200)],
        });

        expect(resolution.comparison?.samples).toHaveLength(2);
        expect(resolution.diagnostics).toEqual([]);
    });
});
