import { adaptAnalysisResultsComparison } from './analysisResultsComparisonAdapter';

const driverRow = (
    rawIndex: number,
    position: number,
    player: { x: number; y: number; z: number },
    overrides: Record<string, unknown> = {},
) => ({
    raw_index: rawIndex,
    Graphics_current_time: rawIndex * 100,
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
    rawIndex: number,
    position: number,
    overrides: Record<string, unknown> = {},
) => ({
    raw_index: rawIndex,
    expert_optimal_time: rawIndex * 90,
    Graphics_normalized_car_position: position,
    expert_optimal_player_pos_x: rawIndex * 10,
    expert_optimal_player_pos_y: rawIndex * 5,
    expert_optimal_player_pos_z: rawIndex * -10,
    expert_optimal_throttle: 0.6,
    expert_optimal_brake: 0.2,
    expert_optimal_gear: 4,
    ...overrides,
});

describe('adaptAnalysisResultsComparison', () => {
    it('joins by raw index, filters the exact inclusive interval, and selects the player car', () => {
        const result = adaptAnalysisResultsComparison({
            baselineRecords: [
                driverRow(10, 0.1, { x: 10, y: 11, z: 12 }),
                driverRow(11, 0.2, { x: 20, y: 21, z: 22 }),
                driverRow(12, 0.3, { x: 30, y: 31, z: 32 }),
            ],
            expertReferenceData: [
                expertRow(12, 0.32),
                expertRow(9, 0.05),
                expertRow(10, 0.12),
                expertRow(11, 0.22),
                expertRow(13, 0.4),
            ],
            startIndex: 10,
            endIndex: 12,
        });

        expect(result.samples).toHaveLength(3);
        expect(result.samples.map((sample) => sample.driverTrackPosition)).toEqual([0.1, 0.2, 0.3]);
        expect(result.samples.map((sample) => sample.expertTrackPosition)).toEqual([0.12, 0.22, 0.32]);
        expect(result.samples[0]).toMatchObject({
            driverTimeMs: 1_000,
            expertTimeMs: 900,
            driverTrackPosition: 0.1,
            expertTrackPosition: 0.12,
            driverTrajectory: { x: 10, y: 11, z: 12 },
            expertTrajectory: { x: 100, y: 50, z: -100 },
            driverGas: 0.5,
            expertGas: 0.6,
        });
        expect(result.samples[2]).toMatchObject({
            driverTimeMs: 1_200,
            expertTimeMs: 1_080,
        });
    });

    it('keeps timestamp mapping independent of distance through a finish-line crossing', () => {
        const result = adaptAnalysisResultsComparison({
            baselineRecords: [
                driverRow(20, 0.98, { x: 1, y: 1, z: 1 }),
                driverRow(21, 0.01, { x: 2, y: 2, z: 2 }),
                driverRow(22, 0.04, { x: 3, y: 3, z: 3 }),
            ],
            expertReferenceData: [
                expertRow(20, 0.96),
                expertRow(21, 0.99),
                expertRow(22, 0.02),
            ],
            startIndex: 20,
            endIndex: 22,
        });

        expect(result.samples.map(({ driverTimeMs, expertTimeMs }) => ({
            driverTimeMs,
            expertTimeMs,
        }))).toEqual([
            { driverTimeMs: 2_000, expertTimeMs: 1_800 },
            { driverTimeMs: 2_100, expertTimeMs: 1_890 },
            { driverTimeMs: 2_200, expertTimeMs: 1_980 },
        ]);
        expect(result.samples.map((sample) => sample.driverTrackPosition)).toEqual([0.98, 0.01, 0.04]);
        expect(result.samples.map((sample) => sample.expertTrackPosition)).toEqual([0.96, 0.99, 0.02]);
    });

    it('retains each available source axis without synthesizing missing coordinates', () => {
        const result = adaptAnalysisResultsComparison({
            baselineRecords: [driverRow(25, 0.4, { x: 1, y: 2, z: 3 }, {
                Graphics_car_coordinates: [
                    { x: -1, y: -2, z: -3 },
                    { x: 10, y: 20 },
                ],
            })],
            expertReferenceData: [expertRow(25, 0.4, {
                expert_optimal_player_pos_x: 30,
                expert_optimal_player_pos_y: 40,
                expert_optimal_player_pos_z: undefined,
            })],
            startIndex: 25,
            endIndex: 25,
        });

        expect(result.samples[0].driverTrajectory).toEqual({ x: 10, y: 20 });
        expect(result.samples[0].expertTrajectory).toEqual({ x: 30, y: 40 });
    });

    it('omits non-finite values without losing other comparable channels', () => {
        const result = adaptAnalysisResultsComparison({
            baselineRecords: [driverRow(30, 0.5, { x: Infinity, y: 2, z: 3 }, {
                Graphics_car_coordinates: [
                    { x: Infinity, y: 1, z: 1 },
                    { x: Infinity, y: 2, z: 3 },
                ],
                Physics_gas: Infinity,
                Physics_brake: 0.3,
                Physics_gear: Number.NaN,
            })],
            expertReferenceData: [expertRow(30, 0.5, {
                expert_optimal_player_pos_x: Number.NaN,
                expert_optimal_throttle: Number.NaN,
                expert_optimal_brake: 0.4,
                expert_optimal_gear: Infinity,
            })],
            startIndex: 30,
            endIndex: 30,
        });

        expect(result.samples).toEqual([{
            driverTimeMs: 3_000,
            expertTimeMs: 2_700,
            driverTrackPosition: 0.5,
            expertTrackPosition: 0.5,
            driverBrake: 0.3,
            expertBrake: 0.4,
        }]);
    });

    it('falls back to baseline array indexes when rows have no explicit raw index', () => {
        const records = [
            driverRow(0, 0.1, { x: 1, y: 1, z: 1 }),
            driverRow(1, 0.2, { x: 2, y: 2, z: 2 }),
        ].map(({ raw_index: _rawIndex, ...row }) => row);
        const result = adaptAnalysisResultsComparison({
            baselineRecords: records,
            expertReferenceData: [expertRow(1, 0.2)],
            startIndex: 1,
            endIndex: 1,
        });

        expect(result.samples).toHaveLength(1);
        expect(result.samples[0].driverTrajectory).toEqual({ x: 2, y: 2, z: 2 });
    });

    it.each([
        ['missing', { Graphics_current_time: undefined }, {}],
        ['non-finite', {}, { expert_optimal_time: Number.POSITIVE_INFINITY }],
    ])('rejects every joined sample when a timestamp is %s', (_case, driverOverrides, expertOverrides) => {
        const result = adaptAnalysisResultsComparison({
            baselineRecords: [driverRow(40, 0.1, { x: 1, y: 2, z: 3 }, driverOverrides)],
            expertReferenceData: [expertRow(40, 0.1, expertOverrides)],
            startIndex: 40,
            endIndex: 40,
        });

        expect(result.samples).toEqual([]);
    });

    it.each([
        ['repeated driver clock', [100, 100], [500, 600]],
        ['decreasing driver clock', [200, 100], [500, 600]],
        ['repeated expert clock', [100, 200], [500, 500]],
        ['decreasing expert clock', [100, 200], [600, 500]],
    ])('rejects the complete comparison for a %s', (_case, driverTimes, expertTimes) => {
        const result = adaptAnalysisResultsComparison({
            baselineRecords: [50, 51].map((rawIndex, index) => driverRow(
                rawIndex,
                index / 10,
                { x: index, y: index, z: index },
                { Graphics_current_time: driverTimes[index] },
            )),
            expertReferenceData: [50, 51].map((rawIndex, index) => expertRow(
                rawIndex,
                index / 10,
                { expert_optimal_time: expertTimes[index] },
            )),
            startIndex: 50,
            endIndex: 51,
        });

        expect(result.samples).toEqual([]);
    });

    it.each([
        ['missing driver position', { Graphics_normalized_car_position: undefined }, {}],
        ['missing expert position', {}, { Graphics_normalized_car_position: undefined }],
        ['non-finite driver position', { Graphics_normalized_car_position: Number.NaN }, {}],
        ['out-of-range expert position', {}, { Graphics_normalized_car_position: 1.1 }],
    ])('rejects the comparison for a %s', (_case, driverOverrides, expertOverrides) => {
        const result = adaptAnalysisResultsComparison({
            baselineRecords: [driverRow(60, 0.2, { x: 1, y: 2, z: 3 }, driverOverrides)],
            expertReferenceData: [expertRow(60, 0.2, expertOverrides)],
            startIndex: 60,
            endIndex: 60,
        });

        expect(result.samples).toEqual([]);
    });

    it.each([
        ['directionally invalid', [0.4, 0.3], [0.4, 0.5]],
        ['non-overlapping', [0.1, 0.2], [0.4, 0.5]],
    ])('rejects %s position streams', (_case, driverPositions, expertPositions) => {
        const result = adaptAnalysisResultsComparison({
            baselineRecords: [70, 71].map((rawIndex, index) => driverRow(
                rawIndex,
                driverPositions[index],
                { x: index, y: index, z: index },
            )),
            expertReferenceData: [70, 71].map((rawIndex, index) => expertRow(
                rawIndex,
                expertPositions[index],
            )),
            startIndex: 70,
            endIndex: 71,
        });

        expect(result.samples).toEqual([]);
    });
});
