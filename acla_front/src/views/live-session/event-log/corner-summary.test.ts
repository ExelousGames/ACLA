import type { DriverCornerRecord } from './CornerRecorder';
import { summarizeDriverCorners } from './corner-summary';

const record = (cornerTimeMs: number | null, overrides: Partial<DriverCornerRecord> = {}): DriverCornerRecord => ({
    id: 'pass', carId: '7', isPlayer: true, cornerId: 'turn-1', cornerName: 'Corner 1',
    entry: null, exit: { timeMs: 100000, position: 0.3, sampleIndex: 10 },
    decelerationStart: null, accelerationStart: null, decelerationToExitMs: null,
    cornerTimeMs, exitSpeed: null, mapId: 'track', ...overrides,
});

describe('driver corner summaries', () => {
    it('trims exit speeds independently of corner times and records incomplete corner exits', () => {
        const speeds = [0.001, 0.03, 0.02, 0.09, 0.04, 0.05, 0.06, 0.07];
        const summary = summarizeDriverCorners(speeds.map((exitSpeed, index) => record(
            index === 0 ? null : (index + 1) * 1000, { exitSpeed },
        )))[0];
        expect(summary).toMatchObject({ sampleCount: 7, exitSpeedSampleCount: 8, exitSpeedIncludedCount: 6 });
        expect(summary.averageExitSpeed).toBeCloseTo(0.045);
        expect(summarizeDriverCorners([null, NaN, Infinity, -1, 0, 0.02, 0.03].map((exitSpeed) => record(5000, { exitSpeed })))[0])
            .toMatchObject({ averageExitSpeed: null, exitSpeedSampleCount: 2, exitSpeedIncludedCount: 0 });
    });

    it('keeps exit speed samples separate by driver and corner', () => {
        const records = [0.01, 0.02, 0.03].flatMap((exitSpeed) => [
            record(1000, { exitSpeed }),
            record(1000, { exitSpeed: exitSpeed * 2, carId: '63', isPlayer: false }),
            record(1000, { exitSpeed: exitSpeed * 3, cornerId: 'turn-2' }),
        ]);
        expect(summarizeDriverCorners(records).map((summary) => summary.averageExitSpeed)).toEqual([0.02, 0.06, 0.04]);
    });
    it('averages the middle 75% and excludes both fast and slow extremes', () => {
        const records = [90000, 5000, 1000, 9000, 4000, 8000, 7000, 6000].map((time) => record(time));
        const original = records.map((pass) => ({ ...pass }));
        expect(summarizeDriverCorners(records)).toEqual([expect.objectContaining({
            carId: '7', cornerId: 'turn-1', averageTimeMs: 6500, sampleCount: 8, includedCount: 6,
        })]);
        expect(records).toEqual(original);
    });

    it('trims more samples as history grows and rounds down to whole passes', () => {
        const middle = Array.from({ length: 12 }, (_, index) => (index + 4) * 1000);
        const times = [1000, 2000, ...middle, 100000, 200000];
        expect(summarizeDriverCorners(times.map((time) => record(time)))[0]).toMatchObject({
            averageTimeMs: 9500, sampleCount: 16, includedCount: 12,
        });
        expect(summarizeDriverCorners([...times, 300000].map((time) => record(time)))[0]).toMatchObject({
            sampleCount: 17, includedCount: 13,
        });
    });

    it.each([[], [1000], [1000, 90000]].map((times) => ({ times })))('waits for three complete passes: $times', ({ times }) => {
        const summary = summarizeDriverCorners([record(null), ...times.map((time) => record(time))])[0];
        expect(summary).toMatchObject({ averageTimeMs: null, sampleCount: times.length, includedCount: 0 });
    });

    it('excludes at least one time at each end for small samples and handles ties', () => {
        expect(summarizeDriverCorners([1000, 6000, 90000].map((time) => record(time)))[0]).toMatchObject({
            averageTimeMs: 6000, sampleCount: 3, includedCount: 1,
        });
        expect(summarizeDriverCorners([5000, 5000, 5000, 5000].map((time) => record(time)))[0]).toMatchObject({
            averageTimeMs: 5000, sampleCount: 4, includedCount: 2,
        });
    });

    it('ignores incomplete or invalid times instead of counting them toward the minimum', () => {
        const times = [null, NaN, Infinity, -Infinity, 0, -1, 5000, 6000];
        expect(summarizeDriverCorners(times.map((time) => record(time)))[0]).toMatchObject({
            averageTimeMs: null, sampleCount: 2, includedCount: 0,
        });
    });

    it('keeps each driver and corner separate, including previously observed player identity', () => {
        const records = [1000, 5000, 9000].flatMap((time) => [
            record(time, { isPlayer: time === 1000 }),
            record(time + 1000, { carId: '63', isPlayer: false }),
            record(time + 2000, { cornerId: 'turn-2', cornerName: 'Corner 2' }),
        ]);
        expect(summarizeDriverCorners(records)).toEqual([
            expect.objectContaining({ carId: '7', cornerId: 'turn-1', isPlayer: true, averageTimeMs: 5000, sampleCount: 3 }),
            expect.objectContaining({ carId: '7', cornerId: 'turn-2', averageTimeMs: 7000, sampleCount: 3 }),
            expect.objectContaining({ carId: '63', cornerId: 'turn-1', isPlayer: false, averageTimeMs: 6000, sampleCount: 3 }),
        ]);
        expect(summarizeDriverCorners([])).toEqual([]);
    });
});
