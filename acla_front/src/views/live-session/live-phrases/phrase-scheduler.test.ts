import { schedulePhrases, type PhraseScheduleCandidate, type PhraseScheduleItem } from './phrase-scheduler';

const candidate = (id: string, priority: number, durationMs = 200): PhraseScheduleCandidate => ({
    id, kind: 'closure', label: id, priority, category: 'other', durationMs,
});

it('schedules higher priorities first and keeps declaration order for ties', () => {
    const candidates = [candidate('throttle', 1), candidate('turn-a', 2), candidate('brake', 3), candidate('turn-b', 2)];
    const schedule = schedulePhrases(1000, candidates);
    expect(schedule.items.map(({ id, status, startAt, endAt }) => ({ id, status, startAt, endAt }))).toEqual([
        { id: 'brake', status: 'next', startAt: 1000, endAt: 1200 },
        { id: 'turn-a', status: 'queued', startAt: 1200, endAt: 1400 },
        { id: 'turn-b', status: 'queued', startAt: 1400, endAt: 1600 },
        { id: 'throttle', status: 'queued', startAt: 1600, endAt: 1800 },
    ]);
    expect(candidates.map(({ id }) => id)).toEqual(['throttle', 'turn-a', 'brake', 'turn-b']);
    expect(candidates.every((item) => !('status' in item))).toBe(true);
});

it('retains active speech even if a more important cue is ready and avoids overlapping slots', () => {
    const active: PhraseScheduleItem = { id: 'throttle', kind: 'phrase', label: 'Squeeze gas.', priority: 1,
        category: 'throttle', status: 'active', startAt: 900, endAt: 1700 };
    const schedule = schedulePhrases(1000, [candidate('throttle', 1), candidate('turn', 2, 400), candidate('brake', 3, 300)], active);
    expect(schedule.items.map(({ id }) => id)).toEqual(['throttle', 'brake', 'turn']);
    expect(schedule.items[0]).toEqual(active);
    expect(schedule.items[0]).not.toBe(active);
    expect(schedule.items[1]).toMatchObject({ status: 'next', startAt: 1700, endAt: 2000 });
    expect(schedule.items[2]).toMatchObject({ startAt: 2000, endAt: 2400 });
    expect(active.endAt).toBe(1700);
});

it('never places pending slots in the past and drops candidates absent from a new prediction', () => {
    const active: PhraseScheduleItem = { id: 'active', kind: 'phrase', label: 'Active', priority: 0,
        category: 'other', status: 'active', startAt: 500, endAt: 900 };
    const first = schedulePhrases(1000, [candidate('brake', 3)], active);
    expect(first.items[1]).toMatchObject({ startAt: 1000, endAt: 1200 });
    expect(schedulePhrases(1100, [], active).items).toEqual([active]);
    expect(schedulePhrases(1200, [])).toEqual({ now: 1200, items: [] });
    expect(first.items).toHaveLength(2);
});
