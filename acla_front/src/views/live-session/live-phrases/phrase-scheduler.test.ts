import { Phrase, type PhraseAudioVariant } from './phrase';
import { filterPhraseAudioVariants, schedulePhrases, type PhraseScheduleCandidate, type PhraseScheduleItem } from './phrase-scheduler';

const candidate = (id: string, priority: number, durationMs = 200): PhraseScheduleCandidate => ({
    id, kind: 'closure', label: id, priority, category: 'other', durationMs,
});
const audio = (durationMs: number): PhraseAudioVariant => ({
    instance: new Phrase('turn', 'Turn in.').createInstance({ speed: durationMs / 1000 }), durationMs,
});

it('keeps only audio variants that fit the remaining timeline including reaction time', () => {
    const variants = [audio(300), audio(500), audio(700)];
    const allowed = filterPhraseAudioVariants(variants, { startAt: 1200, endAt: 1800, reactionAllowanceMs: 100 });
    expect(allowed).toEqual([variants[0], variants[1]]);
    expect(allowed[0]).toBe(variants[0]);
    expect(allowed).not.toBe(variants);
    expect(variants.map(({ durationMs }) => durationMs)).toEqual([300, 500, 700]);
    expect(filterPhraseAudioVariants(variants, { startAt: 1500, endAt: 1800, reactionAllowanceMs: 100 })).toEqual([]);
});

it('allows all measured variants in an open window and rejects invalid durations', () => {
    const valid = [audio(0), audio(300), audio(10000)];
    const variants = [...valid, audio(-1), audio(NaN), audio(Infinity)];
    expect(filterPhraseAudioVariants(variants, { startAt: 1000 })).toEqual(valid);
    expect(filterPhraseAudioVariants(variants, { startAt: 1000, endAt: 999 })).toEqual([]);
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

it('filters the random pool at its queued start after active playback and reserves every allowed choice', () => {
    const active: PhraseScheduleItem = { id: 'active', kind: 'phrase', label: 'Active', priority: 0,
        category: 'other', status: 'active', startAt: 900, endAt: 1400 };
    const audioVariants = [audio(200), audio(400), audio(600)];
    const next = { ...candidate('turn', 2, 600), audioVariants, deadlineAt: 1900, reactionAllowanceMs: 100 };
    const schedule = schedulePhrases(1000, [next, candidate('throttle', 1)], active);
    expect(schedule.items[1]).toMatchObject({ startAt: 1400, endAt: 1900, audioVariants: audioVariants.slice(0, 2) });
    expect(schedule.items[2]).toMatchObject({ startAt: 1900, endAt: 2100 });
    expect(next.audioVariants).toHaveLength(3);
    expect(schedule.items[1].audioVariants).not.toBe(next.audioVariants);
    next.audioVariants.push(audio(100));
    expect(schedule.items[1].audioVariants).toHaveLength(2);
});

it('drops empty random pools without consuming a slot and keeps fixed deliveries authoritative', () => {
    const candidates = [
        { ...candidate('too-long', 3), audioVariants: [audio(300)], deadlineAt: 1100 },
        { ...candidate('empty', 2), audioVariants: [] },
        { ...candidate('fixed', 1, 400), deadlineAt: 900 },
        { ...candidate('short', 0), audioVariants: [audio(100), audio(300)], deadlineAt: 1600 },
    ];
    const schedule = schedulePhrases(1000, candidates);
    expect(schedule.items.map(({ id, status, startAt, endAt }) => ({ id, status, startAt, endAt }))).toEqual([
        { id: 'fixed', status: 'next', startAt: 1000, endAt: 1400 },
        { id: 'short', status: 'queued', startAt: 1400, endAt: 1500 },
    ]);
});
