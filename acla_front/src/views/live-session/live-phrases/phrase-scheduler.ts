export type PhraseScheduleCategory = 'brake' | 'turn' | 'throttle' | 'other';

export const PHRASE_PRIORITIES = { brake: 3, turn: 2, throttle: 1, other: 0 } as const;

/** Stable sorting keeps declaration order when two candidates have equal importance. */
export function comparePriority(a: { priority: number }, b: { priority: number }): number {
    return b.priority - a.priority;
}

export interface PhraseScheduleCandidate {
    id: string;
    kind: 'phrase' | 'closure';
    label: string;
    priority: number;
    category: PhraseScheduleCategory;
    durationMs: number;
    predictedAt?: number;
    durationEstimated?: boolean;
}

export interface PhraseScheduleItem extends Omit<PhraseScheduleCandidate, 'durationMs'> {
    status: 'active' | 'next' | 'queued';
    startAt: number;
    endAt: number;
}

export interface PhraseScheduleSnapshot {
    now: number;
    items: PhraseScheduleItem[];
}

/** Project the current eligible set; callers rebuild it as predictions change. */
export function schedulePhrases(
    now: number, candidates: readonly PhraseScheduleCandidate[], active?: PhraseScheduleItem,
): PhraseScheduleSnapshot {
    let availableAt = Math.max(now, active?.endAt ?? now);
    const pending = [...candidates].filter(({ id }) => id !== active?.id).sort(comparePriority);
    return { now, items: [
        ...(active ? [{ ...active }] : []),
        ...pending.map(({ durationMs, ...candidate }, index): PhraseScheduleItem => {
            const startAt = availableAt;
            availableAt += durationMs;
            return { ...candidate, status: index === 0 ? 'next' : 'queued', startAt, endAt: availableAt };
        }),
    ] };
}
