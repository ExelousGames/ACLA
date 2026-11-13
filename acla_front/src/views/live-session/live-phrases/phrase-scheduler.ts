import type { PhraseAudioVariant } from './phrase';

export type PhraseScheduleCategory = 'brake' | 'turn' | 'throttle' | 'other';

/** Stable sorting keeps declaration order when two candidates have equal importance. */
export function comparePriority(a: { priority: number }, b: { priority: number }): number {
    return b.priority - a.priority;
}

/** Filter the random pool without choosing an audio or changing phrase conditions. */
export function filterPhraseAudioVariants(
    variants: readonly PhraseAudioVariant[],
    window: { startAt: number; endAt?: number; reactionAllowanceMs?: number },
): PhraseAudioVariant[] {
    const availableMs = window.endAt === undefined ? Infinity : window.endAt - window.startAt;
    return variants.filter(({ durationMs }) => Number.isFinite(durationMs) && durationMs >= 0
        && durationMs + (window.reactionAllowanceMs ?? 0) <= availableMs);
}

export interface PhraseScheduleCandidate {
    id: string;
    kind: 'phrase';
    label: string;
    priority: number;
    category: PhraseScheduleCategory;
    durationMs: number;
    /** Only random alternatives are filtered; fixed deliveries keep durationMs. */
    audioVariants?: readonly PhraseAudioVariant[];
    deadlineAt?: number;
    reactionAllowanceMs?: number;
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
    const items: PhraseScheduleItem[] = active ? [{ ...active,
        ...(active.audioVariants ? { audioVariants: [...active.audioVariants] } : {}),
    }] : [];
    let pendingCount = 0;
    pending.forEach(({ durationMs, audioVariants, ...candidate }) => {
        const startAt = availableAt;
        const allowed = audioVariants === undefined ? undefined : filterPhraseAudioVariants(audioVariants, {
            startAt, endAt: candidate.deadlineAt, reactionAllowanceMs: candidate.reactionAllowanceMs,
        });
        if (allowed?.length === 0) return;
        // Reserve enough time for every remaining choice; selection happens on entry.
        availableAt += allowed ? Math.max(...allowed.map((variant) => variant.durationMs))
            + (candidate.reactionAllowanceMs ?? 0) : durationMs;
        items.push({ ...candidate, ...(allowed ? { audioVariants: allowed } : {}),
            status: pendingCount++ === 0 ? 'next' : 'queued', startAt, endAt: availableAt });
    });
    return { now, items };
}
