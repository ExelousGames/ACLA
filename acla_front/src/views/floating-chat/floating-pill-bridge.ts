export const FLOATING_PILL_STORAGE_KEY = 'acla-pill-msg';
export const FLOATING_PILL_RICH_CONTENT_HOLD_MS = 3800;
export const FLOATING_PILL_COMPARISON_COMPLETION_PAUSE_MS = 800;

export type FloatingPillPayloadKind =
    | 'message'
    | 'tool'
    | 'baseline'
    | 'map'
    | 'plan'
    | 'live_range_todo_list'
    | 'driver_expert_comparison';

export interface FloatingPillPayloadInput {
    kind: FloatingPillPayloadKind;
    text?: string;
    data?: unknown;
    emotion?: string;
    tags?: string[];
    name?: string;
}

const getPreviousTimestamp = (): number => {
    try {
        const previous = JSON.parse(localStorage.getItem(FLOATING_PILL_STORAGE_KEY) || '{}');
        const timestamp = Number(previous?.ts);
        return Number.isFinite(timestamp) ? timestamp : 0;
    } catch {
        return 0;
    }
};

export const broadcastFloatingPillPayload = (
    payload: FloatingPillPayloadInput,
): boolean => {
    try {
        const timestamp = Math.max(Date.now(), getPreviousTimestamp() + 1);
        localStorage.setItem(FLOATING_PILL_STORAGE_KEY, JSON.stringify({
            ...payload,
            ts: timestamp,
        }));
        return true;
    } catch {
        return false;
    }
};
