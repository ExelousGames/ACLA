import { RecordingState } from 'views/lap-analysis/recording-state';
import {
    PERSISTED_LIVE_SESSION_DRAFT_VERSION,
    PersistedLiveSessionDraft,
} from './live-session-types';

export const LIVE_SESSION_DRAFT_STORAGE_KEY = 'acla.live-session-drafts';

interface PersistedLiveSessionDraftManifest {
    version: typeof PERSISTED_LIVE_SESSION_DRAFT_VERSION;
    drafts: Record<string, unknown>;
}

const emptyManifest = (): PersistedLiveSessionDraftManifest => ({
    version: PERSISTED_LIVE_SESSION_DRAFT_VERSION,
    drafts: {},
});

export const normalizeLiveSessionOwnerEmail = (email?: string | null): string =>
    typeof email === 'string' ? email.trim().toLowerCase() : '';

const readManifest = (storage: Storage): PersistedLiveSessionDraftManifest => {
    try {
        const raw = storage.getItem(LIVE_SESSION_DRAFT_STORAGE_KEY);
        if (!raw) return emptyManifest();
        const parsed = JSON.parse(raw);
        if (
            !parsed
            || parsed.version !== PERSISTED_LIVE_SESSION_DRAFT_VERSION
            || !parsed.drafts
            || typeof parsed.drafts !== 'object'
            || Array.isArray(parsed.drafts)
        ) {
            return emptyManifest();
        }
        return parsed as PersistedLiveSessionDraftManifest;
    } catch {
        return emptyManifest();
    }
};

const isSupportedGame = (value: unknown): value is PersistedLiveSessionDraft['sessionGame'] =>
    value === 'acc' || value === 'ac' || value === 'iracing';

const isRecordingState = (value: unknown): value is RecordingState =>
    Object.values(RecordingState).includes(value as RecordingState);

const isDraft = (value: unknown, ownerEmail: string): value is PersistedLiveSessionDraft => {
    if (!value || typeof value !== 'object') return false;
    const draft = value as Partial<PersistedLiveSessionDraft>;
    const metadata = draft.recordingMetadata;
    return draft.version === PERSISTED_LIVE_SESSION_DRAFT_VERSION
        && normalizeLiveSessionOwnerEmail(draft.ownerEmail) === ownerEmail
        && isSupportedGame(draft.sessionGame)
        && Boolean(metadata)
        && typeof metadata?.sessionName === 'string'
        && typeof metadata?.mapName === 'string'
        && typeof metadata?.carName === 'string'
        && isSupportedGame(metadata?.gameRecordedFrom)
        && typeof draft.telemetryFilePath === 'string'
        && draft.telemetryFilePath.length > 0
        && typeof draft.recordedSampleCount === 'number'
        && Number.isFinite(draft.recordedSampleCount)
        && draft.recordedSampleCount >= 0
        && isRecordingState(draft.lastRuntimeState)
        && typeof draft.updatedAt === 'string';
};

export const getPersistedLiveSessionDraft = (
    ownerEmail: string,
    storage: Storage = window.localStorage,
): PersistedLiveSessionDraft | null => {
    const normalizedOwnerEmail = normalizeLiveSessionOwnerEmail(ownerEmail);
    if (!normalizedOwnerEmail) return null;
    const candidate = readManifest(storage).drafts[normalizedOwnerEmail];
    return isDraft(candidate, normalizedOwnerEmail) ? candidate : null;
};

export const savePersistedLiveSessionDraft = (
    draft: PersistedLiveSessionDraft,
    storage: Storage = window.localStorage,
): void => {
    const ownerEmail = normalizeLiveSessionOwnerEmail(draft.ownerEmail);
    if (!ownerEmail) return;
    const manifest = readManifest(storage);
    manifest.drafts[ownerEmail] = { ...draft, ownerEmail };
    try {
        storage.setItem(LIVE_SESSION_DRAFT_STORAGE_KEY, JSON.stringify(manifest));
    } catch (error) {
        console.warn('Unable to persist the live session draft', error);
    }
};

export const removePersistedLiveSessionDraft = (
    ownerEmail: string,
    storage: Storage = window.localStorage,
): void => {
    const normalizedOwnerEmail = normalizeLiveSessionOwnerEmail(ownerEmail);
    if (!normalizedOwnerEmail) return;
    const manifest = readManifest(storage);
    if (!Object.prototype.hasOwnProperty.call(manifest.drafts, normalizedOwnerEmail)) return;
    delete manifest.drafts[normalizedOwnerEmail];
    try {
        storage.setItem(LIVE_SESSION_DRAFT_STORAGE_KEY, JSON.stringify(manifest));
    } catch (error) {
        console.warn('Unable to remove the live session draft', error);
    }
};
