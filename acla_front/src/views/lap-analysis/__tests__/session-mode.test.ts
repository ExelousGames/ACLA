import {
    buildAssistantConversationKey,
    resolveAssistantRecordedSessionId,
    resolveRegisteredAssistantIdentity,
} from '../assistant-session-mode';
import {
    getNextRecordingState,
    isLiveSessionAiAvailable,
    RecordingState,
} from '../recording-state';

describe('assistant session mode resolution', () => {
    const createRegistration = (overrides: Record<string, unknown> = {}) => ({
        screenId: 'live-session',
        assistantMode: 'live' as const,
        label: 'Live Session',
        componentRef: { current: null },
        ...overrides,
    });

    it('uses the Front Desk fallback while registration is temporarily unavailable', () => {
        expect(resolveRegisteredAssistantIdentity(null)).toEqual({
            sessionMode: 'front_desk',
            sessionId: undefined,
            label: 'Front Desk',
            conversationKey: 'front_desk:none',
            title: 'AI Assistant - Front Desk',
        });
    });

    it('uses the active registration instead of recording or dashboard state', () => {
        expect(resolveRegisteredAssistantIdentity(createRegistration())).toMatchObject({
            sessionMode: 'live',
            label: 'Live Session',
            conversationKey: 'live:none',
        });
    });

    it('keeps AI available while paused but closes it after ending', () => {
        expect(isLiveSessionAiAvailable(RecordingState.HOLDING)).toBe(true);
        expect(isLiveSessionAiAvailable(RecordingState.RESUME_READY)).toBe(true);
        expect(isLiveSessionAiAvailable(RecordingState.UPLOAD_READY)).toBe(false);
    });

    it('uses the registered recorded id for title and conversation identity', () => {
        expect(resolveRegisteredAssistantIdentity(createRegistration({
            screenId: 'recorded-session',
            assistantMode: 'recorded',
            label: 'Race 12',
            recordedSessionId: 'session-1',
        }) as any)).toMatchObject({
            sessionMode: 'recorded',
            sessionId: 'session-1',
            label: 'Race 12',
            conversationKey: 'recorded:session-1',
            title: 'AI Assistant - Race 12',
        });
    });

    it.each(['live', 'front_desk', 'user_summary'] as const)(
        'does not expose a recorded session id in %s mode',
        (sessionMode) => {
            expect(resolveAssistantRecordedSessionId(sessionMode, 'session-1')).toBeUndefined();
        },
    );

    it('keeps the recorded session id in recorded mode', () => {
        expect(resolveAssistantRecordedSessionId('recorded', 'session-1')).toBe('session-1');
    });

    it('isolates the live conversation from a recorded session selection', () => {
        const sessionId = resolveAssistantRecordedSessionId('live', 'session-1');
        expect(buildAssistantConversationKey('live', sessionId)).toBe('live:none');
    });

    it('builds a stable front desk conversation key', () => {
        expect(buildAssistantConversationKey('front_desk')).toBe('front_desk:none');
    });
});

describe('live session detection recording transitions', () => {
    it('moves from checking to ready when a live session is detected', () => {
        expect(getNextRecordingState(
            RecordingState.CHECKING,
            { type: 'sessionAvailable' },
        )).toBe(RecordingState.READY);
    });

    it('moves from ready to recording when recording starts', () => {
        expect(getNextRecordingState(
            RecordingState.READY,
            { type: 'recordingStarted' },
        )).toBe(RecordingState.RECORDING);
    });

    it('uses the existing pause and resume states', () => {
        expect(getNextRecordingState(
            RecordingState.RECORDING,
            { type: 'recordingStopped', reason: 'pause' },
        )).toBe(RecordingState.HOLDING);

        expect(getNextRecordingState(
            RecordingState.HOLDING,
            { type: 'sessionAvailable' },
        )).toBe(RecordingState.RESUME_READY);

        expect(getNextRecordingState(
            RecordingState.RESUME_READY,
            { type: 'recordingResumed' },
        )).toBe(RecordingState.RECORDING);
    });

    it('resets to checking without introducing a new status', () => {
        expect(getNextRecordingState(
            RecordingState.UPLOAD_READY,
            { type: 'reset' },
        )).toBe(RecordingState.CHECKING);
    });

    it('makes a completed recording upload-ready', () => {
        expect(getNextRecordingState(
            RecordingState.RECORDING,
            { type: 'recordingStopped', reason: 'complete' },
        )).toBe(RecordingState.UPLOAD_READY);
    });
});
