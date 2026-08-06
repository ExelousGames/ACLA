import {
    buildAssistantConversationKey,
    resolveAssistantRecordedSessionId,
    resolveAssistantSessionMode,
} from '../assistant-session-mode';
import { getNextRecordingState, RecordingState } from '../recording-state';

describe('assistant session mode resolution', () => {
    it('defaults to front desk before a session is selected or recording starts', () => {
        expect(resolveAssistantSessionMode({
            recordingState: RecordingState.CHECKING,
        })).toBe('front_desk');
    });

    it('uses live mode while shared recording state is actively recording', () => {
        expect(resolveAssistantSessionMode({
            recordingState: RecordingState.RECORDING,
        })).toBe('live');
    });

    it('uses recorded mode when a recorded session id is selected', () => {
        expect(resolveAssistantSessionMode({
            sessionId: 'session-1',
            recordingState: RecordingState.CHECKING,
        })).toBe('recorded');
    });

    it('keeps explicit assistant overrides authoritative', () => {
        expect(resolveAssistantSessionMode({
            assistantModeOverride: 'user_summary',
            sessionId: 'session-1',
            recordingState: RecordingState.RECORDING,
        })).toBe('user_summary');
    });

    it('keeps a live tab override authoritative while checking with a recorded session selected', () => {
        expect(resolveAssistantSessionMode({
            assistantModeOverride: 'live',
            sessionId: 'session-1',
            recordingState: RecordingState.CHECKING,
        })).toBe('live');
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
