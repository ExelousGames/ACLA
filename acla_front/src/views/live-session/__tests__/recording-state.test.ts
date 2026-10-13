import {
    getNextRecordingState,
    isLiveSessionAiAvailable,
    RecordingState,
} from '../recording-state';

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

    it('keeps AI available while paused but closes it after ending', () => {
        expect(isLiveSessionAiAvailable(RecordingState.HOLDING)).toBe(true);
        expect(isLiveSessionAiAvailable(RecordingState.RESUME_READY)).toBe(true);
        expect(isLiveSessionAiAvailable(RecordingState.UPLOAD_READY)).toBe(false);
    });
});
