export enum RecordingState {
    CHECKING = 'CHECKING',
    READY = 'READY',
    RECORDING = 'RECORDING',
    HOLDING = 'HOLDING',
    RESUME_READY = 'RESUME_READY',
    UPLOAD_READY = 'UPLOAD_READY'
}

export type StopReason = 'manual' | 'pause' | 'error' | 'complete';

export type RecordingEvent =
    | { type: 'sessionAvailable' }
    | { type: 'sessionUnavailable' }
    | { type: 'recordingStarted' }
    | { type: 'recordingResumed' }
    | { type: 'recordingStopped'; reason: StopReason }
    | { type: 'reset' };

export const getNextRecordingState = (
    previous: RecordingState,
    event: RecordingEvent,
): RecordingState => {
    switch (event.type) {
        case 'sessionAvailable':
            if (previous === RecordingState.CHECKING) {
                return RecordingState.READY;
            }
            if (previous === RecordingState.HOLDING) {
                return RecordingState.RESUME_READY;
            }
            return previous;
        case 'sessionUnavailable':
            if (previous === RecordingState.RESUME_READY) {
                return RecordingState.HOLDING;
            }
            if (
                previous === RecordingState.RECORDING
                || previous === RecordingState.HOLDING
                || previous === RecordingState.UPLOAD_READY
            ) {
                return previous;
            }
            return RecordingState.CHECKING;
        case 'recordingStarted':
            return RecordingState.RECORDING;
        case 'recordingResumed':
            return previous === RecordingState.HOLDING || previous === RecordingState.RESUME_READY
                ? RecordingState.RECORDING
                : previous;
        case 'recordingStopped':
            switch (event.reason) {
                case 'pause':
                    return RecordingState.HOLDING;
                case 'error':
                    return RecordingState.READY;
                case 'manual':
                case 'complete':
                    return RecordingState.UPLOAD_READY;
                default:
                    return previous;
            }
        case 'reset':
            return RecordingState.CHECKING;
        default:
            return previous;
    }
};

export const isRecordingLive = (recordingState?: RecordingState | null): boolean =>
    recordingState === RecordingState.RECORDING;

export const isLiveSessionAiAvailable = (recordingState?: RecordingState | null): boolean =>
    recordingState === RecordingState.RECORDING
    || recordingState === RecordingState.HOLDING
    || recordingState === RecordingState.RESUME_READY;

export const hasLiveSessionAssistant = (recordingState?: RecordingState | null): boolean =>
    isLiveSessionAiAvailable(recordingState)
    || recordingState === RecordingState.UPLOAD_READY;
