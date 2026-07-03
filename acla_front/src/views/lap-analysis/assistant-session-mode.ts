import { RecordingState } from './recording-state';

export type SessionAnalysisAssistantMode = 'front_desk' | 'live' | 'recorded' | 'user_summary';

export const buildAssistantConversationKey = (sessionMode: string, sessionId?: string | null): string =>
    `${sessionMode}:${sessionId || 'none'}`;

export const resolveAssistantSessionMode = ({
    assistantModeOverride,
    sessionId,
    recordingState,
}: {
    assistantModeOverride?: SessionAnalysisAssistantMode;
    sessionId?: string | null;
    recordingState?: RecordingState | null;
}): SessionAnalysisAssistantMode => {
    if (assistantModeOverride) {
        return assistantModeOverride;
    }
    if (sessionId) {
        return 'recorded';
    }
    if (recordingState === RecordingState.RECORDING) {
        return 'live';
    }
    return 'front_desk';
};
