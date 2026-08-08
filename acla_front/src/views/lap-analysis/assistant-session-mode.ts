import type {
    AiChatAssistantMode,
    AiChatScreenRegistration,
} from 'contexts/AiChatScreenContext';

export type SessionAnalysisAssistantMode = AiChatAssistantMode;

export const buildAssistantConversationKey = (sessionMode: string, sessionId?: string | null): string =>
    `${sessionMode}:${sessionId || 'none'}`;

export const resolveAssistantRecordedSessionId = (
    sessionMode: SessionAnalysisAssistantMode,
    sessionId?: string | null,
): string | undefined => sessionMode === 'recorded' && sessionId ? sessionId : undefined;

export const resolveRegisteredAssistantIdentity = (
    activeScreen: AiChatScreenRegistration | null,
) => {
    const sessionMode = activeScreen?.assistantMode ?? 'front_desk';
    const sessionId = resolveAssistantRecordedSessionId(
        sessionMode,
        activeScreen?.recordedSessionId,
    );
    const label = activeScreen?.pillLabel ?? 'Front Desk';

    return {
        sessionMode,
        sessionId,
        label,
        conversationKey: buildAssistantConversationKey(sessionMode, sessionId),
        title: sessionMode === 'front_desk'
            ? 'AI Assistant - Front Desk'
            : `AI Assistant - ${label}`,
    };
};
