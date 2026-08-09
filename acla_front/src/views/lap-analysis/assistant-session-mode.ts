import type { AiChatAssistantMode } from 'contexts/AiToolComponentRefContext';

export type SessionAnalysisAssistantMode = AiChatAssistantMode;

export interface AssistantActiveScreen {
    assistantMode: AiChatAssistantMode;
    label: string;
    recordedSessionId?: string;
    componentName?: string;
}

export const buildAssistantConversationKey = (sessionMode: string, sessionId?: string | null): string =>
    `${sessionMode}:${sessionId || 'none'}`;

export const resolveAssistantRecordedSessionId = (
    sessionMode: SessionAnalysisAssistantMode,
    sessionId?: string | null,
): string | undefined => sessionMode === 'recorded' && sessionId ? sessionId : undefined;

export const resolveRegisteredAssistantIdentity = (
    activeScreen: AssistantActiveScreen | null,
) => {
    const sessionMode = activeScreen?.assistantMode ?? 'front_desk';
    const sessionId = resolveAssistantRecordedSessionId(
        sessionMode,
        activeScreen?.recordedSessionId,
    );
    const label = activeScreen?.label ?? 'Front Desk';

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
