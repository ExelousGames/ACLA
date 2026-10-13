import { ChatBubbleIcon, ChevronLeftIcon, ChevronRightIcon, ReaderIcon } from '@radix-ui/react-icons';
import React, { useState } from 'react';
import {
    OPERATION_COMPONENT_NAMES,
    useOptionalOperationComponentSnapshot,
} from 'contexts/OperationComponentRefContext';
import type { AnalysisContextType } from 'views/recorded-session/analysis-context';
import AiChat from 'views/ai-chat/ai-chat';
import LivePhrases from 'views/live-session/live-phrases/LivePhrases';
import type { AssistantActiveScreen } from 'views/ai-chat/assistant-session-mode';
import { DASHBOARD_TABS } from './dashboard-navigation';

interface DashboardAssistantProps {
    activeDashboardTab: string;
}

const DashboardAssistant = ({ activeDashboardTab }: DashboardAssistantProps) => {
    const analysisContext = useOptionalOperationComponentSnapshot<AnalysisContextType>(
        activeDashboardTab === DASHBOARD_TABS.ANALYSIS
            ? OPERATION_COMPONENT_NAMES.SESSION_ANALYSIS
            : null,
    );
    const [isOpen, setIsOpen] = useState(false);
    const [selectedPanel, setSelectedPanel] = useState<'chat' | 'phrases'>('chat');
    const isLiveSession = activeDashboardTab === DASHBOARD_TABS.LIVE_SESSION;
    const showPhrases = isLiveSession && selectedPanel === 'phrases';
    const isRecordedSession = activeDashboardTab === DASHBOARD_TABS.ANALYSIS
        && analysisContext?.activeTab === 'session'
        && Boolean(analysisContext?.sessionSelected?.SessionId);

    const activeScreen: AssistantActiveScreen = activeDashboardTab === DASHBOARD_TABS.LIVE_SESSION
        ? {
            assistantMode: 'live',
            label: 'Live Session',
            componentName: OPERATION_COMPONENT_NAMES.LIVE_SESSION,
        }
        : activeDashboardTab === DASHBOARD_TABS.USER_SUMMARY
            ? {
                assistantMode: 'user_summary',
                label: 'User Summary',
                componentName: OPERATION_COMPONENT_NAMES.USER_SUMMARY,
            }
            : isRecordedSession
                ? {
                    assistantMode: 'recorded',
                    label: analysisContext?.sessionSelected?.session_name || 'Recorded Session',
                    recordedSessionId: analysisContext?.sessionSelected?.SessionId,
                    componentName: OPERATION_COMPONENT_NAMES.SESSION_ANALYSIS,
                }
                : {
                    assistantMode: 'front_desk',
                    label: 'Front Desk',
                    componentName: activeDashboardTab === DASHBOARD_TABS.ANALYSIS
                        ? OPERATION_COMPONENT_NAMES.SESSION_ANALYSIS
                        : undefined,
                };

    const assistantClassName = `main-dashboard-assistant${isOpen ? ' main-dashboard-assistant--open' : ' main-dashboard-assistant--folded'}`;

    return (
        <aside className={assistantClassName} aria-label="AI Assistant">
            <button
                type="button"
                className="main-dashboard-assistant__toggle"
                onClick={() => setIsOpen((open) => !open)}
                aria-controls="main-dashboard-assistant-body"
                aria-expanded={isOpen}
                aria-label={isOpen ? 'Fold AI Assistant' : 'Open AI Assistant'}
                title={isOpen ? 'Fold AI Assistant' : 'Open AI Assistant'}
            >
                {isOpen ? <ChevronRightIcon aria-hidden="true" /> : <ChevronLeftIcon aria-hidden="true" />}
            </button>
            <div className="main-dashboard-assistant__tabs" role="group" aria-label="Sidebar panels">
                <button
                    type="button"
                    aria-label="Assistant"
                    title="Assistant"
                    aria-pressed={!showPhrases}
                    aria-controls="dashboard-chat-panel"
                    onClick={() => { setSelectedPanel('chat'); setIsOpen(true); }}
                >
                    <ChatBubbleIcon aria-hidden="true" />
                </button>
                {isLiveSession && <button
                    type="button"
                    aria-label="Live phrases"
                    title="Live phrases"
                    aria-pressed={showPhrases}
                    aria-controls="dashboard-phrases-panel"
                    onClick={() => { setSelectedPanel('phrases'); setIsOpen(true); }}
                >
                    <ReaderIcon aria-hidden="true" />
                </button>}
            </div>
            <div id="main-dashboard-assistant-body" className="main-dashboard-assistant__body" aria-hidden={!isOpen}>
                <div id="dashboard-chat-panel" className="main-dashboard-assistant__panel" hidden={showPhrases}>
                    <AiChat
                        name={OPERATION_COMPONENT_NAMES.DASHBOARD_ASSISTANT}
                        activeScreen={activeScreen}
                    />
                </div>
                {isLiveSession && <div id="dashboard-phrases-panel" className="main-dashboard-assistant__panel" hidden={!showPhrases}>
                    <LivePhrases name={OPERATION_COMPONENT_NAMES.LIVE_PHRASES} />
                </div>}
            </div>
        </aside>
    );
};

export default DashboardAssistant;
