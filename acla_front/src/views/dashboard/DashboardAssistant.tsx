import { ChatBubbleIcon, ChevronLeftIcon, ChevronRightIcon } from '@radix-ui/react-icons';
import React, { useContext, useState } from 'react';
import {
    AI_TOOL_COMPONENT_NAMES,
} from 'contexts/AiToolComponentRefContext';
import { AnalysisContext } from 'views/lap-analysis/analysis-context';
import AiChat from 'views/lap-analysis/ai-chat/ai-chat';
import type { AssistantActiveScreen } from 'views/lap-analysis/assistant-session-mode';
import { DASHBOARD_TABS } from './dashboard-navigation';

interface DashboardAssistantProps {
    activeDashboardTab: string;
}

const DashboardAssistant = ({ activeDashboardTab }: DashboardAssistantProps) => {
    const analysisContext = useContext(AnalysisContext);
    const [isOpen, setIsOpen] = useState(false);
    const isRecordedSession = activeDashboardTab === DASHBOARD_TABS.ANALYSIS
        && analysisContext.activeTab === 'session'
        && Boolean(analysisContext.sessionSelected?.SessionId);

    const activeScreen: AssistantActiveScreen = activeDashboardTab === DASHBOARD_TABS.LIVE_SESSION
        ? {
            assistantMode: 'live',
            label: 'Live Session',
            componentName: AI_TOOL_COMPONENT_NAMES.LIVE_SESSION,
        }
        : activeDashboardTab === DASHBOARD_TABS.USER_SUMMARY
            ? {
                assistantMode: 'user_summary',
                label: 'User Summary',
                componentName: AI_TOOL_COMPONENT_NAMES.USER_SUMMARY,
            }
            : isRecordedSession
                ? {
                    assistantMode: 'recorded',
                    label: analysisContext.sessionSelected?.session_name || 'Recorded Session',
                    recordedSessionId: analysisContext.sessionSelected?.SessionId,
                    componentName: AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS,
                }
                : {
                    assistantMode: 'front_desk',
                    label: 'Front Desk',
                    componentName: activeDashboardTab === DASHBOARD_TABS.ANALYSIS
                        ? AI_TOOL_COMPONENT_NAMES.SESSION_ANALYSIS
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
                {isOpen ? <ChevronRightIcon /> : <ChevronLeftIcon />}
                <ChatBubbleIcon />
            </button>
            <div id="main-dashboard-assistant-body" className="main-dashboard-assistant__body" aria-hidden={!isOpen}>
                <AiChat
                    name={AI_TOOL_COMPONENT_NAMES.DASHBOARD_ASSISTANT}
                    activeScreen={activeScreen}
                />
            </div>
        </aside>
    );
};

export default DashboardAssistant;
