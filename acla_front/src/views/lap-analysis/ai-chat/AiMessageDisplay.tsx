import React from 'react';
import AiMapToolDisplay, { AiMapDisplayPayload } from './AiMapToolDisplay';
import ToolMessageDisplay, { type ToolMessageDisplayData } from './ToolMessageDisplay';

export type AiChatMessageKind = 'chat' | 'tool';

export type AiChatDisplayMessage = {
    id: string;
    content: string;
    isUser: boolean;
    timestamp: Date;
    isLoading?: boolean;
    kind?: AiChatMessageKind;
    tool?: ToolMessageDisplayData;
    mapDisplay?: AiMapDisplayPayload;
};

type AiMessageDisplayProps = {
    message: AiChatDisplayMessage;
    assistantAvatarLabel?: string;
    assistantWhoLabel?: string;
    debugMode?: boolean;
    surface?: 'chat' | 'pill';
};

const AiMessageDisplay: React.FC<AiMessageDisplayProps> = ({
    message,
    assistantAvatarLabel = 'AI',
    assistantWhoLabel = 'ACLA',
    debugMode = false,
    surface = 'chat',
}) => {
    if (message.kind === 'tool' && message.tool) {
        return (
            <ToolMessageDisplay
                tool={message.tool}
                timestamp={message.timestamp}
                debugMode={debugMode}
                surface={surface}
            />
        );
    }

    const role: 'driver' | 'acla' | 'guidance' = message.isUser
        ? 'driver'
        : message.id.includes('guidance') ? 'guidance' : 'acla';
    const avatarLabel = role === 'driver'
        ? 'YOU'
        : role === 'guidance'
            ? 'TARGET'
            : assistantAvatarLabel;
    const whoLabel = role === 'driver' ? 'YOU'
        : role === 'guidance' ? 'LIVE GUIDANCE'
            : assistantWhoLabel;

    return (
        <div className={`ai-chat__msg ai-chat__msg--${role} ai-chat__msg--${surface}`}>
            <div className="ai-chat__msg-avatar">{avatarLabel}</div>
            <div className="ai-chat__msg-body">
                <div className="ai-chat__msg-meta">
                    <span className="ai-chat__msg-who">{whoLabel}</span>
                    {surface === 'chat' && (
                        <span className="ai-chat__msg-stamp">
                            {message.timestamp.toLocaleTimeString()}
                        </span>
                    )}
                </div>

                {message.isLoading ? (
                    <div className="ai-chat__typing">
                        <span className="ai-chat__typing-dot" />
                        <span className="ai-chat__typing-dot" />
                        <span className="ai-chat__typing-dot" />
                    </div>
                ) : (
                    <>
                        <div className="ai-chat__msg-text">{message.content}</div>
                        {message.mapDisplay && (
                            <AiMapToolDisplay display={message.mapDisplay} surface={surface} />
                        )}
                    </>
                )}
            </div>
        </div>
    );
};

export default AiMessageDisplay;
