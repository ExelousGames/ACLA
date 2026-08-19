import React from 'react';
import type { AiOverlayRenderer } from 'views/floating-chat/ai-overlay-types';
import {
    isOverlayNonEmptyString,
    isOverlayRecord,
} from 'views/floating-chat/overlay-renderer-validation';
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
    assistantWhoLabel = 'Kestrel',
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

export interface AiMessageOverlaySnapshot {
    text: string;
}

const TYPE_INTERVAL_MS = 28;

const TypedMessage: React.FC<{
    snapshot: AiMessageOverlaySnapshot;
    revision: number;
    onComplete(): void;
}> = ({ snapshot, revision, onComplete }) => {
    const [text, setText] = React.useState('');
    const onCompleteRef = React.useRef(onComplete);

    React.useEffect(() => {
        onCompleteRef.current = onComplete;
    }, [onComplete]);

    React.useEffect(() => {
        const target = snapshot.text.trim();
        setText('');
        if (!target) {
            onCompleteRef.current();
            return undefined;
        }
        let index = 0;
        const timer = window.setInterval(() => {
            index += 1;
            setText(target.slice(0, index));
            if (index >= target.length) {
                window.clearInterval(timer);
                onCompleteRef.current();
            }
        }, TYPE_INTERVAL_MS);
        return () => window.clearInterval(timer);
    }, [revision, snapshot.text]);

    return (
        <div className="overlay-card__message" data-testid="overlay-ai-message">
            {text}
            {text.length < snapshot.text.trim().length && <span className="overlay-card__caret" />}
        </div>
    );
};

export const aiMessageDisplayOverlayRenderer: AiOverlayRenderer<AiMessageOverlaySnapshot> = {
    componentType: 'ai_message',
    validateSnapshot: (snapshot): snapshot is AiMessageOverlaySnapshot => (
        isOverlayRecord(snapshot) && isOverlayNonEmptyString(snapshot.text)
    ),
    renderOverlay: (snapshot, status, context) => status === 'folded'
        ? snapshot.text
        : (
            <TypedMessage
                snapshot={snapshot}
                revision={context.revision}
                onComplete={() => context.emitRendererEvent('visual_complete')}
            />
        ),
    dimensions: {
        expanded: { width: 420, height: 92 },
        folded: { width: 280, height: 58 },
    },
};

export default AiMessageDisplay;
