import React from 'react';
import type { AiOverlayRenderer } from 'views/floating-chat/ai-overlay-types';
import {
    isOverlayNonEmptyString,
    isOverlayRecord,
} from 'views/floating-chat/overlay-renderer-validation';

export type ToolMessageDisplayData = {
    runId?: string;
    name: string;
    title: string;
    status: 'started' | 'completed';
    ok?: boolean;
    error?: string | null;
    result?: unknown;
};

type ToolMessageDisplayProps = {
    tool: ToolMessageDisplayData;
    timestamp?: Date | string | number;
    debugMode?: boolean;
    surface?: 'chat' | 'pill';
};

const formatToolDebugResult = (result: unknown): string | null => {
    if (result === undefined) return null;
    try {
        const json = JSON.stringify(result, null, 2);
        return json.length > 4000 ? `${json.slice(0, 4000)}\n... truncated` : json;
    } catch {
        return String(result);
    }
};

const formatTimestamp = (timestamp?: Date | string | number): string | null => {
    if (!timestamp) return null;
    const date = timestamp instanceof Date ? timestamp : new Date(timestamp);
    return Number.isNaN(date.getTime()) ? null : date.toLocaleTimeString();
};

const ToolMessageDisplay: React.FC<ToolMessageDisplayProps> = ({
    tool,
    timestamp,
    debugMode = false,
    surface = 'chat',
}) => {
    const isError = tool.status === 'completed' && tool.ok === false;
    const isRunning = tool.status === 'started';
    const mod = isError ? 'ai-chat__tool--error'
        : isRunning ? 'ai-chat__tool--running'
            : 'ai-chat__tool--ok';
    const debugResult = debugMode && surface === 'chat'
        ? formatToolDebugResult(tool.result)
        : null;
    const timeLabel = formatTimestamp(timestamp);

    return (
        <div className={`ai-chat__tool-wrap ai-chat__tool-wrap--${surface}`}>
            <div className={`ai-chat__tool ${mod} ai-chat__tool--${surface}`}>
                <span className="ai-chat__tool-icon">
                    {isRunning ? '...' : isError ? '!' : 'OK'}
                </span>
                <span>{tool.title}</span>
                {timeLabel && surface === 'chat' && (
                    <span className="ai-chat__tool-stamp">
                        {timeLabel}
                    </span>
                )}
            </div>
            {isError && tool.error && (
                <div className="ai-chat__tool-detail" style={{ color: 'var(--lp-red)' }}>
                    {tool.error}
                </div>
            )}
            {debugMode && surface === 'chat' && (
                <div className="ai-chat__tool-detail">{tool.name}</div>
            )}
            {debugResult && (
                <pre className="ai-chat__tool-result">{debugResult}</pre>
            )}
        </div>
    );
};

export type ToolStatusOverlaySnapshot = ToolMessageDisplayData & { runId: string };

export const toolMessageDisplayOverlayRenderer: AiOverlayRenderer<ToolStatusOverlaySnapshot> = {
    componentType: 'tool_status',
    validateSnapshot: (snapshot): snapshot is ToolStatusOverlaySnapshot => (
        isOverlayRecord(snapshot)
        && isOverlayNonEmptyString(snapshot.runId)
        && isOverlayNonEmptyString(snapshot.name)
        && isOverlayNonEmptyString(snapshot.title)
        && (snapshot.status === 'started' || snapshot.status === 'completed')
    ),
    renderOverlay: (snapshot, status) => status === 'folded'
        ? `${snapshot.status === 'started' ? 'Running' : 'Finished'}: ${snapshot.title}`
        : <ToolMessageDisplay tool={snapshot} surface="pill" />,
    dimensions: {
        expanded: { width: 420, height: 118 },
        folded: { width: 300, height: 58 },
    },
};

export default ToolMessageDisplay;
