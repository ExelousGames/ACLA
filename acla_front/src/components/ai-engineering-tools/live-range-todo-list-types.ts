import type { AiToolOperation } from './ai-tool-operation';
import type { AiOverlayComponentHandle } from 'views/floating-chat/ai-overlay-types';

export type LiveRangeTodoStatus = 'pending' | 'running';

export interface LiveRangeTodoContent {
    title: string;
    description?: string;
}

export interface LiveRangeTodoEventInput {
    id: string;
    normalized_position: number;
    lead_time_seconds?: number;
    eta_seconds?: number;
    content: LiveRangeTodoContent;
    taskStart: (signal: AbortSignal) => AiToolOperation<unknown, object>;
}

export interface LiveRangeTodoEventUpdate {
    id: string;
    normalized_position?: number;
    lead_time_seconds?: number;
    content?: Partial<LiveRangeTodoContent>;
    taskStart?: (signal: AbortSignal) => AiToolOperation<unknown, object>;
}

export interface LiveRangeTodoSnapshotEvent {
    id: string;
    normalized_position: number;
    lead_time_seconds: number;
    content: LiveRangeTodoContent;
    status: LiveRangeTodoStatus;
    eta_seconds: number | null;
    created_at: number;
    updated_at: number;
    started_at?: number;
    lap?: number;
}

export interface LiveRangeTodoListSnapshot {
    readonly events: readonly Readonly<LiveRangeTodoSnapshotEvent>[];
    readonly current_position: number | null;
    readonly rolling_rate: number | null;
    readonly lap?: number;
    readonly created_at: number;
    readonly updated_at: number;
}

export interface LiveRangeTodoListToolResult {
    status: 'ready' | 'empty';
    todo_list: LiveRangeTodoListSnapshot | null;
    message?: string;
}

export type LiveRangeTodoListAiResult = {
    status: 'ready' | 'empty';
    event_count: number;
    pending_count: number;
    running_count: number;
    message?: string;
};

export interface LiveRangeTodoListHandle extends AiOverlayComponentHandle<LiveRangeTodoListSnapshot | null> {
    addEvent: (event: LiveRangeTodoEventInput) => LiveRangeTodoListToolResult;
    replaceEvents: (events: readonly LiveRangeTodoEventInput[]) => LiveRangeTodoListToolResult;
    updateEvents: (updates: readonly LiveRangeTodoEventUpdate[]) => LiveRangeTodoListToolResult;
    removeEvents: (ids: readonly string[]) => LiveRangeTodoListToolResult;
    resetEvents: (ids?: readonly string[]) => LiveRangeTodoListToolResult;
    clear: () => LiveRangeTodoListToolResult;
    get: () => LiveRangeTodoListToolResult;
    getForAi: () => AiToolOperation<LiveRangeTodoListAiResult>;
}
