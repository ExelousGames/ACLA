import type { Tool, ToolCall } from './tool';
import type { Workflow } from './workflow';
import type { AiOverlayComponentHandle } from 'views/floating-chat/ai-overlay-types';

export type LiveRangeTodoStatus = 'pending' | 'running';

export interface LiveRangeTodoContent {
    title: string;
    description?: string;
}

export type LiveRangeTodoListInput = {
    add_event_to_live_range_todo_list: {
        tools: ToolCall<{
            event: Omit<LiveRangeTodoEventInput, 'taskStart'>;
            arguments: Record<string, unknown>;
        }>[];
    };
};

export interface LiveRangeTodoEventInput {
    id: string;
    normalized_position: number;
    lead_time_seconds?: number;
    content: LiveRangeTodoContent;
    taskStart: (signal: AbortSignal) => Tool<unknown, object>;
}

export interface LiveRangeTodoEventUpdate {
    id: string;
    normalized_position?: number;
    lead_time_seconds?: number;
    content?: Partial<LiveRangeTodoContent>;
    taskStart?: (signal: AbortSignal) => Tool<unknown, object>;
}

export interface LiveRangeTodoSnapshotEvent {
    id: string;
    normalized_position: number;
    lead_time_seconds: number;
    content: LiveRangeTodoContent;
    status: LiveRangeTodoStatus;
    // Null means no finite arrival estimate from measured forward movement.
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

export interface LiveRangeTodoListResult {
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
    addEvent: (event: LiveRangeTodoEventInput) => LiveRangeTodoListResult;
    replaceEvents: (events: readonly LiveRangeTodoEventInput[]) => LiveRangeTodoListResult;
    updateEvents: (updates: readonly LiveRangeTodoEventUpdate[]) => LiveRangeTodoListResult;
    removeEvents: (ids: readonly string[]) => LiveRangeTodoListResult;
    resetEvents: (ids?: readonly string[]) => LiveRangeTodoListResult;
    clear: () => LiveRangeTodoListResult;
    get: () => LiveRangeTodoListResult;
    getForAi: () => Workflow<LiveRangeTodoListAiResult>;
}
