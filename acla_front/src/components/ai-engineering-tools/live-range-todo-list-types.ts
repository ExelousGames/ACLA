import type { AiToolOperation } from './ai-tool-operation';

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };

export type LiveRangeTodoStatus = 'pending' | 'running';

export interface LiveRangeTodoContent {
    title: string;
    detail?: string;
    metadata?: JsonValue;
}

export interface LiveRangeTodoEventInput {
    id: string;
    normalized_position: number;
    lead_time_seconds?: number;
    content: LiveRangeTodoContent;
    data: JsonValue;
}

export interface LiveRangeTodoEventUpdate {
    id: string;
    normalized_position?: number;
    lead_time_seconds?: number;
    content?: Partial<LiveRangeTodoContent>;
    data?: JsonValue;
}

export interface LiveRangeTodoSnapshotEvent {
    id: string;
    normalized_position: number;
    lead_time_seconds: number;
    content: LiveRangeTodoContent;
    data: JsonValue;
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

export type LiveRangeTodoDueStatus = {
    source: 'live_range_todo_list';
    event: string;
    event_id: string;
    content: LiveRangeTodoContent;
    normalized_position: number;
    lead_time_seconds: number;
    data: JsonValue;
    lap?: number;
};

export type LiveRangeTodoListAiResult = {
    status: 'ready' | 'empty';
    event_count: number;
    pending_count: number;
    running_count: number;
    message?: string;
};

export interface LiveRangeTodoListHandle {
    getComponentName(): string;
    addEvent: (event: LiveRangeTodoEventInput) => LiveRangeTodoListToolResult;
    replaceEvents: (events: readonly LiveRangeTodoEventInput[]) => LiveRangeTodoListToolResult;
    updateEvents: (updates: readonly LiveRangeTodoEventUpdate[]) => LiveRangeTodoListToolResult;
    removeEvents: (ids: readonly string[]) => LiveRangeTodoListToolResult;
    resetEvents: (ids?: readonly string[]) => LiveRangeTodoListToolResult;
    clear: () => LiveRangeTodoListToolResult;
    get: () => LiveRangeTodoListToolResult;
    setForAi: (args: Record<string, unknown>) => AiToolOperation<LiveRangeTodoListAiResult, LiveRangeTodoDueStatus>;
    updateForAi: (args: Record<string, unknown>) => AiToolOperation<LiveRangeTodoListAiResult, LiveRangeTodoDueStatus>;
    getForAi: () => AiToolOperation<LiveRangeTodoListAiResult>;
}
