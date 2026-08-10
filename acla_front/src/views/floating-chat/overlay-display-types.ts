import type { DriverExpertComparisonData } from 'components/driver-expert-comparison';
import type { DesktopGame } from 'contexts/DesktopGameContext';
import type { LiveRangeTodoListSnapshot } from 'views/live-session/live-range-todo-list-types';
import type { AiMapDisplayPayload } from 'views/lap-analysis/ai-chat/AiMapToolDisplay';
import type { BaselineCollectionTag } from 'views/live-session/BaselineCollection';
import type { ProcedurePlan } from 'views/lap-analysis/ai-chat/ai-chat-plan';
import type { ToolMessageDisplayData } from 'views/lap-analysis/ai-chat/ToolMessageDisplay';

export const OVERLAY_HOLD_MS = 3_800;
export const OVERLAY_COMPARISON_COMPLETION_PAUSE_MS = 800;

export type OverlayPolicy =
    | 'pinned_top'
    | 'fold_until_update'
    | 'visible_until_exit'
    | 'transient';

export type OverlayCardinality = 'singleton' | 'keyed' | 'multiple';

export type OverlayDisplayType =
    | 'ai_message'
    | 'tool_status'
    | 'map'
    | 'driver_expert_comparison'
    | 'baseline_progress'
    | 'procedure_plan'
    | 'live_range_todo';

export interface AiMessageSnapshot {
    text: string;
}

export interface DriverExpertComparisonSnapshot {
    title: string;
    comparison: DriverExpertComparisonData;
    game?: DesktopGame | null;
}

export interface OverlaySnapshotByType {
    ai_message: AiMessageSnapshot;
    tool_status: ToolMessageDisplayData & { runId: string };
    map: AiMapDisplayPayload;
    driver_expert_comparison: DriverExpertComparisonSnapshot;
    baseline_progress: BaselineCollectionTag;
    procedure_plan: ProcedurePlan;
    live_range_todo: LiveRangeTodoListSnapshot;
}

export interface OverlayShellMetadata {
    name?: string;
    emotion?: string;
    agentTags?: string[];
}

export type OverlaySessionMode =
    | 'front_desk'
    | 'live'
    | 'recorded'
    | 'user_summary'
    | 'agent';

export interface OverlaySessionDescriptor {
    /** Identifier of the main or agent AI connection that owns this presentation. */
    aiSessionId: string;
    mode: OverlaySessionMode;
    displayIdentity: OverlayShellMetadata;
}

export interface OverlayPresentationSession extends OverlaySessionDescriptor {
    /** App-generated identity used to isolate replacement presentations. */
    presentationId: string;
}

export type OverlayPresentationChange =
    | { kind: 'started'; presentation: OverlayPresentationSession }
    | { kind: 'ended'; presentationId: string };

export interface OverlayUpsertOptions {
    /** Required by keyed definitions; ignored by singleton definitions. */
    key?: string;
    /** Targets an existing multiple instance when supplied. */
    instanceId?: string;
    metadata?: OverlayShellMetadata;
}

export type OverlayTarget =
    | { instanceId: string }
    | { type: OverlayDisplayType; key?: string };

export type OverlayExitReason =
    | 'manual_dismiss'
    | 'transient_complete'
    | 'producer_exit'
    | 'replaced'
    | 'session_ended'
    | 'component_complete'
    | 'overlay_shutdown'
    | string;

export type OverlayLifecycleEventKind =
    | 'accepted'
    | 'shown'
    | 'updated'
    | 'folded'
    | 'policy_changed'
    | 'exited'
    | 'rejected';

export interface OverlayLifecycleEvent {
    eventId: string;
    presentationId: string;
    instanceId: string;
    type?: OverlayDisplayType;
    kind: OverlayLifecycleEventKind;
    at: number;
    policy?: OverlayPolicy;
    reason?: OverlayExitReason;
    requestId?: string;
    message?: string;
}

export interface OverlayUpsertCommand {
    operation: 'upsert';
    type: OverlayDisplayType;
    snapshot: unknown;
    options?: OverlayUpsertOptions;
}

export interface OverlaySetPolicyCommand {
    operation: 'set_policy';
    target: OverlayTarget;
    policy: OverlayPolicy;
}

export interface OverlayRequestFullSizeCommand {
    operation: 'request_full_size';
    target: OverlayTarget;
}

export interface OverlayExitCommand {
    operation: 'exit';
    target: OverlayTarget;
    reason: OverlayExitReason;
}

export type OverlayDisplayCommand =
    | OverlayUpsertCommand
    | OverlaySetPolicyCommand
    | OverlayRequestFullSizeCommand
    | OverlayExitCommand;

export interface OverlayDisplayRequest {
    presentationId: string;
    requestId: string;
    command: OverlayDisplayCommand;
}

export interface OverlayDisplayAcknowledgement {
    presentationId: string;
    requestId: string;
    accepted: boolean;
    instanceId?: string;
    error?: string;
}

export type OverlayComponentEvent = 'visual_complete';

export const isJsonSafe = (value: unknown, seen = new Set<unknown>()): boolean => {
    if (value === null || typeof value === 'string' || typeof value === 'boolean') return true;
    if (typeof value === 'number') return Number.isFinite(value);
    if (Array.isArray(value)) {
        if (seen.has(value)) return false;
        seen.add(value);
        const valid = value.every((item) => isJsonSafe(item, seen));
        seen.delete(value);
        return valid;
    }
    if (typeof value !== 'object' || value === undefined) return false;
    if (Object.getPrototypeOf(value) !== Object.prototype) return false;
    if (seen.has(value)) return false;
    seen.add(value);
    const valid = Object.values(value as Record<string, unknown>)
        .every((item) => isJsonSafe(item, seen));
    seen.delete(value);
    return valid;
};

export const cloneJsonSnapshot = <T,>(value: T): T => JSON.parse(JSON.stringify(value)) as T;
