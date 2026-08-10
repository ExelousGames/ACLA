import {
    advanceOverlayTimers,
    applyOverlayComponentEvent,
    applyOverlayDisplayRequest,
    beginOverlayPresentation,
    endOverlayPresentation,
    getActiveFullSizeOverlayInstance,
    initialOverlayPresentationState,
    orderOverlayInstances,
    setOverlayEnabled,
    toggleOverlayFold,
} from './overlay-presentation-manager';
import type { OverlayDisplayRequest } from './overlay-display-types';

const presentation = {
    presentationId: 'presentation-current',
    aiSessionId: 'ai-main',
    mode: 'live' as const,
    displayIdentity: { name: 'Kestrel', agentTags: ['Live'] },
};

const activeState = () => beginOverlayPresentation(
    initialOverlayPresentationState,
    presentation,
    900,
).state;

const baseline = (progress = 10) => ({
    status: 'collecting' as const,
    progress_percent: progress,
    detail: `${progress}%`,
    track: null,
    car: null,
    current_lap: null,
    baseline_lap: null,
});

const comparison = (title: string) => ({
    title,
    comparison: {
        samples: [{
            driverTimeMs: 0,
            expertTimeMs: 0,
            driverTrackPosition: 0.2,
            expertTrackPosition: 0.2,
        }],
    },
});

let requestSequence = 0;
const request = (
    command: OverlayDisplayRequest['command'],
    now: number,
    state = activeState(),
) => applyOverlayDisplayRequest(state, {
    presentationId: presentation.presentationId,
    requestId: `request-${++requestSequence}`,
    command,
}, now);

describe('overlay presentation manager', () => {
    beforeEach(() => { requestSequence = 0; });

    it('supports singleton, keyed, and multiple cardinality with stable identities', () => {
        let result = request({
            operation: 'upsert', type: 'baseline_progress', snapshot: baseline(10),
        }, 1_000);
        const singletonId = result.acknowledgement?.instanceId;
        result = request({
            operation: 'upsert', type: 'baseline_progress', snapshot: baseline(70),
        }, 2_000, result.state);
        expect(result.acknowledgement?.instanceId).toBe(singletonId);
        expect(result.state.instances).toHaveLength(1);
        expect(result.state.instances[0].snapshot).toEqual(baseline(70));

        result = request({
            operation: 'upsert',
            type: 'tool_status',
            snapshot: { runId: 'run-a', name: 'tool', title: 'Started', status: 'started' },
            options: { key: 'run-a' },
        }, 3_000, result.state);
        const toolId = result.acknowledgement?.instanceId;
        result = request({
            operation: 'upsert',
            type: 'tool_status',
            snapshot: { runId: 'run-a', name: 'tool', title: 'Done', status: 'completed' },
            options: { key: 'run-a' },
        }, 4_000, result.state);
        expect(result.acknowledgement?.instanceId).toBe(toolId);

        result = request({ operation: 'upsert', type: 'ai_message', snapshot: { text: 'One' } }, 5_000, result.state);
        const firstMessage = result.acknowledgement?.instanceId;
        result = request({ operation: 'upsert', type: 'ai_message', snapshot: { text: 'Two' } }, 6_000, result.state);
        expect(result.acknowledgement?.instanceId).toBe(firstMessage);
        expect(result.state.instances.filter(({ type }) => type === 'ai_message')).toHaveLength(1);
        expect(result.state.instances.find(({ type }) => type === 'ai_message')?.snapshot).toEqual({ text: 'Two' });
    });

    it('rejects invalid types, snapshots, and keyed identity mismatches safely', () => {
        const unknown = request({
            operation: 'upsert',
            type: 'future_type' as any,
            snapshot: {},
        }, 1_000);
        expect(unknown.acknowledgement).toMatchObject({ accepted: false });
        expect(unknown.events[0]).toMatchObject({ kind: 'rejected' });

        const malformed = request({
            operation: 'upsert', type: 'baseline_progress', snapshot: { progress_percent: 20 },
        }, 1_000);
        expect(malformed.acknowledgement?.accepted).toBe(false);

        const mismatched = request({
            operation: 'upsert',
            type: 'tool_status',
            snapshot: { runId: 'actual', name: 'tool', title: 'Started', status: 'started' },
            options: { key: 'different' },
        }, 1_000);
        expect(mismatched.acknowledgement?.accepted).toBe(false);
    });

    it('pulses fold-until-update cards, expands them on update, and keeps hidden timers running', () => {
        let result = request({
            operation: 'upsert', type: 'baseline_progress', snapshot: baseline(20),
        }, 1_000);
        result = advanceOverlayTimers(result.state, 4_799);
        expect(result.state.instances[0].folded).toBe(false);
        result = advanceOverlayTimers(result.state, 4_800);
        expect(result.state.instances[0].folded).toBe(true);

        result = request({
            operation: 'upsert', type: 'baseline_progress', snapshot: baseline(30),
        }, 5_000, result.state);
        expect(result.state.instances[0].folded).toBe(false);

        result = request({
            operation: 'upsert',
            type: 'map',
            snapshot: { status: 'unavailable', title: 'Map' },
        }, 5_000, result.state);
        expect(result.state.enabled).toBe(false);
        result = advanceOverlayTimers(result.state, 8_800);
        expect(result.state.instances.some(({ type }) => type === 'map')).toBe(false);
    });

    it('starts display-owned AI message expiry on visual completion', () => {
        let result = request({
            operation: 'upsert', type: 'ai_message', snapshot: { text: 'Typed' },
        }, 1_000);
        const instanceId = result.acknowledgement!.instanceId!;
        expect(result.state.instances[0].exitAt).toBeNull();

        result = applyOverlayComponentEvent(result.state, instanceId, 'visual_complete', 2_000);
        expect(result.state.instances[0].exitAt).toBe(5_800);
        result = advanceOverlayTimers(result.state, 5_800);
        expect(result.state.instances).toHaveLength(0);
        expect(result.events[0]).toMatchObject({ kind: 'exited', reason: 'transient_complete' });
    });

    it('applies policy changes, producer exits, visibility events, and pinned-first recency ordering', () => {
        let result = request({
            operation: 'upsert', type: 'procedure_plan',
            snapshot: { goal: 'Plan', currentStep: 0, requests: [{ type: 'tool', title: 'Step', status: 'pending' }] },
        }, 1_000);
        const planId = result.acknowledgement!.instanceId!;
        result = request({
            operation: 'set_policy', target: { instanceId: planId }, policy: 'pinned_top',
        }, 2_000, result.state);
        expect(result.state.instances[0].policy).toBe('pinned_top');
        expect(result.events[0].kind).toBe('policy_changed');

        result = request({
            operation: 'upsert', type: 'baseline_progress', snapshot: baseline(40),
        }, 3_000, result.state);
        const ordered = orderOverlayInstances(result.state.instances);
        expect(ordered[0].instanceId).toBe(planId);

        result = setOverlayEnabled(result.state, true, 3_100);
        expect(result.events.every(({ kind }) => kind === 'shown')).toBe(true);
        result = request({
            operation: 'exit', target: { instanceId: planId }, reason: 'producer_exit',
        }, 4_000, result.state);
        expect(result.state.instances.some(({ instanceId }) => instanceId === planId)).toBe(false);
        expect(result.events[0]).toMatchObject({ kind: 'exited', reason: 'producer_exit' });
    });

    it('keeps monotonic full-size priority across upserts, folds, and exits', () => {
        let result = request({
            operation: 'upsert', type: 'driver_expert_comparison', snapshot: comparison('First'),
        }, 1_000);
        const firstId = result.acknowledgement!.instanceId!;
        result = request({
            operation: 'set_policy', target: { instanceId: firstId }, policy: 'visible_until_exit',
        }, 1_010, result.state);
        result = request({
            operation: 'upsert', type: 'driver_expert_comparison', snapshot: comparison('Second'),
        }, 1_020, result.state);
        const secondId = result.acknowledgement!.instanceId!;
        result = request({
            operation: 'set_policy', target: { instanceId: secondId }, policy: 'visible_until_exit',
        }, 1_030, result.state);

        result = request({
            operation: 'request_full_size', target: { instanceId: firstId },
        }, 1_040, result.state);
        expect(getActiveFullSizeOverlayInstance(result.state.instances)?.instanceId).toBe(firstId);
        const firstOrder = result.state.instances.find(({ instanceId }) => instanceId === firstId)
            ?.fullSizeRequestOrder;

        result = request({
            operation: 'upsert',
            type: 'driver_expert_comparison',
            snapshot: comparison('First updated'),
            options: { instanceId: firstId },
        }, 1_050, result.state);
        expect(result.state.instances.find(({ instanceId }) => instanceId === firstId)
            ?.fullSizeRequestOrder).toBe(firstOrder);

        result = request({
            operation: 'request_full_size', target: { instanceId: secondId },
        }, 1_060, result.state);
        expect(getActiveFullSizeOverlayInstance(result.state.instances)?.instanceId).toBe(secondId);
        expect(result.state.instances).toHaveLength(2);

        result = request({
            operation: 'upsert',
            type: 'driver_expert_comparison',
            snapshot: comparison('First updated again'),
            options: { instanceId: firstId },
        }, 1_065, result.state);
        expect(getActiveFullSizeOverlayInstance(result.state.instances)?.instanceId).toBe(secondId);
        expect(result.state.instances.find(({ instanceId }) => instanceId === firstId)
            ?.fullSizeRequestOrder).toBe(firstOrder);

        result = toggleOverlayFold(result.state, secondId, 1_070);
        expect(getActiveFullSizeOverlayInstance(result.state.instances)?.instanceId).toBe(firstId);
        result = toggleOverlayFold(result.state, secondId, 1_080);
        expect(getActiveFullSizeOverlayInstance(result.state.instances)?.instanceId).toBe(secondId);

        result = request({
            operation: 'exit', target: { instanceId: secondId }, reason: 'producer_exit',
        }, 1_090, result.state);
        expect(getActiveFullSizeOverlayInstance(result.state.instances)?.instanceId).toBe(firstId);
        expect(result.state.instances.map(({ instanceId }) => instanceId)).toEqual([firstId]);
    });

    it('continues transient expiry for a requester hidden behind a newer full-size card', () => {
        let result = request({
            operation: 'upsert', type: 'driver_expert_comparison', snapshot: comparison('Older'),
        }, 1_000);
        const olderId = result.acknowledgement!.instanceId!;
        result = request({
            operation: 'request_full_size', target: { instanceId: olderId },
        }, 1_010, result.state);
        result = request({
            operation: 'upsert', type: 'driver_expert_comparison', snapshot: comparison('Newer'),
        }, 1_100, result.state);
        const newerId = result.acknowledgement!.instanceId!;
        result = request({
            operation: 'request_full_size', target: { instanceId: newerId },
        }, 1_110, result.state);
        expect(getActiveFullSizeOverlayInstance(result.state.instances)?.instanceId).toBe(newerId);

        result = advanceOverlayTimers(result.state, 4_800);
        expect(result.state.instances.some(({ instanceId }) => instanceId === olderId)).toBe(false);
        expect(getActiveFullSizeOverlayInstance(result.state.instances)?.instanceId).toBe(newerId);
        expect(result.events).toContainEqual(expect.objectContaining({
            instanceId: olderId,
            kind: 'exited',
            reason: 'transient_complete',
        }));
    });

    it('rejects missing, unsupported, and ambiguous full-size targets without changing the active request', () => {
        let result = request({
            operation: 'upsert', type: 'driver_expert_comparison', snapshot: comparison('Active'),
        }, 1_000);
        const comparisonId = result.acknowledgement!.instanceId!;
        result = request({
            operation: 'request_full_size', target: { instanceId: comparisonId },
        }, 1_010, result.state);
        const activeOrder = result.state.instances[0].fullSizeRequestOrder;

        result = request({
            operation: 'request_full_size', target: { instanceId: 'missing' },
        }, 1_020, result.state);
        expect(result.acknowledgement?.accepted).toBe(false);
        expect(getActiveFullSizeOverlayInstance(result.state.instances)?.instanceId).toBe(comparisonId);

        result = request({
            operation: 'request_full_size',
        } as any, 1_025, result.state);
        expect(result.acknowledgement?.accepted).toBe(false);
        expect(getActiveFullSizeOverlayInstance(result.state.instances)?.instanceId).toBe(comparisonId);

        result = request({
            operation: 'upsert', type: 'baseline_progress', snapshot: baseline(30),
        }, 1_030, result.state);
        result = request({
            operation: 'request_full_size', target: { type: 'baseline_progress' },
        }, 1_040, result.state);
        expect(result.acknowledgement?.accepted).toBe(false);
        expect(getActiveFullSizeOverlayInstance(result.state.instances)?.instanceId).toBe(comparisonId);
        expect(result.state.instances.find(({ instanceId }) => instanceId === comparisonId)
            ?.fullSizeRequestOrder).toBe(activeOrder);

        result = request({
            operation: 'request_full_size', target: { type: 'driver_expert_comparison' },
        }, 1_050, result.state);
        expect(result.acknowledgement?.accepted).toBe(false);
        expect(getActiveFullSizeOverlayInstance(result.state.instances)?.instanceId).toBe(comparisonId);
    });

    it('preserves the presentation and its content while the global overlay setting is off', () => {
        let result = request({
            operation: 'upsert', type: 'ai_message', snapshot: { text: 'Keep this' },
        }, 1_000);
        result = setOverlayEnabled(result.state, true, 1_100);
        result = setOverlayEnabled(result.state, false, 1_200);
        expect(result.state.enabled).toBe(false);
        expect(result.state.presentation).toEqual(presentation);
        expect(result.state.instances[0].snapshot).toEqual({ text: 'Keep this' });

        result = setOverlayEnabled(result.state, true, 1_300);
        expect(result.state.enabled).toBe(true);
        expect(result.state.instances[0]).toMatchObject({ shown: true, snapshot: { text: 'Keep this' } });
    });

    it('replaces presentations, rejects stale requests, and only ends a matching presentation', () => {
        let result = request({
            operation: 'upsert', type: 'ai_message', snapshot: { text: 'Old context' },
        }, 1_000);
        const replacement = {
            presentationId: 'presentation-new',
            aiSessionId: 'ai-agent',
            mode: 'agent' as const,
            displayIdentity: { name: 'Track Guide', agentTags: ['Agent'] },
        };

        result = beginOverlayPresentation(result.state, replacement, 2_000);
        expect(result.state.presentation).toEqual(replacement);
        expect(result.state.instances).toHaveLength(0);
        expect(result.events[0]).toMatchObject({
            presentationId: presentation.presentationId,
            kind: 'exited',
            reason: 'replaced',
        });

        const stale = applyOverlayDisplayRequest(result.state, {
            presentationId: presentation.presentationId,
            requestId: 'stale-request',
            command: { operation: 'upsert', type: 'ai_message', snapshot: { text: 'Too late' } },
        }, 2_100);
        expect(stale.acknowledgement).toMatchObject({
            presentationId: presentation.presentationId,
            accepted: false,
        });
        expect(stale.state.instances).toHaveLength(0);

        const oldCleanup = endOverlayPresentation(result.state, presentation.presentationId, 2_200);
        expect(oldCleanup.state).toBe(result.state);
        const matchingCleanup = endOverlayPresentation(result.state, replacement.presentationId, 2_300);
        expect(matchingCleanup.state.presentation).toBeNull();
        expect(matchingCleanup.state.enabled).toBe(result.state.enabled);
    });
});
