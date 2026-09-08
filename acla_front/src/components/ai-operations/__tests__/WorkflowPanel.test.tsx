import React, { createRef, StrictMode } from 'react';
import { act, cleanup, render, screen } from '@testing-library/react';
import {
    createOperationComponentRefDirectory,
    OPERATION_COMPONENT_NAMES,
    type OperationComponentRef,
    type OperationComponentRefDirectory,
} from 'contexts/OperationComponentRefContext';
import { liveTelemetryStore } from 'views/live-session/live-telemetry-store';
import WorkflowPanel, { type WorkflowPanelHandle } from '../WorkflowPanel';
import type { ProcedurePlanInput } from '../ProcedurePlan';
import type { RepeatablePlanInput } from '../RepeatablePlan';
import type { LiveRangeTodoListHandle } from '../live-range-todo-list-types';
import { createControlledOperation } from '../operation';
import { asTool, type ToolDispatcher } from '../tool';

let mockDirectory: OperationComponentRefDirectory;

jest.mock('views/floating-chat/AiOverlayManager', () => ({
    __esModule: true,
    default: () => null,
}));

jest.mock('contexts/OperationComponentRefContext', () => ({
    ...jest.requireActual('contexts/OperationComponentRefContext'),
    useOperationComponentRefs: () => ({ directory: mockDirectory, revision: 0 }),
    useRegisterOperationComponentRef: function useRegisterOperationComponentRef(ref: OperationComponentRef) {
        const { useLayoutEffect } = jest.requireActual('react');
        useLayoutEffect(() => {
            const directory = mockDirectory;
            directory.registerComponentRef(ref);
            return () => { directory.unregisterComponentRef(ref); };
        }, [ref]);
    },
}));

const procedure = (): ProcedurePlanInput => ({
    set_procedure_plan: {
        goal: 'Review telemetry',
        tools: [{ query_analysis_result: { title: 'Read telemetry', arguments: { query: 'analyses' } } }],
    },
});

const repeatable = (): RepeatablePlanInput => ({
    create_repeatable_plan: {
        name: 'Improve consistency',
        tools: [{ query_analysis_result: { id: 'read', title: 'Read telemetry', arguments: { query: 'analyses' } } }],
        stop_when: {
            tool: { query_analysis_result: { arguments: { query: '0' } } },
            operator: 'eq',
            target: 0,
        },
    },
});

const pendingTool = () => {
    const controller = createControlledOperation<Record<string, unknown>>();
    const operation = asTool(controller.operation);
    void operation.result.catch(() => undefined);
    return { controller, operation, abort: jest.spyOn(operation, 'abort') };
};

const renderPanel = (dispatchOperation: ToolDispatcher) => {
    const ref = createRef<WorkflowPanelHandle>();
    const view = render(
        <StrictMode>
            <WorkflowPanel ref={ref} dispatchOperation={dispatchOperation} live sessionGame="acc" />
        </StrictMode>,
    );
    expect(mockDirectory.getComponentNames()).toEqual(['workflow-panel']);
    expect(mockDirectory.findComponentRef('workflow-panel')?.current).toBe(ref.current);
    expect(mockDirectory.findComponentRef(OPERATION_COMPONENT_NAMES.DASHBOARD_ASSISTANT)).toBeNull();
    return { ...view, ref };
};

const makeLiveTaskDue = () => {
    const firstSequence = liveTelemetryStore.getSnapshot().sampleIndex + 2;
    [0, 0.5].forEach((position, index) => {
        const sequence = firstSequence + index;
        expect(liveTelemetryStore.publishFrame({
            type: 'frame',
            game: 'acc',
            sequence,
            committedSequence: sequence,
            committedCount: sequence,
            sample: { Graphics_normalized_car_position: position, Graphics_completed_laps: 1 },
        })).toBe(true);
    });
};

const addLiveTask = (panel: WorkflowPanelHandle, task: ReturnType<typeof pendingTool>) => {
    const runner = panel.initializeLiveRangeTodoList();
    const taskStart = jest.fn((_signal: AbortSignal) => task.operation);
    runner.addEvent({
        id: 'corner-review',
        normalized_position: 0.5,
        lead_time_seconds: 0,
        content: { title: 'Review this corner' },
        taskStart,
    });
    return { runner, taskStart };
};

describe('WorkflowPanel standalone lifecycle', () => {
    beforeEach(() => {
        mockDirectory = createOperationComponentRefDirectory();
        liveTelemetryStore.resetSession();
    });

    afterEach(() => {
        cleanup();
        liveTelemetryStore.resetSession();
        jest.restoreAllMocks();
    });

    it.each(['procedure', 'repeatable'] as const)('removes an aborted %s plan from the panel and component directory', async (kind) => {
        const child = pendingTool();
        const dispatch = Object.assign(jest.fn(() => child.operation), { validate: jest.fn() });
        const { ref } = renderPanel(dispatch);
        const input = procedure();
        input.set_procedure_plan.tools.push({ query_analysis_result: { title: 'Next step', arguments: { query: '0' } } });
        let operation!: ReturnType<WorkflowPanelHandle['createProcedurePlan']> | ReturnType<WorkflowPanelHandle['createRepeatablePlan']>;
        act(() => {
            operation = kind === 'procedure'
                ? ref.current!.createProcedurePlan(input, dispatch)
                : ref.current!.createRepeatablePlan(repeatable(), dispatch);
        });
        const title = kind === 'procedure' ? 'Review telemetry' : 'Improve consistency';
        expect(screen.getByText(title)).toBeInTheDocument();

        await act(async () => {
            operation.abort();
            await expect(operation.result).rejects.toMatchObject({ name: 'AbortError' });
        });

        expect(child.abort).toHaveBeenCalledTimes(1);
        expect(screen.queryByText(title)).not.toBeInTheDocument();
        expect(mockDirectory.getComponentNames()).toEqual(['workflow-panel']);
        await act(async () => { child.controller.resolve('complete', { value: 'late result' }); });
        expect(dispatch).toHaveBeenCalledTimes(1);
        expect(screen.queryByText(title)).not.toBeInTheDocument();
    });

    it('registers each runner before dispatch and aborts replaced procedure and repeatable tools once', async () => {
        const children = [pendingTool(), pendingTool(), pendingTool()];
        const registrations: string[][] = [];
        const dispatch = Object.assign(jest.fn(() => {
            registrations.push(mockDirectory.getComponentNames());
            return children[registrations.length - 1].operation;
        }), { validate: jest.fn() });
        const { ref, unmount } = renderPanel(dispatch);
        let first!: ReturnType<WorkflowPanelHandle['createProcedurePlan']>;
        let second!: ReturnType<WorkflowPanelHandle['createRepeatablePlan']>;
        let third!: ReturnType<WorkflowPanelHandle['createProcedurePlan']>;

        act(() => {
            first = ref.current!.createProcedurePlan(procedure(), dispatch);
            void first.result.catch(() => undefined);
            expect(registrations).toEqual([['procedure-plan', 'workflow-panel']]);
        });
        expect(screen.getByLabelText('Procedure plan')).toBeInTheDocument();
        act(() => {
            second = ref.current!.createRepeatablePlan(repeatable(), dispatch);
            void second.result.catch(() => undefined);
            expect(registrations).toHaveLength(2);
            expect(registrations[1]).toEqual(['repeatable-plan', 'workflow-panel']);
        });
        expect(children[0].abort).toHaveBeenCalledTimes(1);
        expect(screen.queryByLabelText('Procedure plan')).not.toBeInTheDocument();
        expect(screen.getByText('Improve consistency')).toBeInTheDocument();
        await expect(first.result).rejects.toBeInstanceOf(Error);

        act(() => {
            third = ref.current!.createProcedurePlan(procedure(), dispatch);
            void third.result.catch(() => undefined);
            expect(registrations).toHaveLength(3);
            expect(registrations[2]).toEqual(['procedure-plan', 'workflow-panel']);
        });
        expect(children[1].abort).toHaveBeenCalledTimes(1);
        expect(children[2].abort).not.toHaveBeenCalled();
        await expect(second.result).rejects.toBeInstanceOf(Error);
        unmount();
        await expect(third.result).rejects.toBeInstanceOf(Error);
        children.forEach((child) => expect(child.abort).toHaveBeenCalledTimes(1));
        expect(mockDirectory.getComponentNames()).toEqual([]);
    });

    it('preserves the running workflow when a later tool or stop condition fails validation', async () => {
        const child = pendingTool();
        const dispatch = Object.assign(jest.fn(() => child.operation), { validate: jest.fn() });
        const { ref } = renderPanel(dispatch);
        act(() => {
            void ref.current!.createProcedurePlan(procedure(), dispatch).result.catch(() => undefined);
        });
        const registered = mockDirectory.findComponentRef('procedure-plan');
        const invalidDispatch = Object.assign(jest.fn(() => child.operation), {
            validate: jest.fn((name: string) => {
                if (name === 'show_map') throw new Error('Forbidden later tool');
            }),
        });
        const invalidProcedure = procedure();
        invalidProcedure.set_procedure_plan.tools.push({ show_map: { title: 'Invalid later tool', arguments: {} } });
        const invalidRepeatable = repeatable();
        invalidRepeatable.create_repeatable_plan.tools.push({ show_map: { id: 'invalid', title: 'Invalid later tool' } });
        const invalidStop = repeatable();
        invalidStop.create_repeatable_plan.stop_when.tool = { show_map: {} };

        for (const create of [
            () => ref.current!.createProcedurePlan(invalidProcedure, invalidDispatch),
            () => ref.current!.createRepeatablePlan(invalidRepeatable, invalidDispatch),
            () => ref.current!.createRepeatablePlan(invalidStop, invalidDispatch),
        ]) {
            await act(async () => {
                const rejected = create();
                expect(rejected.kind).toBe('workflow');
                await expect(rejected.result).rejects.toThrow('Forbidden later tool');
            });
            expect(mockDirectory.findComponentRef('procedure-plan')).toBe(registered);
            expect(mockDirectory.getComponentNames()).toEqual(['procedure-plan', 'workflow-panel']);
            expect(child.abort).not.toHaveBeenCalled();
            expect(screen.getByText('Review telemetry')).toBeInTheDocument();
        }
        expect(dispatch).toHaveBeenCalledTimes(1);
        expect(invalidDispatch).not.toHaveBeenCalled();
    });

    it('runs a hidden live task and unregisters it on telemetry session reset without clearing the visible plan', () => {
        const task = pendingTool();
        const planChild = pendingTool();
        const dispatch = Object.assign(jest.fn(() => planChild.operation), { validate: jest.fn() });
        const { ref } = renderPanel(dispatch);
        let live!: ReturnType<typeof addLiveTask>;
        act(() => { live = addLiveTask(ref.current!, task); });
        expect(screen.getByLabelText('Live range to-do list')).toBeInTheDocument();
        act(() => {
            void ref.current!.createProcedurePlan(procedure(), dispatch).result.catch(() => undefined);
        });
        expect(screen.queryByLabelText('Live range to-do list')).not.toBeInTheDocument();
        expect(mockDirectory.findComponentRef<LiveRangeTodoListHandle>('live-range-todo-list')?.current).toBe(live.runner);
        expect(live.taskStart).not.toHaveBeenCalled();

        act(makeLiveTaskDue);
        expect(live.taskStart).toHaveBeenCalledTimes(1);
        expect(live.runner.get().todo_list?.events[0].status).toBe('running');
        expect(task.abort).not.toHaveBeenCalled();
        act(() => { liveTelemetryStore.resetSession(); });
        expect(task.abort).toHaveBeenCalledTimes(1);
        expect(live.taskStart.mock.calls[0][0].aborted).toBe(true);
        expect(mockDirectory.findComponentRef('live-range-todo-list')).toBeNull();
        expect(planChild.abort).not.toHaveBeenCalled();
        expect(screen.getByLabelText('Procedure plan')).toBeInTheDocument();
        act(makeLiveTaskDue);
        expect(live.taskStart).toHaveBeenCalledTimes(1);
        expect(mockDirectory.findComponentRef('live-range-todo-list')).toBeNull();
    });

    it('cancels visible and hidden work on reset and unmount and can create fresh runners after reset in StrictMode', async () => {
        const unusedDispatch = Object.assign(jest.fn(() => { throw new Error('Unexpected dispatch'); }), { validate: jest.fn() });
        const { ref, container, unmount } = renderPanel(unusedDispatch);
        for (const action of ['reset', 'unmount'] as const) {
            const task = pendingTool();
            const planChild = pendingTool();
            const dispatch = Object.assign(jest.fn(() => planChild.operation), { validate: jest.fn() });
            let live!: ReturnType<typeof addLiveTask>;
            act(() => {
                live = addLiveTask(ref.current!, task);
                void ref.current!.createRepeatablePlan(repeatable(), dispatch).result.catch(() => undefined);
                makeLiveTaskDue();
            });
            expect(live.taskStart).toHaveBeenCalledTimes(1);
            expect(dispatch).toHaveBeenCalledTimes(1);
            expect(task.abort).not.toHaveBeenCalled();
            expect(planChild.abort).not.toHaveBeenCalled();
            if (action === 'reset') act(() => { ref.current!.reset(); });
            else unmount();

            await expect(task.operation.result).rejects.toMatchObject({ name: 'AbortError' });
            await expect(planChild.operation.result).rejects.toMatchObject({ name: 'AbortError' });
            expect(task.abort).toHaveBeenCalledTimes(1);
            expect(planChild.abort).toHaveBeenCalledTimes(1);
            expect(live.taskStart.mock.calls[0][0].aborted).toBe(true);
            expect(container).toBeEmptyDOMElement();
            expect(mockDirectory.getComponentNames()).toEqual(action === 'reset' ? ['workflow-panel'] : []);
            act(() => { liveTelemetryStore.resetSession(); });
            expect(task.abort).toHaveBeenCalledTimes(1);
            expect(planChild.abort).toHaveBeenCalledTimes(1);
        }
    });
});
