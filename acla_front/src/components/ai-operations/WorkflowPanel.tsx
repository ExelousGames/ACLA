import React, { forwardRef, useCallback, useImperativeHandle, useMemo, useRef, useState } from 'react';
import {
    OPERATION_COMPONENT_NAMES,
    type NamedOperationComponentHandle,
    useOperationComponentRefs,
    useRegisterOperationComponentRef,
} from 'contexts/OperationComponentRefContext';
import type { DesktopGame } from 'contexts/DesktopGameContext';
import ProcedurePlan, { useProcedurePlanWorkflow, type ProcedurePlanHandle, type ProcedurePlanInput, type AppendProcedurePlanInput } from './ProcedurePlan';
import RepeatablePlan, { useRepeatablePlanWorkflow, type RepeatablePlanHandle, type RepeatablePlanInput, type AppendRepeatablePlanInput } from './RepeatablePlan';
import LiveRangeTodoList, { LiveRangeTodoListRunner, useLiveRangeTodoListWorkflow } from './LiveRangeTodoList';
import type { MountedWorkflow, MountWorkflow } from './WorkflowComponentBase';
import type { WorkflowDispatcher } from './tool';
import type { LiveRangeTodoListHandle, LiveRangeTodoListInput, CreateLiveRangeTodoListInput } from './live-range-todo-list-types';

export interface WorkflowPanelHandle extends NamedOperationComponentHandle {
    createProcedurePlan(input: ProcedurePlanInput, dispatch: WorkflowDispatcher): ReturnType<ProcedurePlanHandle['createProcedurePlan']>;
    appendProcedurePlan(input: AppendProcedurePlanInput, dispatch: WorkflowDispatcher): ReturnType<ProcedurePlanHandle['appendProcedurePlan']>;
    createRepeatablePlan(input: RepeatablePlanInput, dispatch: WorkflowDispatcher): ReturnType<RepeatablePlanHandle['createRepeatablePlan']>;
    appendRepeatablePlan(input: AppendRepeatablePlanInput, dispatch: WorkflowDispatcher): ReturnType<RepeatablePlanHandle['appendRepeatablePlan']>;
    createLiveRangeTodoList(input: CreateLiveRangeTodoListInput, dispatch: WorkflowDispatcher): ReturnType<LiveRangeTodoListHandle['createLiveRangeTodoList']>;
    appendLiveRangeTodoList(input: LiveRangeTodoListInput, dispatch: WorkflowDispatcher): ReturnType<LiveRangeTodoListHandle['appendLiveRangeTodoList']>;
    initializeLiveRangeTodoList(): LiveRangeTodoListHandle;
    handleUserText(text: string): void;
    handleToolStatus(data: Record<string, unknown>): void;
    reset(): void;
}

interface WorkflowPanelProps {
    dispatchOperation: WorkflowDispatcher;
    live: boolean;
    sessionGame: DesktopGame | null;
}

/** Coordinates the visible workflow; each workflow owns its runtime and state. */
const WorkflowPanel = forwardRef<WorkflowPanelHandle, WorkflowPanelProps>(({
    dispatchOperation,
    live,
    sessionGame,
}, ref) => {
    const { directory: componentRefs } = useOperationComponentRefs();
    const [active, setActive] = useState<(MountedWorkflow & { key: number }) | null>(null);
    const activeRef = useRef<MountedWorkflow | null>(null);
    const keyRef = useRef(0);

    const mountWorkflow = useCallback<MountWorkflow>((workflow) => {
        const previous = activeRef.current;
        if (previous?.runner === workflow.runner) return;
        workflow.runner.addComponentRef(componentRefs);
        activeRef.current = workflow;
        setActive({ ...workflow, key: ++keyRef.current });
    }, [componentRefs]);

    const onEmpty = useCallback((runner: LiveRangeTodoListRunner) => {
        if (activeRef.current?.runner !== runner) return;
        activeRef.current = null;
        setActive(null);
    }, []);

    const procedure = useProcedurePlanWorkflow({ mountWorkflow, dispatchOperation });
    const repeatable = useRepeatablePlanWorkflow({ mountWorkflow });
    const liveRange = useLiveRangeTodoListWorkflow({ mountWorkflow, onEmpty, live, sessionGame });
    const resetProcedure = procedure.reset;
    const resetRepeatable = repeatable.reset;
    const resetLiveRange = liveRange.reset;
    const reset = useCallback(() => {
        activeRef.current = null;
        resetProcedure();
        resetRepeatable();
        resetLiveRange();
        setActive(null);
    }, [resetProcedure, resetRepeatable, resetLiveRange]);

    const handle = useMemo<WorkflowPanelHandle>(() => ({
        getComponentName: () => OPERATION_COMPONENT_NAMES.WORKFLOW_PANEL,
        createProcedurePlan: procedure.createProcedurePlan,
        appendProcedurePlan: procedure.appendProcedurePlan,
        createRepeatablePlan: repeatable.createRepeatablePlan,
        appendRepeatablePlan: repeatable.appendRepeatablePlan,
        createLiveRangeTodoList: liveRange.createLiveRangeTodoList,
        appendLiveRangeTodoList: liveRange.appendLiveRangeTodoList,
        initializeLiveRangeTodoList: liveRange.initializeLiveRangeTodoList,
        handleUserText: procedure.handleUserText,
        handleToolStatus: procedure.handleToolStatus,
        reset,
    }), [procedure.createProcedurePlan, procedure.appendProcedurePlan, procedure.handleUserText, procedure.handleToolStatus,
        repeatable.createRepeatablePlan, repeatable.appendRepeatablePlan, liveRange.createLiveRangeTodoList,
        liveRange.appendLiveRangeTodoList, liveRange.initializeLiveRangeTodoList, reset]);
    const handleRef = useRef<WorkflowPanelHandle | null>(handle);
    handleRef.current = handle;
    useRegisterOperationComponentRef(handleRef);
    useImperativeHandle(ref, () => handle, [handle]);

    if (!active) return null;
    const kind = active.runner.getComponentName();
    if (kind === OPERATION_COMPONENT_NAMES.PROCEDURE_PLAN && !procedure.snapshot) return null;
    return (
        <div className="ai-chat__tool-list" key={active.key}>
            {kind === OPERATION_COMPONENT_NAMES.PROCEDURE_PLAN && procedure.snapshot && (
                <ProcedurePlan plan={procedure.snapshot} surface="chat" />
            )}
            {kind === OPERATION_COMPONENT_NAMES.REPEATABLE_PLAN && (
                <RepeatablePlan snapshot={repeatable.snapshot} surface="chat" />
            )}
            {active.runner instanceof LiveRangeTodoListRunner && (
                <LiveRangeTodoList runner={active.runner} surface="chat" />
            )}
        </div>
    );
});

export default WorkflowPanel;
