import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
    OPERATION_COMPONENT_NAMES,
    type NamedOperationComponentHandle,
} from 'contexts/OperationComponentRefContext';
import type {
    AiOverlayComponentHandle,
    AiOverlayRenderer,
    AiOverlayRendererEvent,
} from 'views/floating-chat/ai-overlay-types';
import {
    isOverlayNonEmptyString,
    isOverlayRecord,
} from 'views/floating-chat/overlay-renderer-validation';
import {
    OperationComponentErrorConstructor,
    DuplicateGoalStepIdError,
    GoalComponentError,
    GoalStopWhenFailedError,
    GoalStopWhenInputIncompatibleError,
    GoalReplacedError,
    GoalStepFailedError,
    GoalTaskRetryUnavailableError,
    InvalidGoalStopWhenError,
    InvalidGoalNameError,
    InvalidGoalStepsError,
    RecursiveGoalStopWhenError,
    RecursiveGoalStepError,
} from 'contexts/OperationComponentError';
import { serializeError, type SerializedError } from 'errors/OperationError';
import { WorkflowComponentBase, type MountWorkflow } from './WorkflowComponentBase';
import { asWorkflow, readWorkflowCall, type Workflow, type WorkflowCall } from './workflow';
import { assertTool, readToolCall, type ToolCall, type ToolDispatcher } from './tool';
import type { FrontendToolName } from 'views/lap-analysis/ai-chat/ai-command-registry';
import {
    createControlledOperation,
    createOperationFrom,
    mapOperation,
    type ControlledOperation,
    type Operation,
    type OperationExecutionOutput,
    type OperationStatusPayload,
} from './operation';
import RepeatablePlanOverlayDisplay, { getRepeatablePlanOverlaySummary } from './RepeatablePlanOverlayDisplay';

export type NestedOperationResult = OperationExecutionOutput;
export type NestedOperationStatus = OperationStatusPayload;

export const GOAL_COMPARISON_OPERATORS = [
    'eq',
    'neq',
    'lt',
    'lte',
    'gt',
    'gte',
] as const;

export type GoalComparisonOperator = typeof GOAL_COMPARISON_OPERATORS[number];
export type RepeatablePlanInput = WorkflowCall<'create_repeatable_plan', {
    goal: string;
    tools: ToolCall<{ id: string; title: string; arguments?: Record<string, unknown> }>[];
    stop_when: {
        tool: ToolCall<{ arguments?: Record<string, unknown> }>['tool'];
        operator: GoalComparisonOperator;
        target: number;
    };
}>;
export type GoalStatus = 'running' | 'achieved' | 'missed' | 'error';
export type GoalStepStatus = 'pending' | 'running' | 'completed' | 'error';
export type GoalStopWhenStatus = GoalStepStatus;

export type GoalStepDescriptor = {
    id: string;
    title: string;
    name: string;
    arguments?: Record<string, unknown>;
};

export type GoalStopWhenOperation = {
    name: string;
    arguments?: Record<string, unknown>;
};

export type GoalStopWhen = {
    tool: GoalStopWhenOperation;
    operator: GoalComparisonOperator;
    target: number;
};

export type GoalRequest = {
    name: string;
    steps: GoalStepDescriptor[];
    stop_when: GoalStopWhen;
};

export type GoalTaskDescriptor = {
    title: string;
    name: string;
    arguments?: Record<string, unknown>;
};

export type GoalStepSnapshot = GoalStepDescriptor & {
    status: GoalStepStatus;
    attempts: number;
    run_id: string | null;
    error: string | null;
};

export type GoalSourceResultMetadata = {
    tool_name: string;
    run_id: string;
    status: string;
};

export type GoalStepSourceResultMetadata = GoalSourceResultMetadata & {
    step_id: string;
};

export type GoalTaskResult = {
    step_id: string;
    tool_name: string;
    attempt: number;
    status: 'completed' | 'error';
    source_result?: GoalStepSourceResultMetadata;
    error?: SerializedError;
};

export type GoalStopWhenResult = {
    tool_name: string;
    attempt: number;
    status: GoalStopWhenStatus;
    value: number | null;
    error?: string;
    source_result?: GoalSourceResultMetadata;
};

export type GoalSnapshot = {
    name: string;
    status: GoalStatus;
    steps: GoalStepSnapshot[];
    stop_when: GoalStopWhen | null;
    stop_when_result: GoalStopWhenResult | null;
    target: number | null;
    actual: number | null;
    completed_steps: string[];
    failed_step?: string;
    error?: string;
};

export type GoalRunResult = Pick<
    GoalSnapshot,
    | 'name'
    | 'target'
    | 'actual'
    | 'completed_steps'
    | 'stop_when'
    | 'stop_when_result'
> & {
    status: 'achieved' | 'missed' | 'failed';
    task_results: GoalTaskResult[];
    failed_step?: string;
    error?: string;
};

export type GoalAiResult = Omit<GoalRunResult, 'name'> & { goal: string };

export interface RepeatablePlanHandle extends NamedOperationComponentHandle, AiOverlayComponentHandle<GoalSnapshot | null> {
    createRepeatablePlan(input: RepeatablePlanInput): Workflow<GoalAiResult>;
    retryFailedTask(): Workflow<GoalAiResult>;
    getSnapshot(): GoalSnapshot | null;
    clear(): void;
}

export type RepeatablePlanDisplayProps = {
    snapshot: GoalSnapshot;
    surface?: 'chat' | 'pill';
};

export type RepeatablePlanProps = {
    snapshot: GoalSnapshot | null;
    surface?: 'chat' | 'pill';
};

const RETRY_DELAY_MS = 1000;
const RECURSIVE_GOAL_OPERATION_NAMES = new Set([
    'create_repeatable_plan',
    'retry_repeatable_plan_task',
]);
const createRepeatablePlanRunId = (): string => (
    `goal-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`
);

const isRecord = (value: unknown): value is Record<string, unknown> => (
    Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

const hasOnlyKeys = (value: Record<string, unknown>, allowedKeys: readonly string[]): boolean => {
    const allowed = new Set(allowedKeys);
    return Reflect.ownKeys(value).every((key) => typeof key === 'string' && allowed.has(key));
};

const toNonEmptyString = (value: unknown): string | null => {
    if (typeof value !== 'string') return null;
    const normalized = value.trim();
    return normalized || null;
};

const isGoalComparisonOperator = (value: unknown): value is GoalComparisonOperator => (
    typeof value === 'string'
    && (GOAL_COMPARISON_OPERATORS as readonly string[]).includes(value)
);

const parseGoalStepDescriptor = (value: unknown): GoalStepDescriptor | null => {
    const step = readToolCall(value);
    if (!step || !hasOnlyKeys(step, ['name', 'id', 'title', 'arguments'])) return null;
    const name = step.name as string;
    const id = toNonEmptyString(step.id);
    const title = toNonEmptyString(step.title);
    if (!id || !title || !name) return null;
    if (step.arguments !== undefined && !isRecord(step.arguments)) return null;
    return {
        id,
        title,
        name,
        ...(step.arguments !== undefined ? { arguments: { ...step.arguments } } : {}),
    };
};

const parseGoalStopWhenOperation = (value: unknown): GoalStopWhenOperation | null => {
    const tool = readToolCall({ tool: value });
    if (!tool || !hasOnlyKeys(tool, ['name', 'arguments'])) return null;
    const name = tool.name as string;
    if (!name || (tool.arguments !== undefined && !isRecord(tool.arguments))) return null;
    return {
        name,
        ...(tool.arguments !== undefined ? { arguments: { ...tool.arguments } } : {}),
    };
};

const parseGoalStopWhen = (value: unknown): GoalStopWhen | null => {
    const stopWhen = isRecord(value) ? value : null;
    if (!stopWhen || !hasOnlyKeys(
        stopWhen,
        ['tool', 'operator', 'target'],
    )) return null;
    const tool = parseGoalStopWhenOperation(stopWhen.tool);
    if (
        !tool
        || !isGoalComparisonOperator(stopWhen.operator)
        || typeof stopWhen.target !== 'number'
        || !Number.isFinite(stopWhen.target)
    ) {
        return null;
    }
    return {
        tool,
        operator: stopWhen.operator,
        target: stopWhen.target,
    };
};

export const validateGoalRequest = (
    value: unknown,
    componentName = 'repeatable-plan',
): { request: GoalRequest } | { error: GoalComponentError; name?: string } => {
    const input = readWorkflowCall(value, 'create_repeatable_plan');
    const name = toNonEmptyString(input?.goal);
    if (!input || !name || !hasOnlyKeys(input, ['name', 'goal', 'tools', 'stop_when'])) {
        return {
            error: new InvalidGoalNameError(componentName, 'Provide a valid repeatable plan name.'),
            ...(name ? { name } : {}),
        };
    }
    if (!Array.isArray(input.tools) || input.tools.length === 0) {
        return {
            error: new InvalidGoalStepsError(componentName, 'Provide at least one valid repeatable plan step.'),
            name,
        };
    }
    const steps = input.tools.map(parseGoalStepDescriptor);
    if (steps.some((step) => !step)) {
        return {
            error: new InvalidGoalStepsError(componentName, 'Every repeatable plan step must have a valid id, title, name, and arguments object.'),
            name,
        };
    }
    const parsedSteps = steps as GoalStepDescriptor[];
    const ids = new Set<string>();
    for (const step of parsedSteps) {
        if (ids.has(step.id)) {
            return {
                error: new DuplicateGoalStepIdError(componentName, `Repeatable plan step id '${step.id}' is duplicated.`),
                name,
            };
        }
        ids.add(step.id);
        if (RECURSIVE_GOAL_OPERATION_NAMES.has(step.name)) {
            return {
                error: new RecursiveGoalStepError(componentName, 'Repeatable plan steps cannot invoke repeatable-plan management workflows.'),
                name,
            };
        }
    }
    const stopWhen = parseGoalStopWhen(input.stop_when);
    if (!stopWhen) {
        return {
            error: new InvalidGoalStopWhenError(componentName, 'Provide a valid repeatable plan stop condition.'),
            name,
        };
    }
    if (RECURSIVE_GOAL_OPERATION_NAMES.has(stopWhen.tool.name)) {
        return {
            error: new RecursiveGoalStopWhenError(componentName, 'Repeatable plan stop condition cannot invoke a repeatable-plan management workflow.'),
            name,
        };
    }
    return { request: { name, steps: parsedSteps, stop_when: stopWhen } };
};

export const parseRepeatablePlanInput = (
    value: unknown,
    componentName = 'repeatable-plan',
): GoalRequest => {
    const validation = validateGoalRequest(value, componentName);
    if ('error' in validation) throw validation.error;
    return validation.request;
};

const evaluateGoalStopWhenInput = (value: unknown): number | null => {
    if (
        !isRecord(value)
        || value.status !== 'ready'
        || typeof value.data !== 'number'
        || !Number.isFinite(value.data)
    ) {
        return null;
    }
    return value.data;
};

export const compareGoalValues = (
    actual: number,
    operator: GoalComparisonOperator,
    target: number,
): boolean => {
    switch (operator) {
        case 'eq': return actual === target;
        case 'neq': return actual !== target;
        case 'lt': return actual < target;
        case 'lte': return actual <= target;
        case 'gt': return actual > target;
        case 'gte': return actual >= target;
        default: return false;
    }
};

const cloneStopWhenOperation = (tool: GoalStopWhenOperation): GoalStopWhenOperation => ({
    ...tool,
    ...(tool.arguments ? { arguments: { ...tool.arguments } } : {}),
});

const cloneStopWhen = (stopWhen: GoalStopWhen): GoalStopWhen => ({
    ...stopWhen,
    tool: cloneStopWhenOperation(stopWhen.tool),
});

const cloneStopWhenResult = (
    result: GoalStopWhenResult | null,
): GoalStopWhenResult | null => result ? ({
    ...result,
    ...(result.source_result ? { source_result: { ...result.source_result } } : {}),
}) : null;

const cloneSnapshot = (snapshot: GoalSnapshot): GoalSnapshot => ({
    ...snapshot,
    steps: snapshot.steps.map((step) => ({
        ...step,
        ...(step.arguments ? { arguments: { ...step.arguments } } : {}),
    })),
    stop_when: snapshot.stop_when
        ? cloneStopWhen(snapshot.stop_when)
        : null,
    stop_when_result: cloneStopWhenResult(snapshot.stop_when_result),
    completed_steps: [...snapshot.completed_steps],
});

const cloneTaskResults = (taskResults: GoalTaskResult[]): GoalTaskResult[] => (
    taskResults.map((result) => ({
        ...result,
        ...(result.source_result ? { source_result: { ...result.source_result } } : {}),
    }))
);

const toRunResult = (
    snapshot: GoalSnapshot & { status: GoalRunResult['status'] },
    taskResults: GoalTaskResult[],
): GoalRunResult => ({
    name: snapshot.name,
    status: snapshot.status,
    stop_when: snapshot.stop_when
        ? cloneStopWhen(snapshot.stop_when)
        : null,
    stop_when_result: cloneStopWhenResult(snapshot.stop_when_result),
    target: snapshot.target,
    actual: snapshot.actual,
    completed_steps: [...snapshot.completed_steps],
    task_results: cloneTaskResults(taskResults),
});

type RuntimeTaskExecutionResult = {
    value: unknown;
    error?: GoalComponentError;
    source_result: GoalSourceResultMetadata;
};

type ActiveGoalOperation = {
    controller: ControlledOperation<
        GoalRunResult,
        never,
        'complete' | 'failed' | 'cancelled' | 'replaced'
    >;
    nestedOperation: Operation<NestedOperationResult, NestedOperationStatus> | null;
};

const toGoalAiResult = (result: GoalRunResult): GoalAiResult => {
    const { name, ...safeResult } = result;
    return { ...safeResult, goal: name };
};

export class RepeatablePlanRunner
extends WorkflowComponentBase<GoalSnapshot | null>
implements RepeatablePlanHandle {
    private currentSnapshot: GoalSnapshot | null = null;
    private request: GoalRequest | null = null;
    private failedStepIndex: number | null = null;
    private stopWhenFailed = false;
    private stepAttempts: number[] = [];
    private stopWhenAttempts = 0;
    private taskResults: GoalTaskResult[] = [];
    private generation = 0;
    private activeOperation: ActiveGoalOperation | null = null;

    constructor(
        componentName: string,
        private readonly dispatchOperation: ToolDispatcher,
        private readonly onChange?: (snapshot: GoalSnapshot | null) => void,
    ) {
        super(componentName, null);
    }

    createRepeatablePlan(input: RepeatablePlanInput): Workflow<GoalAiResult> {
        return asWorkflow(mapOperation(this.create(input), toGoalAiResult));
    }

    getComponentType(): string {
        return 'repeatable-plan';
    }

    getOverlayBehavior(snapshot: GoalSnapshot | null) {
        return {
            placement: 'flow' as const,
            requestedStatus: 'expanded' as const,
            remove: snapshot === null || snapshot.status === 'achieved',
        };
    }

    getOverlayMetadata() {
        return {};
    }

    handleOverlayRendererEvent(_event: AiOverlayRendererEvent): void {
        // Repeatable plan overlays have no renderer-originated events.
    }

    getSnapshot(): GoalSnapshot | null {
        return this.currentSnapshot ? cloneSnapshot(this.currentSnapshot) : null;
    }

    create(input: RepeatablePlanInput): Workflow<GoalRunResult> {
        try {
            const request = parseRepeatablePlanInput(input, this.getComponentName());
            request.steps.forEach((step) => this.dispatchOperation.validate(step.name));
            this.dispatchOperation.validate(request.stop_when.tool.name);
            return this.startOperation(() => this.runCreate(request));
        } catch (error) {
            return asWorkflow(createOperationFrom(() => { throw error; }, 'failed'));
        }
    }

    private async runCreate(input: GoalRequest): Promise<GoalRunResult> {
        this.generation += 1;
        this.request = input;
        this.failedStepIndex = null;
        this.stopWhenFailed = false;
        this.stepAttempts = input.steps.map(() => 0);
        this.stopWhenAttempts = 0;
        this.taskResults = [];
        this.publish(this.createRunningSnapshot(input));
        return this.runPreparation(input, this.generation, 0);
    }

    retryFailedTask(): Workflow<GoalAiResult> {
        return asWorkflow(mapOperation(this.retryFailedTaskResult(), toGoalAiResult));
    }

    private retryFailedTaskResult(): Workflow<GoalRunResult> {
        const request = this.request;
        if (!request) return asWorkflow(createOperationFrom(() => this.runRetryFailedTask(), 'failed'));
        return this.startOperation(() => this.runRetryFailedTask());
    }

    private async runRetryFailedTask(): Promise<GoalRunResult> {
        const request = this.request;
        if (!request || !this.currentSnapshot || this.currentSnapshot.status !== 'error') {
            throw new GoalTaskRetryUnavailableError(
                this.getComponentName(),
                'The failed repeatable plan task could not be retried.',
            );
        }
        const generation = ++this.generation;
        if (this.failedStepIndex !== null) {
            const index = this.failedStepIndex;
            this.failedStepIndex = null;
            const { failed_step: _failedStep, error: _error, ...snapshot } = this.currentSnapshot;
            this.publish({ ...snapshot, status: 'running', actual: null });
            return this.runPreparation(request, generation, index);
        }
        if (this.stopWhenFailed) {
            this.stopWhenFailed = false;
            const { failed_step: _failedStep, error: _error, ...snapshot } = this.currentSnapshot;
            this.publish({
                ...snapshot,
                status: 'running',
                actual: null,
                stop_when_result: this.pendingStopWhenResult(request),
            });
            return this.runStopWhen(request, generation);
        }
        throw new GoalTaskRetryUnavailableError(
            this.getComponentName(),
            'The failed repeatable plan task could not be retried.',
        );
    }

    clear(): void {
        this.cancelActiveOperation('cancelled', new GoalReplacedError(
            this.getComponentName(),
            'The repeatable plan run was cleared.',
        ));
        this.generation += 1;
        this.currentSnapshot = null;
        this.request = null;
        this.failedStepIndex = null;
        this.stopWhenFailed = false;
        this.onChange?.(null);
        this.publishSnapshot(null);
    }

    protected onDispose(): void {
        this.cancelActiveOperation('cancelled', new GoalReplacedError(
            this.getComponentName(),
            'The repeatable plan run was disposed.',
        ));
        this.generation += 1;
        this.currentSnapshot = null;
        this.request = null;
    }

    private startOperation(
        run: () => Promise<GoalRunResult>,
    ): Workflow<GoalRunResult> {
        this.cancelActiveOperation('replaced', new GoalReplacedError(
            this.getComponentName(),
            'The repeatable plan run was replaced.',
        ));
        let operation!: ActiveGoalOperation;
        const controller = createControlledOperation<
            GoalRunResult,
            never,
            'complete' | 'failed' | 'cancelled' | 'replaced'
        >([], () => this.abortOperation(operation));
        operation = {
            controller,
            nestedOperation: null,
        };
        this.activeOperation = operation;
        void run().then(
            (result) => operation.controller.resolve('complete', result),
            (error) => operation.controller.reject(
                'failed',
                error instanceof Error ? error : new Error(String(error)),
            ),
        ).finally(() => {
            if (this.activeOperation === operation) this.activeOperation = null;
        });
        return asWorkflow(operation.controller.operation);
    }

    private cancelActiveOperation(
        status: 'cancelled' | 'replaced',
        error: Error,
    ): void {
        const operation = this.activeOperation;
        if (!operation) return;
        this.activeOperation = null;
        operation.nestedOperation?.abort();
        operation.nestedOperation = null;
        operation.controller.reject(status, error);
    }

    private abortOperation(operation: ActiveGoalOperation): void {
        operation.nestedOperation?.abort();
        operation.nestedOperation = null;
        if (this.activeOperation !== operation) return;
        this.activeOperation = null;
        this.clear();
        this.deleteComponentRef();
    }

    private async runPreparation(
        request: GoalRequest,
        generation: number,
        startIndex: number,
    ): Promise<GoalRunResult> {
        for (let index = startIndex; index < request.steps.length; index += 1) {
            if (generation !== this.generation) {
                throw new GoalReplacedError(this.getComponentName(), 'The repeatable plan run was cancelled.');
            }
            const step = request.steps[index];
            const attempt = (this.stepAttempts[index] ?? 0) + 1;
            const runId = createRepeatablePlanRunId();
            this.stepAttempts[index] = attempt;
            this.updateStep(index, {
                status: 'running',
                attempts: attempt,
                run_id: runId,
                error: null,
            });
            const execution = await this.executeTask(
                step.name,
                step.arguments,
                GoalStepFailedError,
                'The repeatable plan step failed.',
                runId,
            );
            if (generation !== this.generation) {
                throw new GoalReplacedError(this.getComponentName(), 'The repeatable plan run was cancelled.');
            }
            const sourceResult = { ...execution.source_result, step_id: step.id };
            this.taskResults.push({
                step_id: step.id,
                tool_name: step.name,
                attempt,
                status: execution.error ? 'error' : 'completed',
                source_result: sourceResult,
                ...(execution.error ? { error: serializeError(execution.error) } : {}),
            });
            if (execution.error) {
                this.failedStepIndex = index;
                this.stopWhenFailed = false;
                this.updateStep(index, {
                    status: 'error',
                    run_id: sourceResult.run_id,
                    error: execution.error.message,
                });
                const snapshot: GoalSnapshot = {
                    ...this.currentSnapshot!,
                    status: 'error',
                    actual: null,
                    failed_step: step.id,
                    error: execution.error.message,
                };
                this.publish(snapshot);
                return this.failedRunResult(snapshot);
            }
            this.updateStep(index, {
                status: 'completed',
                run_id: sourceResult.run_id,
                error: null,
            });
            this.publish({
                ...this.currentSnapshot!,
                completed_steps: this.currentSnapshot!.steps
                    .filter((item) => item.status === 'completed')
                    .map((item) => item.id),
            });
        }
        return this.runStopWhen(request, generation);
    }

    private async runStopWhen(
        request: GoalRequest,
        generation: number,
    ): Promise<GoalRunResult> {
        const attempt = ++this.stopWhenAttempts;
        this.publish({
            ...this.currentSnapshot!,
            stop_when_result: {
                tool_name: request.stop_when.tool.name,
                attempt,
                status: 'running',
                value: null,
            },
        });
        const execution = await this.executeTask(
            request.stop_when.tool.name,
            request.stop_when.tool.arguments,
            GoalStopWhenFailedError,
            'The repeatable plan stop condition check failed.',
        );
        if (generation !== this.generation) {
            throw new GoalReplacedError(this.getComponentName(), 'The repeatable plan run was cancelled.');
        }
        let error = execution.error;
        let actual: number | null = null;
        if (!error) {
            actual = evaluateGoalStopWhenInput(execution.value);
            if (actual === null) {
                error = new GoalStopWhenInputIncompatibleError(
                    this.getComponentName(),
                    'Repeatable plan stop condition requires a ready query result with finite numeric data.',
                );
            }
        }
        if (error) {
            this.failedStepIndex = null;
            this.stopWhenFailed = true;
            const snapshot: GoalSnapshot = {
                ...this.currentSnapshot!,
                status: 'error',
                actual: null,
                error: error.message,
                stop_when_result: {
                    tool_name: request.stop_when.tool.name,
                    attempt,
                    status: 'error',
                    value: null,
                    error: error.message,
                    ...(execution.source_result
                        ? { source_result: { ...execution.source_result } }
                        : {}),
                },
            };
            this.publish(snapshot);
            return this.failedRunResult(snapshot);
        }

        this.failedStepIndex = null;
        this.stopWhenFailed = false;
        const achieved = compareGoalValues(
            actual!,
            request.stop_when.operator,
            request.stop_when.target,
        );
        const snapshot: GoalSnapshot & { status: 'achieved' | 'missed' } = {
            ...this.currentSnapshot!,
            status: achieved ? 'achieved' : 'missed',
            actual,
            stop_when_result: {
                tool_name: request.stop_when.tool.name,
                attempt,
                status: 'completed',
                value: actual,
                ...(execution.source_result
                    ? { source_result: { ...execution.source_result } }
                    : {}),
                },
            };
        this.publish(snapshot);
        if (!achieved) {
            await this.retryDelay();
            if (generation !== this.generation) {
                throw new GoalReplacedError(this.getComponentName(), 'The repeatable plan run was cancelled.');
            }
            this.publish(this.createRunningSnapshot(request));
            return this.runPreparation(request, generation, 0);
        }
        this.finish();
        return toRunResult(snapshot, this.taskResults);
    }

    private async executeTask(
        toolName: string,
        argumentsValue: Record<string, unknown> | undefined,
        FailureError: OperationComponentErrorConstructor<GoalComponentError>,
        fallbackMessage: string,
        runId = createRepeatablePlanRunId(),
    ): Promise<RuntimeTaskExecutionResult> {
        const activeOperation = this.activeOperation;
        let operation: Operation<NestedOperationResult, NestedOperationStatus> | null = null;
        try {
            const dispatchedOperation = this.dispatchOperation(
                toolName as FrontendToolName,
                argumentsValue ?? {},
            );
            assertTool(dispatchedOperation);
            operation = dispatchedOperation;
            if (
                !activeOperation
                || this.activeOperation !== activeOperation
                || activeOperation.controller.signal.aborted
            ) {
                dispatchedOperation.abort();
            } else {
                activeOperation.nestedOperation = dispatchedOperation;
            }
            const termination = await new Promise<{
                status: string;
                result: NestedOperationResult | Error;
            }>((resolve) => dispatchedOperation.notifyTerminated(resolve));
            if (termination.result instanceof Error) throw termination.result;
            return {
                value: termination.result,
                source_result: {
                    tool_name: toolName,
                    run_id: runId,
                    status: termination.status,
                },
            };
        } catch (error) {
            return {
                value: null,
                source_result: {
                    tool_name: toolName,
                    run_id: runId,
                    status: 'failed',
                },
                error: new FailureError(
                    this.getComponentName(),
                    error instanceof Error && error.message ? error.message : fallbackMessage,
                    { cause: error },
                ),
            };
        } finally {
            if (activeOperation?.nestedOperation === operation) {
                activeOperation.nestedOperation = null;
            }
        }
    }

    private createRunningSnapshot(request: GoalRequest): GoalSnapshot {
        return {
            name: request.name,
            status: 'running',
            steps: request.steps.map((step, index) => ({
                ...step,
                ...(step.arguments ? { arguments: { ...step.arguments } } : {}),
                status: 'pending',
                attempts: this.stepAttempts[index] ?? 0,
                run_id: null,
                error: null,
            })),
            stop_when: cloneStopWhen(request.stop_when),
            stop_when_result: this.pendingStopWhenResult(request),
            target: request.stop_when.target,
            actual: null,
            completed_steps: [],
        };
    }

    private pendingStopWhenResult(request: GoalRequest): GoalStopWhenResult {
        return {
            tool_name: request.stop_when.tool.name,
            attempt: this.stopWhenAttempts,
            status: 'pending',
            value: null,
        };
    }

    private failedRunResult(snapshot: GoalSnapshot): GoalRunResult {
        return {
            name: snapshot.name,
            status: 'failed',
            stop_when: snapshot.stop_when
                ? cloneStopWhen(snapshot.stop_when)
                : null,
            stop_when_result: cloneStopWhenResult(snapshot.stop_when_result),
            target: snapshot.target,
            actual: snapshot.actual,
            completed_steps: [...snapshot.completed_steps],
            task_results: cloneTaskResults(this.taskResults),
            ...(snapshot.failed_step ? { failed_step: snapshot.failed_step } : {}),
            ...(snapshot.error ? { error: snapshot.error } : {}),
        };
    }

    private publish(snapshot: GoalSnapshot): void {
        this.currentSnapshot = cloneSnapshot(snapshot);
        this.publishSnapshot(this.getSnapshot());
        this.onChange?.(this.getSnapshot());
    }

    private updateStep(index: number, update: Partial<GoalStepSnapshot>): void {
        if (!this.currentSnapshot) return;
        this.publish({
            ...this.currentSnapshot,
            steps: this.currentSnapshot.steps.map((step, stepIndex) => (
                stepIndex === index ? { ...step, ...update } : step
            )),
        });
    }

    private finish(): void {
        // The completed repeatable plan stays mounted until AI Chat replaces it.
    }

    private retryDelay(): Promise<void> {
        return new Promise((resolve) => setTimeout(resolve, RETRY_DELAY_MS));
    }
}

export const useRepeatablePlanWorkflow = ({
    mountWorkflow,
}: {
    mountWorkflow: MountWorkflow;
}) => {
    const runnerRef = useRef<RepeatablePlanRunner | null>(null);
    const [snapshot, setSnapshot] = useState<GoalSnapshot | null>(null);

    const dispose = useCallback(() => {
        const runner = runnerRef.current;
        runnerRef.current = null;
        runner?.dispose();
    }, []);

    useEffect(() => dispose, [dispose]);

    const createRepeatablePlan = useCallback((
        input: RepeatablePlanInput,
        dispatcher: ToolDispatcher,
    ): Workflow<GoalAiResult> => {
        try {
            const request = parseRepeatablePlanInput(input);
            request.steps.forEach((step) => dispatcher.validate(step.name));
            dispatcher.validate(request.stop_when.tool.name);
            const runner = new RepeatablePlanRunner(
                OPERATION_COMPONENT_NAMES.REPEATABLE_PLAN,
                dispatcher,
                (next) => {
                    if (runnerRef.current !== runner) return;
                    setSnapshot(next);
                },
            );
            try {
                mountWorkflow({ runner, dispose });
                runnerRef.current = runner;
                setSnapshot(null);
                return runner.createRepeatablePlan(input);
            } catch (error) {
                if (runnerRef.current === runner) runnerRef.current = null;
                runner.dispose();
                throw error;
            }
        } catch (error) {
            return asWorkflow(createOperationFrom(() => { throw error; }, 'failed'));
        }
    }, [dispose, mountWorkflow]);

    const reset = useCallback(() => {
        dispose();
        setSnapshot(null);
    }, [dispose]);

    return { createRepeatablePlan, snapshot, reset };
};

const getComparisonText = (snapshot: GoalSnapshot): string => {
    const stopWhen = snapshot.stop_when;
    if (!stopWhen) return snapshot.error || 'Invalid repeatable plan';
    const actual = snapshot.actual === null ? '—' : String(snapshot.actual);
    return `${actual} ${stopWhen.operator} ${stopWhen.target}`;
};

export const RepeatablePlanDisplay: React.FC<RepeatablePlanDisplayProps> = ({ snapshot, surface = 'chat' }) => {
    if (surface === 'chat' && snapshot.status === 'achieved') return null;

    const stopWhenResult = snapshot.stop_when_result;
    return (
        <section
            className={`ai-chat__goal ai-chat__goal--${surface} ai-chat__goal--${snapshot.status}`}
            aria-label="Repeatable plan"
            aria-live="polite"
        >
            <div className="ai-chat__goal-head">
                <div>
                    <span className="ai-chat__goal-kicker">REPEATABLE PLAN · {snapshot.status}</span>
                    <div className="ai-chat__goal-title">{snapshot.name}</div>
                </div>
            </div>
            {snapshot.steps.length > 0 && (
                <ol className="ai-chat__goal-steps">
                    {snapshot.steps.map((step) => (
                        <li
                            key={step.id}
                            className={`ai-chat__goal-step ai-chat__goal-step--${step.status}`}
                        >
                            <span className="ai-chat__goal-step-dot" aria-hidden="true" />
                            <span className="ai-chat__goal-step-copy">
                                <span>{step.title}</span>
                                <span>{step.status}{step.attempts > 1 ? ` · attempt ${step.attempts}` : ''}</span>
                            </span>
                        </li>
                    ))}
                </ol>
            )}
            {snapshot.stop_when && stopWhenResult && (
                <div
                    className={`ai-chat__goal-stop-when ai-chat__goal-stop-when--${stopWhenResult.status}`}
                    aria-label="Stop when"
                >
                    <span className="ai-chat__goal-step-dot" aria-hidden="true" />
                    <span className="ai-chat__goal-step-copy">
                        <span>Stop when · {stopWhenResult.tool_name}</span>
                        <span>
                            {stopWhenResult.status}
                            {stopWhenResult.attempt > 0
                                ? ` · attempt ${stopWhenResult.attempt}`
                                : ''}
                        </span>
                        <span className="ai-chat__goal-metric">{getComparisonText(snapshot)}</span>
                    </span>
                </div>
            )}
            {snapshot.error && snapshot.steps.length === 0 && (
                <div className="ai-chat__goal-error">{snapshot.error}</div>
            )}
        </section>
    );
};

export const repeatablePlanOverlayRenderer: AiOverlayRenderer<GoalSnapshot> = {
    componentType: 'repeatable-plan',
    validateSnapshot: (snapshot): snapshot is GoalSnapshot => (
        isOverlayRecord(snapshot)
        && isOverlayNonEmptyString(snapshot.name)
        && ['running', 'achieved', 'missed', 'error'].includes(String(snapshot.status))
        && Array.isArray(snapshot.steps)
        && (snapshot.target === null || (typeof snapshot.target === 'number' && Number.isFinite(snapshot.target)))
        && (snapshot.actual === null || (typeof snapshot.actual === 'number' && Number.isFinite(snapshot.actual)))
    ),
    renderOverlay: (snapshot, status) => status === 'folded'
        ? getRepeatablePlanOverlaySummary(snapshot)
        : <RepeatablePlanOverlayDisplay snapshot={snapshot} />,
    dimensions: {
        expanded: { width: 420, height: 176 },
        folded: { width: 340, height: 58 },
    },
};

const RepeatablePlan: React.FC<RepeatablePlanProps> = ({ snapshot, surface = 'chat' }) => (
    snapshot ? <RepeatablePlanDisplay snapshot={snapshot} surface={surface} /> : null
);

export default RepeatablePlan;
