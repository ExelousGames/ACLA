import React from 'react';
import type { NamedAiToolComponentHandle } from 'contexts/AiToolComponentRefContext';
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
    AiToolComponentErrorConstructor,
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
} from 'contexts/AiToolComponentError';
import { serializeError, type SerializedError } from 'errors/AiToolError';
import { AiToolComponentBase } from './AiToolComponentBase';
import {
    createControlledAiToolOperation,
    createAiToolOperationFrom,
    mapAiToolOperation,
    type ControlledAiToolOperation,
    type AiToolOperation,
} from './ai-tool-operation';
import GoalOverlayDisplay, { getGoalOverlaySummary } from './GoalOverlayDisplay';

export type NestedAiToolResult = Record<string, unknown> | string;
export interface NestedAiToolStatus {
    [key: string]: unknown;
}

export type AiToolDispatcher = (
    name: string,
    args?: Record<string, unknown>,
    signal?: AbortSignal,
) => AiToolOperation<NestedAiToolResult, NestedAiToolStatus>;

export const GOAL_COMPARISON_OPERATORS = [
    'eq',
    'neq',
    'lt',
    'lte',
    'gt',
    'gte',
] as const;

export type GoalComparisonOperator = typeof GOAL_COMPARISON_OPERATORS[number];
export type GoalStatus = 'running' | 'achieved' | 'missed' | 'error';
export type GoalStepStatus = 'pending' | 'running' | 'completed' | 'error';
export type GoalStopWhenStatus = GoalStepStatus;

export type GoalStepDescriptor = {
    id: string;
    title: string;
    name: string;
    arguments?: Record<string, unknown>;
};

export type GoalStopWhenTool = {
    name: string;
    arguments?: Record<string, unknown>;
};

export type GoalStopWhen = {
    tool: GoalStopWhenTool;
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

export interface GoalHandle extends NamedAiToolComponentHandle, AiOverlayComponentHandle<GoalSnapshot | null> {
    createGoal(input: GoalRequest): AiToolOperation<GoalAiResult>;
    retryFailedTask(): AiToolOperation<GoalAiResult>;
    getSnapshot(): GoalSnapshot | null;
    clear(): void;
}

export type GoalDisplayProps = {
    snapshot: GoalSnapshot;
    surface?: 'chat' | 'pill';
};

export type GoalProps = {
    snapshot: GoalSnapshot | null;
    surface?: 'chat' | 'pill';
};

const RETRY_DELAY_MS = 1000;
const RECURSIVE_GOAL_TOOL_NAMES = new Set(['create_goal', 'retry_goal_task']);
const createGoalRunId = (): string => (
    `goal-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`
);

const isRecord = (value: unknown): value is Record<string, unknown> => (
    Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

const hasOnlyKeys = (value: Record<string, unknown>, allowedKeys: readonly string[]): boolean => {
    const allowed = new Set(allowedKeys);
    return Object.keys(value).every((key) => allowed.has(key));
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
    const step = isRecord(value) ? value : null;
    if (!step || !hasOnlyKeys(step, ['id', 'title', 'name', 'arguments'])) return null;
    const id = toNonEmptyString(step.id);
    const title = toNonEmptyString(step.title);
    const name = toNonEmptyString(step.name);
    if (!id || !title || !name) return null;
    if (step.arguments !== undefined && !isRecord(step.arguments)) return null;
    return {
        id,
        title,
        name,
        ...(step.arguments !== undefined ? { arguments: { ...step.arguments } } : {}),
    };
};

const parseGoalStopWhenTool = (value: unknown): GoalStopWhenTool | null => {
    const tool = isRecord(value) ? value : null;
    if (!tool || !hasOnlyKeys(tool, ['name', 'arguments'])) return null;
    const name = toNonEmptyString(tool.name);
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
    const tool = parseGoalStopWhenTool(stopWhen.tool);
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
    componentName = 'goal',
): { request: GoalRequest } | { error: GoalComponentError; name?: string } => {
    const input = isRecord(value) ? value : null;
    const name = toNonEmptyString(input?.name);
    if (!input || !name || !hasOnlyKeys(input, ['name', 'steps', 'stop_when'])) {
        return {
            error: new InvalidGoalNameError(componentName, 'Provide a valid goal name.'),
            ...(name ? { name } : {}),
        };
    }
    if (!Array.isArray(input.steps) || input.steps.length === 0) {
        return {
            error: new InvalidGoalStepsError(componentName, 'Provide at least one valid goal step.'),
            name,
        };
    }
    const steps = input.steps.map(parseGoalStepDescriptor);
    if (steps.some((step) => !step)) {
        return {
            error: new InvalidGoalStepsError(componentName, 'Every goal step must have a valid id, title, name, and arguments object.'),
            name,
        };
    }
    const parsedSteps = steps as GoalStepDescriptor[];
    const ids = new Set<string>();
    for (const step of parsedSteps) {
        if (ids.has(step.id)) {
            return {
                error: new DuplicateGoalStepIdError(componentName, `Goal step id '${step.id}' is duplicated.`),
                name,
            };
        }
        ids.add(step.id);
        if (RECURSIVE_GOAL_TOOL_NAMES.has(step.name)) {
            return {
                error: new RecursiveGoalStepError(componentName, 'Goal steps cannot invoke goal-management tools.'),
                name,
            };
        }
    }
    const stopWhen = parseGoalStopWhen(input.stop_when);
    if (!stopWhen) {
        return {
            error: new InvalidGoalStopWhenError(componentName, 'Provide a valid goal stop condition.'),
            name,
        };
    }
    if (RECURSIVE_GOAL_TOOL_NAMES.has(stopWhen.tool.name)) {
        return {
            error: new RecursiveGoalStopWhenError(componentName, 'Goal stop condition cannot invoke a goal-management tool.'),
            name,
        };
    }
    return { request: { name, steps: parsedSteps, stop_when: stopWhen } };
};

export const buildGoalRequest = (
    value: unknown,
    componentName = 'goal',
): { request: GoalRequest } | { error: GoalComponentError; name?: string } => (
    validateGoalRequest(value, componentName)
);

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

const cloneStopWhenTool = (tool: GoalStopWhenTool): GoalStopWhenTool => ({
    ...tool,
    ...(tool.arguments ? { arguments: { ...tool.arguments } } : {}),
});

const cloneStopWhen = (stopWhen: GoalStopWhen): GoalStopWhen => ({
    ...stopWhen,
    tool: cloneStopWhenTool(stopWhen.tool),
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
    controller: ControlledAiToolOperation<
        GoalRunResult,
        never,
        'complete' | 'failed' | 'cancelled' | 'replaced'
    >;
    nestedOperation: AiToolOperation<NestedAiToolResult, NestedAiToolStatus> | null;
};

const toGoalAiResult = (result: GoalRunResult): GoalAiResult => {
    const { name, ...safeResult } = result;
    return { ...safeResult, goal: name };
};

export class GoalRunner
extends AiToolComponentBase<GoalSnapshot | null>
implements GoalHandle {
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
        private readonly dispatchTool: AiToolDispatcher,
        private readonly onChange?: (snapshot: GoalSnapshot | null) => void,
    ) {
        super(componentName, null);
    }

    createGoal(input: GoalRequest): AiToolOperation<GoalAiResult> {
        return mapAiToolOperation(this.create(input), toGoalAiResult);
    }

    getComponentType(): string {
        return 'goal';
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
        // Goal overlays have no renderer-originated events.
    }

    getSnapshot(): GoalSnapshot | null {
        return this.currentSnapshot ? cloneSnapshot(this.currentSnapshot) : null;
    }

    create(input: GoalRequest): AiToolOperation<GoalRunResult> {
        const validation = validateGoalRequest(input, this.getComponentName());
        if ('error' in validation) {
            return createAiToolOperationFrom(() => this.runCreate(input), 'failed');
        }
        return this.startOperation(() => this.runCreate(input));
    }

    private async runCreate(input: GoalRequest): Promise<GoalRunResult> {
        const validation = validateGoalRequest(input, this.getComponentName());
        if ('error' in validation) {
            const snapshot: GoalSnapshot = {
                name: validation.name || 'Goal',
                status: 'error',
                steps: [],
                stop_when: null,
                stop_when_result: null,
                target: null,
                actual: null,
                completed_steps: [],
                error: validation.error.message,
            };
            this.publish(snapshot);
            throw validation.error;
        }

        this.generation += 1;
        this.request = validation.request;
        this.failedStepIndex = null;
        this.stopWhenFailed = false;
        this.stepAttempts = validation.request.steps.map(() => 0);
        this.stopWhenAttempts = 0;
        this.taskResults = [];
        this.publish(this.createRunningSnapshot(validation.request));
        return this.runPreparation(validation.request, this.generation, 0);
    }

    retryFailedTask(): AiToolOperation<GoalAiResult> {
        return mapAiToolOperation(this.retryFailedTaskResult(), toGoalAiResult);
    }

    private retryFailedTaskResult(): AiToolOperation<GoalRunResult> {
        const request = this.request;
        if (!request) return createAiToolOperationFrom(() => this.runRetryFailedTask(), 'failed');
        return this.startOperation(() => this.runRetryFailedTask());
    }

    private async runRetryFailedTask(): Promise<GoalRunResult> {
        const request = this.request;
        if (!request || !this.currentSnapshot || this.currentSnapshot.status !== 'error') {
            throw new GoalTaskRetryUnavailableError(
                this.getComponentName(),
                'The failed goal task could not be retried.',
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
            'The failed goal task could not be retried.',
        );
    }

    clear(): void {
        this.cancelActiveOperation('cancelled', new GoalReplacedError(
            this.getComponentName(),
            'The goal run was cleared.',
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
            'The goal run was disposed.',
        ));
        this.generation += 1;
        this.currentSnapshot = null;
        this.request = null;
    }

    private startOperation(
        run: () => Promise<GoalRunResult>,
    ): AiToolOperation<GoalRunResult> {
        this.cancelActiveOperation('replaced', new GoalReplacedError(
            this.getComponentName(),
            'The goal run was replaced.',
        ));
        let operation!: ActiveGoalOperation;
        const controller = createControlledAiToolOperation<
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
        return operation.controller.operation;
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
        this.generation += 1;
    }

    private async runPreparation(
        request: GoalRequest,
        generation: number,
        startIndex: number,
    ): Promise<GoalRunResult> {
        for (let index = startIndex; index < request.steps.length; index += 1) {
            if (generation !== this.generation) {
                throw new GoalReplacedError(this.getComponentName(), 'The goal run was cancelled.');
            }
            const step = request.steps[index];
            const attempt = (this.stepAttempts[index] ?? 0) + 1;
            const runId = createGoalRunId();
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
                'The goal step failed.',
                runId,
            );
            if (generation !== this.generation) {
                throw new GoalReplacedError(this.getComponentName(), 'The goal run was cancelled.');
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
            'The goal stop condition check failed.',
        );
        if (generation !== this.generation) {
            throw new GoalReplacedError(this.getComponentName(), 'The goal run was cancelled.');
        }
        let error = execution.error;
        let actual: number | null = null;
        if (!error) {
            actual = evaluateGoalStopWhenInput(execution.value);
            if (actual === null) {
                error = new GoalStopWhenInputIncompatibleError(
                    this.getComponentName(),
                    'Goal stop condition requires a ready query result with finite numeric data.',
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
                throw new GoalReplacedError(this.getComponentName(), 'The goal run was cancelled.');
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
        FailureError: AiToolComponentErrorConstructor<GoalComponentError>,
        fallbackMessage: string,
        runId = createGoalRunId(),
    ): Promise<RuntimeTaskExecutionResult> {
        const activeOperation = this.activeOperation;
        let operation: AiToolOperation<NestedAiToolResult, NestedAiToolStatus> | null = null;
        try {
            const dispatchedOperation = this.dispatchTool(
                toolName,
                argumentsValue,
            );
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
                result: NestedAiToolResult | Error;
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
        // The completed goal stays mounted until AI Chat replaces it.
    }

    private retryDelay(): Promise<void> {
        return new Promise((resolve) => setTimeout(resolve, RETRY_DELAY_MS));
    }
}

const getComparisonText = (snapshot: GoalSnapshot): string => {
    const stopWhen = snapshot.stop_when;
    if (!stopWhen) return snapshot.error || 'Invalid goal';
    const actual = snapshot.actual === null ? '—' : String(snapshot.actual);
    return `${actual} ${stopWhen.operator} ${stopWhen.target}`;
};

export const GoalDisplay: React.FC<GoalDisplayProps> = ({ snapshot, surface = 'chat' }) => {
    if (surface === 'chat' && snapshot.status === 'achieved') return null;

    const stopWhenResult = snapshot.stop_when_result;
    return (
        <section
            className={`ai-chat__goal ai-chat__goal--${surface} ai-chat__goal--${snapshot.status}`}
            aria-label="Goal"
            aria-live="polite"
        >
            <div className="ai-chat__goal-head">
                <div>
                    <span className="ai-chat__goal-kicker">GOAL · {snapshot.status}</span>
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

export const goalOverlayRenderer: AiOverlayRenderer<GoalSnapshot> = {
    componentType: 'goal',
    validateSnapshot: (snapshot): snapshot is GoalSnapshot => (
        isOverlayRecord(snapshot)
        && isOverlayNonEmptyString(snapshot.name)
        && ['running', 'achieved', 'missed', 'error'].includes(String(snapshot.status))
        && Array.isArray(snapshot.steps)
        && (snapshot.target === null || (typeof snapshot.target === 'number' && Number.isFinite(snapshot.target)))
        && (snapshot.actual === null || (typeof snapshot.actual === 'number' && Number.isFinite(snapshot.actual)))
    ),
    renderOverlay: (snapshot, status) => status === 'folded'
        ? getGoalOverlaySummary(snapshot)
        : <GoalOverlayDisplay snapshot={snapshot} />,
    dimensions: {
        expanded: { width: 420, height: 176 },
        folded: { width: 340, height: 58 },
    },
};

const Goal: React.FC<GoalProps> = ({ snapshot, surface = 'chat' }) => (
    snapshot ? <GoalDisplay snapshot={snapshot} surface={surface} /> : null
);

export default Goal;
