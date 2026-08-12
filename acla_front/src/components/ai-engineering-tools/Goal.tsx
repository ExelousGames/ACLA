import React, { useEffect, useMemo, useRef, useState } from 'react';
import type { NamedAiToolComponentHandle } from 'contexts/AiToolComponentRefContext';
import { useRegisterAiToolComponentRef } from 'contexts/AiToolComponentRefContext';
import {
    AiToolComponentErrorConstructor,
    DuplicateGoalStepIdError,
    GoalClearedError,
    GoalComponentError,
    GoalDeterminationFailedError,
    GoalDeterminationOutputToolMismatchError,
    GoalDeterminationTaskUnavailableError,
    GoalDeterminationValueNotNumericError,
    GoalDisposedError,
    GoalReplacedError,
    GoalStepFailedError,
    GoalStepOutputToolMismatchError,
    GoalStepTaskUnavailableError,
    GoalTaskRetryUnavailableError,
    InvalidGoalDeterminationError,
    InvalidGoalNameError,
    InvalidGoalStepsError,
    RecursiveGoalDeterminationError,
    RecursiveGoalStepError,
} from 'contexts/AiToolComponentError';
import { AiToolComponentBase } from './AiToolComponentBase';
import type { TaskStartFunction } from './task-start-function';

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
export type GoalDeterminationStatus = GoalStepStatus;

export type GoalStepDescriptor = {
    id: string;
    title: string;
    name: string;
    arguments?: Record<string, unknown>;
};

export type GoalDeterminationTool = {
    name: string;
    arguments?: Record<string, unknown>;
};

export type GoalDetermination = {
    tool: GoalDeterminationTool;
    result_path: string;
    operator: GoalComparisonOperator;
    target: number;
};

export type GoalRequest = {
    name: string;
    steps: GoalStepDescriptor[];
    determination: GoalDetermination;
};

export type GoalTaskDescriptor = {
    title: string;
    name: string;
    arguments?: Record<string, unknown>;
};

export type GoalStep = GoalStepDescriptor & {
    taskStart: TaskStartFunction;
};

export type GoalExecutableDetermination = GoalDetermination & {
    taskStart: TaskStartFunction;
};

export type GoalExecutableRequest = Omit<GoalRequest, 'steps' | 'determination'> & {
    steps: GoalStep[];
    determination: GoalExecutableDetermination;
};

export type GoalStepSnapshot = GoalStepDescriptor & {
    status: GoalStepStatus;
    attempts: number;
    run_id?: string;
    error?: string;
};

export type GoalSourceResultMetadata = {
    tool_name: string;
    run_id: string;
    status: string;
    final: true;
};

export type GoalStepSourceResultMetadata = GoalSourceResultMetadata & {
    step_id: string;
};

export type GoalTaskResult = {
    step_id: string;
    tool_name: string;
    attempt: number;
    status: 'completed' | 'error';
    value: unknown;
    error?: string;
    source_result?: GoalStepSourceResultMetadata;
};

export type GoalDeterminationResult = {
    tool_name: string;
    attempt: number;
    status: GoalDeterminationStatus;
    value: number | null;
    error?: string;
    source_result?: GoalSourceResultMetadata;
};

export type GoalSnapshot = {
    name: string;
    status: GoalStatus;
    steps: GoalStepSnapshot[];
    determination: GoalDetermination | null;
    determination_result: GoalDeterminationResult | null;
    target: number | null;
    actual: number | null;
    completed_steps: string[];
    failed_step?: string;
    error?: string;
};

export type GoalToolOutputEnvelope = {
    tool_name: string;
    run_id: string;
    status: string;
    output: unknown;
    final: boolean;
    message?: string;
};

export type GoalRunResult = Pick<
    GoalSnapshot,
    | 'name'
    | 'target'
    | 'actual'
    | 'completed_steps'
    | 'determination'
    | 'determination_result'
> & {
    status: 'achieved' | 'missed';
    task_results: GoalTaskResult[];
};

export interface GoalHandle extends NamedAiToolComponentHandle {
    createGoal(input: GoalExecutableRequest): Promise<GoalRunResult>;
    retryFailedTask(): Promise<GoalRunResult>;
    acceptToolOutput(envelope: GoalToolOutputEnvelope): void;
    getSnapshot(): GoalSnapshot | null;
    clear(): void;
}

export type GoalDisplayProps = {
    snapshot: GoalSnapshot;
    surface?: 'chat' | 'pill';
};

export type GoalProps = {
    name: string;
    onSnapshotChange?: (snapshot: GoalSnapshot | null) => void;
    surface?: 'chat' | 'pill';
};

export type GoalTaskStartFunctionSelector = (
    task: GoalTaskDescriptor,
) => TaskStartFunction | null | undefined;

type FinalWaiter = {
    toolName: string;
    runId: string | null;
    resolve: (envelope: GoalToolOutputEnvelope) => void;
    reject: (error: Error) => void;
};

const RETRY_DELAY_MS = 1000;
const MAX_GOAL_ATTEMPTS = 2;
const RECURSIVE_GOAL_TOOL_NAMES = new Set(['create_goal', 'retry_goal_task']);
const UNSAFE_RESULT_PATH_SEGMENTS = new Set(['__proto__', 'prototype', 'constructor']);
const RESULT_PATH_SEGMENT_RE = /^(?:[A-Za-z_][A-Za-z0-9_]*|0|[1-9][0-9]*)$/;

const isRecord = (value: unknown): value is Record<string, unknown> => (
    Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

const hasOnlyKeys = (value: Record<string, unknown>, allowedKeys: readonly string[]): boolean => {
    const allowed = new Set(allowedKeys);
    return Object.keys(value).every((key) => allowed.has(key));
};

const isGoalToolOutputEnvelope = (value: unknown): value is GoalToolOutputEnvelope => (
    isRecord(value)
    && typeof value.tool_name === 'string'
    && typeof value.run_id === 'string'
    && typeof value.status === 'string'
    && typeof value.final === 'boolean'
    && Object.prototype.hasOwnProperty.call(value, 'output')
);

const normalizeTaskValue = (value: unknown): unknown => (
    value === undefined ? null : value
);

const normalizeTaskError = (value: unknown, fallback = 'The goal step failed.'): string => {
    if (value instanceof Error && value.message.trim()) return value.message.trim();
    if (typeof value === 'string' && value.trim()) return value.trim();
    if (isRecord(value)) {
        if (typeof value.error === 'string' && value.error.trim()) return value.error.trim();
        if (typeof value.message === 'string' && value.message.trim()) return value.message.trim();
    }
    if (value !== null && value !== undefined) {
        if (typeof value !== 'object') return String(value);
        try {
            const serialized = JSON.stringify(value);
            if (serialized && serialized !== '{}') return serialized;
        } catch {
            // Fall through to the stable generic error.
        }
    }
    return fallback;
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

export const isSafeGoalResultPath = (path: unknown): path is string => {
    const normalized = toNonEmptyString(path);
    if (!normalized) return false;
    const segments = normalized.split('.');
    return segments.every((segment) => (
        RESULT_PATH_SEGMENT_RE.test(segment)
        && !UNSAFE_RESULT_PATH_SEGMENTS.has(segment)
    ));
};

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

const parseGoalDeterminationTool = (value: unknown): GoalDeterminationTool | null => {
    const tool = isRecord(value) ? value : null;
    if (!tool || !hasOnlyKeys(tool, ['name', 'arguments'])) return null;
    const name = toNonEmptyString(tool.name);
    if (!name || (tool.arguments !== undefined && !isRecord(tool.arguments))) return null;
    return {
        name,
        ...(tool.arguments !== undefined ? { arguments: { ...tool.arguments } } : {}),
    };
};

const parseGoalDetermination = (value: unknown): GoalDetermination | null => {
    const determination = isRecord(value) ? value : null;
    if (!determination || !hasOnlyKeys(
        determination,
        ['tool', 'result_path', 'operator', 'target'],
    )) return null;
    const tool = parseGoalDeterminationTool(determination.tool);
    const resultPath = toNonEmptyString(determination.result_path);
    if (
        !tool
        || !resultPath
        || !isSafeGoalResultPath(resultPath)
        || !isGoalComparisonOperator(determination.operator)
        || typeof determination.target !== 'number'
        || !Number.isFinite(determination.target)
    ) {
        return null;
    }
    return {
        tool,
        result_path: resultPath,
        operator: determination.operator,
        target: determination.target,
    };
};

export const validateGoalRequest = (
    value: unknown,
    componentName = 'goal',
): { request: GoalRequest } | { error: GoalComponentError; name?: string } => {
    const input = isRecord(value) ? value : null;
    const name = toNonEmptyString(input?.name);
    if (!input || !name || !hasOnlyKeys(input, ['name', 'steps', 'determination'])) {
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
    const determination = parseGoalDetermination(input.determination);
    if (!determination) {
        return {
            error: new InvalidGoalDeterminationError(componentName, 'Provide a valid goal determination.'),
            name,
        };
    }
    if (RECURSIVE_GOAL_TOOL_NAMES.has(determination.tool.name)) {
        return {
            error: new RecursiveGoalDeterminationError(componentName, 'Goal determination cannot invoke a goal-management tool.'),
            name,
        };
    }
    return { request: { name, steps: parsedSteps, determination } };
};

export const buildGoalRequest = (
    value: unknown,
    selectTaskStartFunction: GoalTaskStartFunctionSelector,
    componentName = 'goal',
): { request: GoalExecutableRequest } | { error: GoalComponentError; name?: string } => {
    const validation = validateGoalRequest(value, componentName);
    if ('error' in validation) return validation;
    const { request } = validation;
    const steps: GoalStep[] = [];
    for (const step of request.steps) {
        const taskStart = selectTaskStartFunction(step);
        if (typeof taskStart !== 'function') {
            return {
                error: new GoalStepTaskUnavailableError(
                    componentName,
                    `Goal step tool '${step.name}' is unavailable.`,
                ),
                name: request.name,
            };
        }
        steps.push({ ...step, taskStart });
    }
    const determinationTaskStart = selectTaskStartFunction({
        title: request.name,
        name: request.determination.tool.name,
        ...(request.determination.tool.arguments
            ? { arguments: { ...request.determination.tool.arguments } }
            : {}),
    });
    if (typeof determinationTaskStart !== 'function') {
        return {
            error: new GoalDeterminationTaskUnavailableError(
                componentName,
                `Goal determination tool '${request.determination.tool.name}' is unavailable.`,
            ),
            name: request.name,
        };
    }
    return {
        request: {
            name: request.name,
            steps,
            determination: {
                ...cloneDetermination(request.determination),
                taskStart: determinationTaskStart,
            },
        },
    };
};

const validateExecutableGoalRequest = (
    value: unknown,
    componentName: string,
): { request: GoalExecutableRequest } | { error: GoalComponentError; name?: string } => {
    const input = isRecord(value) ? value : null;
    const name = toNonEmptyString(input?.name);
    if (!input || !name || !hasOnlyKeys(input, ['name', 'steps', 'determination'])) {
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
    const publicSteps: GoalStepDescriptor[] = [];
    const taskStarts: TaskStartFunction[] = [];
    for (const rawStep of input.steps) {
        if (
            !isRecord(rawStep)
            || !hasOnlyKeys(rawStep, ['id', 'title', 'name', 'arguments', 'taskStart'])
            || typeof rawStep.taskStart !== 'function'
        ) {
            return {
                error: new InvalidGoalStepsError(componentName, 'Every executable goal step must include a task start function.'),
                name,
            };
        }
        const descriptor = parseGoalStepDescriptor({
            id: rawStep.id,
            title: rawStep.title,
            name: rawStep.name,
            ...(rawStep.arguments !== undefined ? { arguments: rawStep.arguments } : {}),
        });
        if (!descriptor) {
            return {
                error: new InvalidGoalStepsError(componentName, 'Every goal step must have a valid id, title, name, and arguments object.'),
                name,
            };
        }
        publicSteps.push(descriptor);
        taskStarts.push(rawStep.taskStart as TaskStartFunction);
    }
    const rawDetermination = isRecord(input.determination) ? input.determination : null;
    if (
        !rawDetermination
        || !hasOnlyKeys(
            rawDetermination,
            ['tool', 'result_path', 'operator', 'target', 'taskStart'],
        )
        || typeof rawDetermination.taskStart !== 'function'
    ) {
        return {
            error: new InvalidGoalDeterminationError(componentName, 'Provide a valid executable goal determination.'),
            name,
        };
    }
    const publicRequest = {
        name,
        steps: publicSteps,
        determination: {
            tool: rawDetermination.tool,
            result_path: rawDetermination.result_path,
            operator: rawDetermination.operator,
            target: rawDetermination.target,
        },
    };
    const validation = validateGoalRequest(publicRequest, componentName);
    if ('error' in validation) return validation;
    return {
        request: {
            name: validation.request.name,
            steps: validation.request.steps.map((step, index) => ({
                ...step,
                taskStart: taskStarts[index],
            })),
            determination: {
                ...cloneDetermination(validation.request.determination),
                taskStart: rawDetermination.taskStart as TaskStartFunction,
            },
        },
    };
};

export const extractGoalResultPath = (value: unknown, path: string): unknown => {
    if (!isSafeGoalResultPath(path)) return undefined;
    let current = value;
    for (const segment of path.split('.')) {
        if (Array.isArray(current) && /^\d+$/.test(segment)) {
            const index = Number(segment);
            if (!Object.prototype.hasOwnProperty.call(current, index)) return undefined;
            current = current[index];
            continue;
        }
        if (!isRecord(current) || !Object.prototype.hasOwnProperty.call(current, segment)) {
            return undefined;
        }
        current = current[segment];
    }
    return current;
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

const cloneDeterminationTool = (tool: GoalDeterminationTool): GoalDeterminationTool => ({
    ...tool,
    ...(tool.arguments ? { arguments: { ...tool.arguments } } : {}),
});

const cloneDetermination = (determination: GoalDetermination): GoalDetermination => ({
    ...determination,
    tool: cloneDeterminationTool(determination.tool),
});

const cloneDeterminationResult = (
    result: GoalDeterminationResult | null,
): GoalDeterminationResult | null => result ? ({
    ...result,
    ...(result.source_result ? { source_result: { ...result.source_result } } : {}),
}) : null;

const cloneSnapshot = (snapshot: GoalSnapshot): GoalSnapshot => ({
    ...snapshot,
    steps: snapshot.steps.map((step) => ({
        ...step,
        ...(step.arguments ? { arguments: { ...step.arguments } } : {}),
    })),
    determination: snapshot.determination
        ? cloneDetermination(snapshot.determination)
        : null,
    determination_result: cloneDeterminationResult(snapshot.determination_result),
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
    determination: snapshot.determination
        ? cloneDetermination(snapshot.determination)
        : null,
    determination_result: cloneDeterminationResult(snapshot.determination_result),
    target: snapshot.target,
    actual: snapshot.actual,
    completed_steps: [...snapshot.completed_steps],
    task_results: cloneTaskResults(taskResults),
});

type TaskExecutionResult = {
    value: unknown;
    error?: GoalComponentError;
    source_result?: GoalSourceResultMetadata;
};

export class GoalRunner extends AiToolComponentBase<GoalSnapshot | null> {
    private currentSnapshot: GoalSnapshot | null = null;
    private controller: AbortController | null = null;
    private generation = 0;
    private finalWaiter: FinalWaiter | null = null;
    private request: GoalExecutableRequest | null = null;
    private failedStepIndex: number | null = null;
    private determinationFailed = false;
    private stepAttempts: number[] = [];
    private determinationAttempts = 0;
    private taskResults: GoalTaskResult[] = [];
    private goalAttempt = 1;
    private readonly cancellationErrors = new WeakMap<AbortController, GoalComponentError>();
    private readonly onChange?: (snapshot: GoalSnapshot | null) => void;

    constructor(
        componentName: string,
        onChange?: (snapshot: GoalSnapshot | null) => void,
    ) {
        super(componentName, null);
        this.onChange = onChange;
    }

    getSnapshot(): GoalSnapshot | null {
        return this.currentSnapshot ? cloneSnapshot(this.currentSnapshot) : null;
    }

    acceptToolOutput(envelope: GoalToolOutputEnvelope): void {
        const waiter = this.finalWaiter;
        if (!waiter || waiter.toolName !== envelope.tool_name) return;
        if (waiter.runId && waiter.runId !== envelope.run_id) return;
        if (!waiter.runId) waiter.runId = envelope.run_id;
        if (!envelope.final) return;
        this.finalWaiter = null;
        waiter.resolve(envelope);
    }

    async create(input: GoalExecutableRequest): Promise<GoalRunResult> {
        this.cancelActive(
            GoalReplacedError,
            'The goal was replaced by a newer goal.',
        );
        const generation = ++this.generation;
        this.request = null;
        this.failedStepIndex = null;
        this.determinationFailed = false;
        this.stepAttempts = [];
        this.determinationAttempts = 0;
        this.taskResults = [];
        this.goalAttempt = 1;
        const validation = validateExecutableGoalRequest(input, this.getComponentName());
        if ('error' in validation) {
            const invalidSnapshot: GoalSnapshot = {
                name: validation.name || 'Goal',
                status: 'error',
                steps: [],
                determination: null,
                determination_result: null,
                target: null,
                actual: null,
                completed_steps: [],
                error: validation.error.message,
            };
            this.publish(invalidSnapshot);
            throw validation.error;
        }

        const request = validation.request;
        this.request = request;
        this.stepAttempts = request.steps.map(() => 0);
        const controller = new AbortController();
        this.controller = controller;
        this.publish(this.createRunningSnapshot(request));
        return this.runPreparation(
            request,
            generation,
            controller,
            0,
            'The goal could not be completed.',
        );
    }

    async retryFailedTask(): Promise<GoalRunResult> {
        const request = this.request;
        if (!request || !this.currentSnapshot || this.currentSnapshot.status !== 'error') {
            return this.retryUnavailableError();
        }

        const generation = ++this.generation;
        const controller = new AbortController();
        this.controller = controller;
        const failureMessage = 'The failed goal task could not be retried.';
        if (
            this.failedStepIndex !== null
            && this.currentSnapshot.failed_step === request.steps[this.failedStepIndex]?.id
        ) {
            const startIndex = this.failedStepIndex;
            this.failedStepIndex = null;
            const { failed_step: _failedStep, error: _error, ...retrySnapshot } = this.currentSnapshot;
            this.publish({
                ...retrySnapshot,
                status: 'running',
                actual: null,
            });
            return this.runPreparation(
                request,
                generation,
                controller,
                startIndex,
                failureMessage,
            );
        }
        if (this.determinationFailed) {
            this.determinationFailed = false;
            const { failed_step: _failedStep, error: _error, ...retrySnapshot } = this.currentSnapshot;
            this.publish({
                ...retrySnapshot,
                status: 'running',
                actual: null,
                determination_result: this.pendingDeterminationResult(request),
            });
            return this.runDetermination(
                request,
                generation,
                controller,
                failureMessage,
            );
        }
        this.controller = null;
        return this.retryUnavailableError();
    }

    clear(): void {
        this.cancelActive(GoalClearedError, 'The goal was cleared.');
        this.generation += 1;
        this.currentSnapshot = null;
        this.request = null;
        this.failedStepIndex = null;
        this.determinationFailed = false;
        this.stepAttempts = [];
        this.determinationAttempts = 0;
        this.taskResults = [];
        this.onChange?.(null);
        this.publishSnapshot(null);
        this.deleteComponentRef();
    }

    protected onDispose(): void {
        this.cancelActive(GoalDisposedError, 'The goal runner was disposed.');
        this.currentSnapshot = null;
        this.request = null;
        this.failedStepIndex = null;
        this.determinationFailed = false;
    }

    private async runPreparation(
        request: GoalExecutableRequest,
        generation: number,
        controller: AbortController,
        startIndex: number,
        failureMessage: string,
    ): Promise<GoalRunResult> {
        for (let index = startIndex; index < request.steps.length; index += 1) {
            const step = request.steps[index];
            if (!this.isCurrent(generation, controller)) {
                return this.cancelledError(controller, failureMessage);
            }

            const attempt = (this.stepAttempts[index] ?? 0) + 1;
            this.stepAttempts[index] = attempt;
            this.updateStep(index, {
                status: 'running',
                attempts: attempt,
                run_id: undefined,
                error: undefined,
            });

            const execution = await this.executeTask(
                step.name,
                step.taskStart,
                controller.signal,
                GoalStepFailedError,
                GoalStepOutputToolMismatchError,
                'The goal step failed.',
                `Goal step '${step.id}' returned output for a different tool.`,
            );
            if (!this.isCurrent(generation, controller)) {
                return this.cancelledError(controller, failureMessage);
            }

            const sourceResult = execution.source_result ? {
                ...execution.source_result,
                step_id: step.id,
            } : undefined;
            const taskResult: GoalTaskResult = {
                step_id: step.id,
                tool_name: step.name,
                attempt,
                status: execution.error ? 'error' : 'completed',
                value: execution.value,
                ...(execution.error ? { error: execution.error.message } : {}),
                ...(sourceResult ? { source_result: sourceResult } : {}),
            };
            this.taskResults.push(taskResult);

            if (execution.error) {
                this.updateStep(index, {
                    status: 'error',
                    run_id: sourceResult?.run_id,
                    error: execution.error.message,
                });
                this.failedStepIndex = index;
                this.determinationFailed = false;
                const errorSnapshot: GoalSnapshot = {
                    ...this.currentSnapshot!,
                    status: 'error',
                    actual: null,
                    failed_step: step.id,
                    error: execution.error.message,
                };
                this.publish(errorSnapshot);
                this.controller = null;
                throw execution.error;
            }

            this.updateStep(index, {
                status: 'completed',
                run_id: sourceResult?.run_id,
                error: undefined,
            });
            const completed = this.currentSnapshot!.steps
                .filter((item) => item.status === 'completed')
                .map((item) => item.id);
            this.publish({ ...this.currentSnapshot!, completed_steps: completed });
        }

        return this.runDetermination(
            request,
            generation,
            controller,
            failureMessage,
        );
    }

    private async runDetermination(
        request: GoalExecutableRequest,
        generation: number,
        controller: AbortController,
        failureMessage: string,
    ): Promise<GoalRunResult> {
        if (!this.isCurrent(generation, controller)) {
            return this.cancelledError(controller, failureMessage);
        }
        const attempt = this.determinationAttempts + 1;
        this.determinationAttempts = attempt;
        this.publish({
            ...this.currentSnapshot!,
            determination_result: {
                tool_name: request.determination.tool.name,
                attempt,
                status: 'running',
                value: null,
            },
        });

        const execution = await this.executeTask(
            request.determination.tool.name,
            request.determination.taskStart,
            controller.signal,
            GoalDeterminationFailedError,
            GoalDeterminationOutputToolMismatchError,
            'The goal determination failed.',
            'The goal determination returned output for a different tool.',
        );
        if (!this.isCurrent(generation, controller)) {
            return this.cancelledError(controller, failureMessage);
        }

        let error = execution.error;
        let actual: number | null = null;
        if (!error) {
            const value = extractGoalResultPath(
                execution.value,
                request.determination.result_path,
            );
            if (typeof value !== 'number' || !Number.isFinite(value)) {
                error = new GoalDeterminationValueNotNumericError(
                    this.getComponentName(),
                    `Goal determination path '${request.determination.result_path}' did not resolve to a finite number.`,
                );
            } else {
                actual = value;
            }
        }
        if (error) {
            this.failedStepIndex = null;
            this.determinationFailed = true;
            const { failed_step: _failedStep, error: _error, ...currentSnapshot } = this.currentSnapshot!;
            this.publish({
                ...currentSnapshot,
                status: 'error',
                actual: null,
                determination_result: {
                    tool_name: request.determination.tool.name,
                    attempt,
                    status: 'error',
                    value: null,
                    error: error.message,
                    ...(execution.source_result
                        ? { source_result: { ...execution.source_result } }
                        : {}),
                },
            });
            this.controller = null;
            throw error;
        }

        this.failedStepIndex = null;
        this.determinationFailed = false;
        const achieved = compareGoalValues(
            actual!,
            request.determination.operator,
            request.determination.target,
        );
        const { failed_step: _failedStep, error: _error, ...currentSnapshot } = this.currentSnapshot!;
        const finalSnapshot: GoalSnapshot & { status: 'achieved' | 'missed' } = {
            ...currentSnapshot,
            status: achieved ? 'achieved' : 'missed',
            actual,
            determination_result: {
                tool_name: request.determination.tool.name,
                attempt,
                status: 'completed',
                value: actual,
                ...(execution.source_result
                    ? { source_result: { ...execution.source_result } }
                    : {}),
            },
        };
        this.publish(finalSnapshot);
        if (achieved || this.goalAttempt === MAX_GOAL_ATTEMPTS) {
            this.controller = null;
            this.deleteComponentRef();
            return toRunResult(finalSnapshot, this.taskResults);
        }

        try {
            await this.retryDelay(controller.signal);
        } catch {
            return this.cancelledError(controller, failureMessage);
        }
        if (!this.isCurrent(generation, controller)) {
            return this.cancelledError(controller, failureMessage);
        }
        this.goalAttempt += 1;
        this.publish(this.createRunningSnapshot(request));
        return this.runPreparation(
            request,
            generation,
            controller,
            0,
            failureMessage,
        );
    }

    private createRunningSnapshot(request: GoalExecutableRequest): GoalSnapshot {
        return {
            name: request.name,
            status: 'running',
            steps: request.steps.map((step, index) => ({
                id: step.id,
                title: step.title,
                name: step.name,
                ...(step.arguments ? { arguments: { ...step.arguments } } : {}),
                status: 'pending',
                attempts: this.stepAttempts[index] ?? 0,
            })),
            determination: cloneDetermination(request.determination),
            determination_result: this.pendingDeterminationResult(request),
            target: request.determination.target,
            actual: null,
            completed_steps: [],
        };
    }

    private pendingDeterminationResult(
        request: GoalExecutableRequest,
    ): GoalDeterminationResult {
        return {
            tool_name: request.determination.tool.name,
            attempt: this.determinationAttempts,
            status: 'pending',
            value: null,
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

    private isCurrent(generation: number, controller: AbortController): boolean {
        return (
            generation === this.generation
            && this.controller === controller
            && !controller.signal.aborted
        );
    }

    private cancelActive(
        ErrorType: AiToolComponentErrorConstructor<GoalComponentError>,
        message: string,
    ): void {
        if (!this.controller) return;
        const controller = this.controller;
        const error = new ErrorType(this.getComponentName(), message);
        this.cancellationErrors.set(controller, error);
        controller.abort();
        this.controller = null;
        this.finalWaiter?.reject(error);
        this.finalWaiter = null;
    }

    private cancelledError(controller: AbortController, message: string): never {
        throw this.cancellationErrors.get(controller)
            || new GoalReplacedError(this.getComponentName(), message);
    }

    private retryUnavailableError(): never {
        throw new GoalTaskRetryUnavailableError(
            this.getComponentName(),
            'The failed goal task could not be retried.',
        );
    }

    private async executeTask(
        toolName: string,
        taskStart: TaskStartFunction,
        signal: AbortSignal,
        FailureError: AiToolComponentErrorConstructor<GoalComponentError>,
        MismatchError: AiToolComponentErrorConstructor<GoalComponentError>,
        fallbackMessage: string,
        mismatchMessage: string,
    ): Promise<TaskExecutionResult> {
        let deliveredEnvelope: GoalToolOutputEnvelope | null = null;
        let waiter!: FinalWaiter;
        const finalPromise = new Promise<GoalToolOutputEnvelope>((resolve, reject) => {
            waiter = {
                toolName,
                runId: null,
                resolve: (envelope) => {
                    deliveredEnvelope = envelope;
                    resolve(envelope);
                },
                reject,
            };
            this.finalWaiter = waiter;
        });
        void finalPromise.catch(() => undefined);
        const abortFinalWait = () => {
            if (this.finalWaiter !== waiter) return;
            this.finalWaiter = null;
            waiter.reject(new Error('goal_cancelled'));
        };
        signal.addEventListener('abort', abortFinalWait, { once: true });

        try {
            const returned = await taskStart(signal);
            if (deliveredEnvelope) {
                return this.executionFromEnvelope(deliveredEnvelope);
            }
            if (isGoalToolOutputEnvelope(returned)) {
                if (returned.tool_name !== toolName) {
                    return {
                        value: normalizeTaskValue(returned.output),
                        error: new MismatchError(this.getComponentName(), mismatchMessage),
                    };
                }
                if (returned.final) return this.executionFromEnvelope(returned);
                waiter.runId = returned.run_id;
                return this.executionFromEnvelope(await finalPromise);
            }
            if (returned === undefined && waiter.runId) {
                return this.executionFromEnvelope(await finalPromise);
            }

            return { value: normalizeTaskValue(returned) };
        } catch (error) {
            return {
                value: null,
                error: error instanceof GoalComponentError
                    ? error
                    : new FailureError(
                        this.getComponentName(),
                        normalizeTaskError(error, fallbackMessage),
                        { cause: error },
                    ),
            };
        } finally {
            signal.removeEventListener('abort', abortFinalWait);
            if (this.finalWaiter === waiter) this.finalWaiter = null;
        }
    }

    private executionFromEnvelope(
        envelope: GoalToolOutputEnvelope,
    ): TaskExecutionResult {
        const sourceResult: GoalSourceResultMetadata = {
            tool_name: envelope.tool_name,
            run_id: envelope.run_id,
            status: envelope.status,
            final: true,
        };
        return {
            value: normalizeTaskValue(envelope.output),
            source_result: sourceResult,
        };
    }

    private retryDelay(signal: AbortSignal): Promise<void> {
        return new Promise((resolve, reject) => {
            const timer = setTimeout(resolve, RETRY_DELAY_MS);
            signal.addEventListener('abort', () => {
                clearTimeout(timer);
                reject(new Error('goal_cancelled'));
            }, { once: true });
        });
    }
}

const getComparisonText = (snapshot: GoalSnapshot): string => {
    const determination = snapshot.determination;
    if (!determination) return snapshot.error || 'Invalid goal';
    const actual = snapshot.actual === null ? '—' : String(snapshot.actual);
    return `${actual} ${determination.operator} ${determination.target}`;
};

export const GoalDisplay: React.FC<GoalDisplayProps> = ({ snapshot, surface = 'chat' }) => {
    const determinationResult = snapshot.determination_result;
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
                                {step.error && <span className="ai-chat__goal-error">{step.error}</span>}
                            </span>
                        </li>
                    ))}
                </ol>
            )}
            {snapshot.determination && determinationResult && (
                <div
                    className={`ai-chat__goal-determination ai-chat__goal-determination--${determinationResult.status}`}
                    aria-label="Determination"
                >
                    <span className="ai-chat__goal-step-dot" aria-hidden="true" />
                    <span className="ai-chat__goal-step-copy">
                        <span>Determination · {determinationResult.tool_name}</span>
                        <span>
                            {determinationResult.status}
                            {determinationResult.attempt > 0
                                ? ` · attempt ${determinationResult.attempt}`
                                : ''}
                        </span>
                        <span className="ai-chat__goal-metric">{getComparisonText(snapshot)}</span>
                        {determinationResult.error && (
                            <span className="ai-chat__goal-error">{determinationResult.error}</span>
                        )}
                    </span>
                </div>
            )}
            {snapshot.error && snapshot.steps.length === 0 && (
                <div className="ai-chat__goal-error">{snapshot.error}</div>
            )}
        </section>
    );
};

const Goal: React.FC<GoalProps> = ({
    name,
    onSnapshotChange,
    surface = 'chat',
}) => {
    const [snapshot, setSnapshot] = useState<GoalSnapshot | null>(null);
    const onSnapshotChangeRef = useRef(onSnapshotChange);
    onSnapshotChangeRef.current = onSnapshotChange;
    const runnerRef = useRef<GoalRunner | null>(null);
    if (!runnerRef.current) {
        runnerRef.current = new GoalRunner(name, (next) => {
            setSnapshot(next);
            onSnapshotChangeRef.current?.(next);
        });
    }

    const handle = useMemo<GoalHandle>(() => ({
        getComponentName: () => name,
        createGoal: (input) => runnerRef.current!.create(input),
        retryFailedTask: () => runnerRef.current!.retryFailedTask(),
        acceptToolOutput: (envelope) => runnerRef.current!.acceptToolOutput(envelope),
        getSnapshot: () => runnerRef.current!.getSnapshot(),
        clear: () => runnerRef.current!.clear(),
    }), [name]);
    useRegisterAiToolComponentRef(name, handle);

    useEffect(() => () => runnerRef.current?.dispose(), []);

    return snapshot ? <GoalDisplay snapshot={snapshot} surface={surface} /> : null;
};

export default Goal;
