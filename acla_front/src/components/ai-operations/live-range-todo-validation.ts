import { InvalidLiveRangeTodoListError } from 'contexts/OperationComponentError';
import { OPERATION_COMPONENT_NAMES } from 'contexts/OperationComponentRefContext';
import { readToolCall, type WorkflowDispatcher } from './tool';
import { readWorkflowCall } from './workflow';
import type { FrontendOperationName } from 'views/lap-analysis/ai-chat/ai-command-registry';
import type { LiveRangeTodoEventInput } from './live-range-todo-list-types';

const isRecord = (value: unknown): value is Record<string, unknown> => Boolean(value) && typeof value === 'object' && !Array.isArray(value);
const hasOwn = (value: Record<string, unknown>, key: string) => Object.prototype.hasOwnProperty.call(value, key);

const invalidLiveRangeTodoList = (message: string): never => {
    throw new InvalidLiveRangeTodoListError(
        OPERATION_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST,
        message,
    );
};

const assertExactKeys = (
    value: Record<string, unknown>,
    allowed: readonly string[],
    label: string,
): void => {
    const unsupported = Reflect.ownKeys(value).find((key) => (
        typeof key !== 'string' || !allowed.includes(key)
    ));
    if (unsupported !== undefined) {
        invalidLiveRangeTodoList(
            `${label} property '${String(unsupported)}' is not supported.`,
        );
    }
};

const isJsonSafe = (value: unknown, ancestors = new Set<object>()): boolean => {
    if (value === null || typeof value === 'string' || typeof value === 'boolean') return true;
    if (typeof value === 'number') return Number.isFinite(value);
    if (typeof value !== 'object') return false;
    if (ancestors.has(value)) return false;
    const prototype = Object.getPrototypeOf(value);
    if (!Array.isArray(value) && prototype !== Object.prototype && prototype !== null) return false;
    ancestors.add(value);
    const valid = Array.isArray(value)
        ? value.every((entry) => isJsonSafe(entry, ancestors))
        : Reflect.ownKeys(value).every((key) => (
            typeof key === 'string'
            && isJsonSafe((value as Record<string, unknown>)[key], ancestors)
        ));
    ancestors.delete(value);
    return valid;
};

type PreparedLiveRangeTodoEvent = {
    event: Omit<LiveRangeTodoEventInput, 'taskStart'>;
    tool: {
        name: FrontendOperationName;
        arguments: Record<string, unknown>;
    };
};

export const validateLiveRangeTodoBatch = (
    args: unknown,
    dispatchNested: WorkflowDispatcher,
    workflowName: 'add_event_to_live_range_todo_list' | 'create_live_range_todo_list' = 'add_event_to_live_range_todo_list',
): PreparedLiveRangeTodoEvent[] => {
    const input = readWorkflowCall(args, workflowName);
    if (!input) invalidLiveRangeTodoList(`Provide workflow with name ${workflowName}.`);
    const request = input as Record<string, unknown>;
    assertExactKeys(request, ['name', 'tools'], 'Live range to-do request');
    if (!Array.isArray(request.tools) || request.tools.length === 0) {
        invalidLiveRangeTodoList('Provide at least one tool to schedule.');
    }
    const rawEvents = request.tools as unknown[];

    const ids = new Set<string>();
    return rawEvents.map((item, index) => {
        const itemLabel = `Live range to-do item ${index + 1}`;
        const toolValue = readToolCall(item);
        if (!toolValue) invalidLiveRangeTodoList(`${itemLabel} requires tool with a name.`);
        const rawItem = toolValue as Record<string, unknown>;
        const toolName = rawItem.name as string;
        try {
            dispatchNested.validate(toolName);
        } catch (error) {
            invalidLiveRangeTodoList(error instanceof Error ? error.message : String(error));
        }
        assertExactKeys(rawItem, ['name', 'event', 'arguments'], itemLabel);
        if (!hasOwn(rawItem, 'event') || !hasOwn(rawItem, 'arguments')) {
            invalidLiveRangeTodoList(`${itemLabel} requires event and arguments objects.`);
        }

        const eventValue = rawItem.event;
        if (!isRecord(eventValue)) invalidLiveRangeTodoList(`${itemLabel} event must be an object.`);
        const rawEvent = eventValue as Record<string, unknown>;
        assertExactKeys(
            rawEvent,
            ['id', 'normalized_position', 'lead_time_seconds', 'content'],
            `${itemLabel} event`,
        );
        const id = typeof rawEvent.id === 'string' ? rawEvent.id.trim() : '';
        if (!id) invalidLiveRangeTodoList(`${itemLabel} event requires a non-empty id.`);
        if (ids.has(id)) invalidLiveRangeTodoList(`Duplicate live range to-do event id: ${id}.`);
        ids.add(id);
        if (
            typeof rawEvent.normalized_position !== 'number'
            || !Number.isFinite(rawEvent.normalized_position)
            || rawEvent.normalized_position < 0
            || rawEvent.normalized_position > 1
        ) {
            invalidLiveRangeTodoList(`Event '${id}' normalized_position must be between 0 and 1.`);
        }
        if (hasOwn(rawEvent, 'lead_time_seconds') && (
            typeof rawEvent.lead_time_seconds !== 'number'
            || !Number.isFinite(rawEvent.lead_time_seconds)
            || rawEvent.lead_time_seconds < 0
        )) {
            invalidLiveRangeTodoList(`Event '${id}' lead_time_seconds must be zero or greater.`);
        }
        if (!isRecord(rawEvent.content)) {
            invalidLiveRangeTodoList(`Event '${id}' requires a structured content object.`);
        }
        const rawContent = rawEvent.content as Record<string, unknown>;
        assertExactKeys(rawContent, ['title', 'description'], `Event '${id}' content`);
        const title = typeof rawContent.title === 'string'
            ? rawContent.title.trim()
            : '';
        if (!title) invalidLiveRangeTodoList(`Event '${id}' content requires a non-empty title.`);
        if (hasOwn(rawContent, 'description')
            && typeof rawContent.description !== 'string') {
            invalidLiveRangeTodoList(`Event '${id}' content description must be a string.`);
        }

        const rawTool = rawItem;
        if (!hasOwn(rawTool, 'arguments') || !isRecord(rawTool.arguments)) {
            invalidLiveRangeTodoList(`Scheduled tool '${toolName}' requires an arguments object.`);
        }
        if (!isJsonSafe(rawTool.arguments)) {
            invalidLiveRangeTodoList(`Scheduled tool '${toolName}' arguments must be JSON-safe.`);
        }
        const normalizedPosition = rawEvent.normalized_position as number;
        const leadTimeSeconds = rawEvent.lead_time_seconds as number | undefined;
        const description = rawContent.description as string | undefined;
        const toolArguments = rawTool.arguments as Record<string, unknown>;

        return {
            event: {
                id,
                normalized_position: normalizedPosition,
                ...(leadTimeSeconds !== undefined
                    ? { lead_time_seconds: leadTimeSeconds }
                    : {}),
                content: {
                    title,
                    ...(description !== undefined
                        ? { description }
                        : {}),
                },
            },
            tool: {
                name: toolName as FrontendOperationName,
                arguments: JSON.parse(JSON.stringify(toolArguments)),
            },
        };
    });
};
