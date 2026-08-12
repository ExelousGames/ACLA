import type {
    JsonValue,
    LiveRangeTodoEventInput,
    LiveRangeTodoEventUpdate,
    LiveRangeTodoListHandle,
    LiveRangeTodoListToolResult,
    TaskStartFunction,
} from 'components/ai-engineering-tools';
import { AI_TOOL_COMPONENT_NAMES } from 'contexts/AiToolComponentRefContext';
import {
    InvalidLiveRangeTodoListError,
    LiveRangeTodoListUnavailableError,
} from 'contexts/AiToolComponentError';

const isRecord = (value: unknown): value is Record<string, unknown> => (
    Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

const hasOwn = (value: Record<string, unknown>, key: string) => (
    Object.prototype.hasOwnProperty.call(value, key)
);

const missingLiveRangeTodoList = (): never => {
    throw new LiveRangeTodoListUnavailableError(
        AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST,
        'The AI chat live range to-do runtime is unavailable.',
    );
};

const invalidLiveRangeTodoList = (
    message: string,
): never => {
    throw new InvalidLiveRangeTodoListError(
        AI_TOOL_COMPONENT_NAMES.LIVE_RANGE_TODO_LIST,
        message,
    );
};

export type LiveRangeTodoTaskDescriptor = Omit<LiveRangeTodoEventInput, 'taskStart'>;

export type LiveRangeTodoTaskStartFunctionFactory = (
    event: LiveRangeTodoTaskDescriptor,
) => TaskStartFunction;

const attachTaskStartFunction = (
    value: unknown,
    createTaskStartFunction: LiveRangeTodoTaskStartFunctionFactory,
): LiveRangeTodoEventInput => {
    if (!isRecord(value)) return value as LiveRangeTodoEventInput;
    const event: LiveRangeTodoTaskDescriptor = {
        id: value.id as string,
        normalized_position: value.normalized_position as number,
        ...(hasOwn(value, 'lead_time_seconds')
            ? { lead_time_seconds: value.lead_time_seconds as number }
            : {}),
        content: value.content as LiveRangeTodoEventInput['content'],
        data: (value.data === undefined ? {} : value.data) as JsonValue,
    };
    return { ...event, taskStart: createTaskStartFunction(event) };
};

const serializableUpdate = (value: unknown): LiveRangeTodoEventUpdate => {
    if (!isRecord(value)) return value as LiveRangeTodoEventUpdate;
    return {
        id: value.id as string,
        ...(hasOwn(value, 'normalized_position')
            ? { normalized_position: value.normalized_position as number }
            : {}),
        ...(hasOwn(value, 'lead_time_seconds')
            ? { lead_time_seconds: value.lead_time_seconds as number }
            : {}),
        ...(hasOwn(value, 'content')
            ? { content: value.content as LiveRangeTodoEventUpdate['content'] }
            : {}),
        ...(hasOwn(value, 'data') ? { data: value.data as JsonValue } : {}),
    };
};

export interface LiveRangeTodoAiAdapter {
    set: (args: Record<string, unknown>) => LiveRangeTodoListToolResult;
    update: (args: Record<string, unknown>) => LiveRangeTodoListToolResult;
    get: () => LiveRangeTodoListToolResult;
}

export const createLiveRangeTodoAiAdapter = (
    handle: LiveRangeTodoListHandle | null,
    createTaskStartFunction: LiveRangeTodoTaskStartFunctionFactory,
): LiveRangeTodoAiAdapter => ({
    set(args) {
        if (!handle) return missingLiveRangeTodoList();
        if (!Array.isArray(args.events)) {
            return invalidLiveRangeTodoList('Provide an events array.');
        }
        return handle.replaceEvents(args.events.map((event) => (
            attachTaskStartFunction(event, createTaskStartFunction)
        )));
    },

    update(args) {
        if (!handle) return missingLiveRangeTodoList();
        const action = typeof args.action === 'string' ? args.action : '';
        if (action === 'add_events') {
            if (!Array.isArray(args.events) || args.events.length === 0) {
                return invalidLiveRangeTodoList('Provide at least one event to add.');
            }
            let result = handle.get();
            for (const event of args.events) {
                result = handle.addEvent(
                    attachTaskStartFunction(event, createTaskStartFunction),
                );
            }
            return result;
        }
        if (action === 'update_events') {
            if (!Array.isArray(args.events)) {
                return invalidLiveRangeTodoList('Provide an events array.');
            }
            return handle.updateEvents(args.events.map(serializableUpdate));
        }
        if (action === 'remove_events') {
            return handle.removeEvents(args.ids as readonly string[]);
        }
        if (action === 'reset_events') {
            return args.ids === undefined
                ? handle.resetEvents()
                : handle.resetEvents(args.ids as readonly string[]);
        }
        if (action === 'clear') return handle.clear();
        return invalidLiveRangeTodoList(
            `Unsupported live range to-do list action: ${action || '(missing)'}.`,
        );
    },

    get() {
        if (!handle) return missingLiveRangeTodoList();
        return handle.get();
    },
});
