import type {
    JsonValue,
    LiveRangeTodoEventInput,
    LiveRangeTodoEventUpdate,
    LiveRangeTodoListHandle,
    LiveRangeTodoListToolResult,
    TaskStartFunction,
} from 'components/ai-engineering-tools';
import { AiToolError } from './ai-tool-base';

const isRecord = (value: unknown): value is Record<string, unknown> => (
    Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

const hasOwn = (value: Record<string, unknown>, key: string) => (
    Object.prototype.hasOwnProperty.call(value, key)
);

const missingLiveRangeTodoList = (): never => {
    throw new AiToolError(
        'live_range_todo_list_unavailable',
        'The AI chat live range to-do runtime is unavailable.',
    );
};

const invalidLiveRangeTodoList = (
    message: string,
): never => {
    throw new AiToolError('invalid_live_range_todo_list', message);
};

const returnOrThrowResult = (
    result: LiveRangeTodoListToolResult,
): LiveRangeTodoListToolResult => {
    if (result.status === 'error') {
        throw new AiToolError(
            result.error || 'invalid_live_range_todo_list',
            result.message || 'The live range to-do list request is invalid.',
        );
    }
    return result;
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
        return returnOrThrowResult(handle.replaceEvents(args.events.map((event) => (
            attachTaskStartFunction(event, createTaskStartFunction)
        ))));
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
                result = returnOrThrowResult(handle.addEvent(
                    attachTaskStartFunction(event, createTaskStartFunction),
                ));
            }
            return returnOrThrowResult(result);
        }
        if (action === 'update_events') {
            if (!Array.isArray(args.events)) {
                return invalidLiveRangeTodoList('Provide an events array.');
            }
            return returnOrThrowResult(handle.updateEvents(args.events.map(serializableUpdate)));
        }
        if (action === 'remove_events') {
            return returnOrThrowResult(handle.removeEvents(args.ids as readonly string[]));
        }
        if (action === 'reset_events') {
            return returnOrThrowResult(args.ids === undefined
                ? handle.resetEvents()
                : handle.resetEvents(args.ids as readonly string[]));
        }
        if (action === 'clear') return returnOrThrowResult(handle.clear());
        return invalidLiveRangeTodoList(
            `Unsupported live range to-do list action: ${action || '(missing)'}.`,
        );
    },

    get() {
        if (!handle) return missingLiveRangeTodoList();
        return returnOrThrowResult(handle.get());
    },
});
