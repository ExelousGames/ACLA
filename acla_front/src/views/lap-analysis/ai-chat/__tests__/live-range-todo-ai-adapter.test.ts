import {
    createLiveRangeTodoAiAdapter,
} from '../live-range-todo-ai-adapter';
import type {
    LiveRangeTodoListHandle,
    LiveRangeTodoListToolResult,
    TaskStartFunction,
} from 'components/ai-engineering-tools';

const emptyResult: LiveRangeTodoListToolResult = {
    status: 'empty',
    todo_list: {
        events: [],
        current_position: null,
        rolling_rate: null,
        created_at: 1,
        updated_at: 1,
    },
};

const createHandle = (): jest.Mocked<LiveRangeTodoListHandle> => {
    const handle: LiveRangeTodoListHandle = {
        getComponentName: () => 'live-range-todo-list',
        addEvent: jest.fn((_event) => ({ ...emptyResult, status: 'ready' })),
        replaceEvents: jest.fn((_events) => ({ ...emptyResult, status: 'ready' })),
        updateEvents: jest.fn((_updates) => ({ ...emptyResult, status: 'ready' })),
        removeEvents: jest.fn((_ids) => emptyResult),
        resetEvents: jest.fn((_ids) => emptyResult),
        clear: jest.fn(() => emptyResult),
        get: jest.fn(() => emptyResult),
    };
    return handle as jest.Mocked<LiveRangeTodoListHandle>;
};

describe('live range to-do AI adapter', () => {
    it('attaches the selected task function only to AI set and add events', () => {
        const handle = createHandle();
        const taskStart = jest.fn() as TaskStartFunction;
        const selectTaskStart = jest.fn(() => taskStart);
        const adapter = createLiveRangeTodoAiAdapter(handle, selectTaskStart);

        adapter.set({
            events: [{
                id: 'set-event',
                normalized_position: 0.2,
                content: { title: 'Set event' },
                data: { event: 'custom_notification' },
            }],
        });
        adapter.update({
            action: 'add_events',
            events: [{
                id: 'added-event',
                normalized_position: 0.4,
                content: { title: 'Added event' },
            }],
        });

        expect(handle.replaceEvents).toHaveBeenCalledWith([
            expect.objectContaining({
                id: 'set-event',
                data: { event: 'custom_notification' },
                taskStart,
            }),
        ]);
        expect(handle.addEvent).toHaveBeenCalledWith(expect.objectContaining({
            id: 'added-event',
            data: {},
            taskStart,
        }));
        expect(selectTaskStart).toHaveBeenCalledTimes(2);
    });

    it('forwards only serializable AI updates so stored non-AI callbacks are preserved', () => {
        const handle = createHandle();
        const notifyAi = jest.fn() as TaskStartFunction;
        const nonAiTaskStart = jest.fn() as TaskStartFunction;
        const adapter = createLiveRangeTodoAiAdapter(handle, () => notifyAi);
        let storedTaskStart = nonAiTaskStart;
        handle.updateEvents.mockImplementation((updates) => {
            if (updates[0]?.taskStart) storedTaskStart = updates[0].taskStart;
            return { ...emptyResult, status: 'ready' };
        });

        adapter.update({
            action: 'update_events',
            events: [{
                id: 'component-event',
                content: { detail: 'AI-updated detail' },
                data: { note: 'serializable only' },
                taskStart: notifyAi,
                action: { name: 'notify_ai' },
            }],
        });

        expect(handle.updateEvents).toHaveBeenCalledWith([{
            id: 'component-event',
            content: { detail: 'AI-updated detail' },
            data: { note: 'serializable only' },
        }]);
        const forwardedUpdate = handle.updateEvents.mock.calls[0][0][0];
        expect(forwardedUpdate).not.toHaveProperty('taskStart');
        expect(forwardedUpdate).not.toHaveProperty('action');
        expect(storedTaskStart).toBe(nonAiTaskStart);
    });

    it('maps remove, reset, clear, and get without registering callbacks', () => {
        const handle = createHandle();
        const adapter = createLiveRangeTodoAiAdapter(handle, () => jest.fn());

        adapter.update({ action: 'remove_events', ids: ['one'] });
        adapter.update({ action: 'reset_events', ids: ['two'] });
        adapter.update({ action: 'reset_events' });
        adapter.update({ action: 'clear' });
        adapter.get();

        expect(handle.removeEvents).toHaveBeenCalledWith(['one']);
        expect(handle.resetEvents).toHaveBeenNthCalledWith(1, ['two']);
        expect(handle.resetEvents).toHaveBeenNthCalledWith(2);
        expect(handle.clear).toHaveBeenCalled();
        expect(handle.get).toHaveBeenCalled();
    });

    it('returns the panel unavailable error while the queue is unmounted', () => {
        const adapter = createLiveRangeTodoAiAdapter(null, () => jest.fn());

        [
            adapter.set({ events: [] }),
            adapter.update({ action: 'clear' }),
            adapter.get(),
        ].forEach((result) => {
            expect(result).toMatchObject({
                status: 'error',
                error: 'live_range_todo_list_unavailable',
                todo_list: null,
            });
        });
    });
});
