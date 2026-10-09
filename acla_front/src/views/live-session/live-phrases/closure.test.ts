import { Action, Closure, State, describeCondition } from './closure';

interface Context { enter: boolean; ready: boolean; exit: boolean }
const context: Context = { enter: true, ready: true, exit: false };

it('starts at root and enters the first eligible child in declaration order', () => {
    const run = jest.fn();
    const first = new Closure<Context>('first', 'Enter the first branch when requested.', ({ enter }) => enter, [new Action('work', 'Invoke the work callback.', run)]);
    const second = new Closure<Context>('second', 'Enter the fallback branch.', () => true);
    const root = new Closure<Context>('root', 'Choose the first eligible child.', () => true, [first, second]);
    const state = new State(root);
    expect(state.current).toBe(root);
    expect(state.path).toEqual([root]);
    state.step(context);
    expect(state.closure).toBe(first);
    expect(state.current).toBe(first.children[0]);
    expect(run).toHaveBeenCalledWith(context, state);
    const other = new State(root);
    other.step({ ...context, enter: false });
    expect(other.current).toBe(second);
});

it('retains a nested closure when entry conditions change or its children finish', () => {
    const run = jest.fn();
    const nested = new Closure<Context>('nested', 'Run work in a nested scope.', ({ enter }) => enter, [new Action('work', 'Invoke the work callback.', run)]);
    const parentWork = jest.fn();
    const parent = new Closure<Context>('parent', 'Choose between nested and parent work.', ({ enter }) => enter, [nested, new Action('parent work', 'Invoke the parent callback.', parentWork)]);
    const siblingWork = jest.fn();
    const root = new Closure<Context>('root', 'Choose the first eligible child.', () => true, [parent, new Action('sibling work', 'Invoke the sibling callback.', siblingWork)]);
    const state = new State(root);
    state.step(context);
    state.step({ ...context, enter: false });
    state.step({ ...context, enter: false });
    expect(state.path).toEqual([root, parent, nested, nested.children[0]]);
    expect(run).toHaveBeenCalledTimes(1);
    expect(parentWork).not.toHaveBeenCalled();
    expect(siblingWork).not.toHaveBeenCalled();
});

it('evaluates action conditions inside the entered closure and runs callbacks once per visit', () => {
    const run = jest.fn(() => 42);
    const action = new Action<Context>('conditional', 'Run the callback when ready.', run, ({ ready }) => ready);
    const child = new Closure<Context>('child', 'Run the child actions.', () => true, [action]);
    const state = new State(new Closure<Context>('root', 'Choose the first eligible child.', () => true, [child]));
    state.step({ ...context, ready: false });
    expect(state.current).toBe(child);
    expect(run).not.toHaveBeenCalled();
    expect(state.step(context)).toBe(42);
    state.step(context);
    expect(run).toHaveBeenCalledTimes(1);
});

it('uses an explicit exit action to return directly to root and start a new visit', () => {
    const run = jest.fn();
    const exit = Action.exitToRoot<Context>(({ exit }) => exit);
    const nested = new Closure<Context>('nested', 'Run work in a nested scope.', () => true, [new Action('work', 'Invoke the work callback.', run), exit]);
    const root = new Closure<Context>('root', 'Choose the first eligible child.', () => true, [new Closure('parent', 'Choose between nested and parent work.', () => true, [nested])]);
    const state = new State(root);
    state.step(context);
    state.step(context);
    expect(state.closure).toBe(nested);
    state.step({ ...context, exit: true });
    expect(state.current).toBe(root);
    expect(state.path).toEqual([root]);
    state.step(context);
    expect(run).toHaveBeenCalledTimes(2);
});

it('runs unconditional actions, including exit to root, as soon as they are reached', () => {
    const calls: string[] = [];
    const root = new Closure<Context>('root', 'Choose the first eligible child.', () => true, [new Closure('child', 'Run the child actions.', () => true, [
        new Action('one', 'Record the first call.', () => calls.push('one')),
        new Action('two', 'Record the second call.', () => calls.push('two')),
        Action.exitToRoot(),
    ])]);
    const state = new State(root);
    state.step(context);
    state.step(context);
    expect(calls).toEqual(['one', 'two']);
    state.step(context);
    expect(state.current).toBe(root);
});

it('keeps independent states for the same tree', () => {
    const run = jest.fn();
    const root = new Closure<Context>('root', 'Choose the first eligible child.', () => true, [new Action('work', 'Invoke the work callback.', run)]);
    const first = new State(root);
    const second = new State(root);
    first.step(context);
    expect(second.current).toBe(root);
    second.step(context);
    expect(run).toHaveBeenCalledTimes(2);
});

it('waits for an asynchronous callback without repeating it or advancing to the next action', async () => {
    let finish!: (value: number) => void;
    const run = jest.fn(() => new Promise<number>((resolve) => { finish = resolve; }));
    const after = jest.fn();
    const action = new Action<Context>('async work', 'Wait for asynchronous work to finish.', run);
    const state = new State(new Closure<Context>('root', 'Choose the first eligible child.', () => true, [action, new Action('after', 'Run after the asynchronous work finishes.', after)]));
    const result = state.step(context);
    state.step(context);
    expect(state.current).toBe(action);
    expect(run).toHaveBeenCalledTimes(1);
    expect(after).not.toHaveBeenCalled();
    finish(42);
    await expect(result).resolves.toBe(42);
    state.step(context);
    expect(after).toHaveBeenCalledTimes(1);
});

it('propagates action failures without retrying their side effects or leaving the closure', async () => {
    const error = new Error('failed');
    const run = jest.fn(async () => { throw error; });
    const child = new Closure<Context>('child', 'Run the child actions.', () => true, [new Action('work', 'Invoke the work callback.', run)]);
    const state = new State(new Closure('root', 'Choose the first eligible child.', () => true, [child]));
    await expect(state.step(context)).rejects.toBe(error);
    state.step(context);
    expect(state.closure).toBe(child);
    expect(run).toHaveBeenCalledTimes(1);
});

describe('tree inspection', () => {
    it('includes mixed descendants and duplicate names with distinct IDs without executing callbacks', () => {
        const run = jest.fn();
        const branch = new Closure<Context>('same name', 'Nested branch.', describeCondition(({ enter }) => enter, 'Enter requested'), [
            new Action('same name', 'Nested work.', run, describeCondition(({ ready }) => ready, 'Work ready'), {
                inspect: ({ ready }) => ({ fields: { Ready: ready, Attempts: 0, Destination: 'logger' } }),
            }),
        ]);
        const state = new State(new Closure<Context>('root', 'Root.', () => true, [
            new Action('same name', 'Root work.', run), branch,
        ]));
        const snapshot = state.snapshot({ ...context, ready: false });
        expect(snapshot.children.map((child) => child.kind)).toEqual(['action', 'closure']);
        expect(snapshot.children[1].children[0]).toMatchObject({
            id: 'root/1/0', name: 'same name', description: 'Nested work.', kind: 'action', status: 'Inactive',
            execution: { status: 'idle' }, fields: { Ready: false, Attempts: 0, Destination: 'logger' },
            conditions: [{ description: 'Work ready', conditionFit: false }], children: [],
        });
        expect(snapshot.children[0].id).not.toBe(snapshot.children[1].id);
        expect(JSON.parse(JSON.stringify(snapshot))).toMatchObject(snapshot);
        expect(state.current).toBe(state.root);
        expect(run).not.toHaveBeenCalled();
    });

    it('never evaluates opaque predicates while inspecting and exposes only their last checked result', () => {
        const predicate = jest.fn(({ ready }: Context) => ready);
        const action = new Action<Context>('opaque', 'Opaque callback.', jest.fn(), predicate);
        const state = new State(new Closure<Context>('root', 'Root.', () => true, [action]));
        expect(state.snapshot(context).children[0].conditions[0].conditionFit).toBeNull();
        expect(predicate).not.toHaveBeenCalled();
        state.step({ ...context, ready: false });
        expect(state.snapshot(context).children[0].conditions[0]).toMatchObject({
            description: 'Custom condition (last checked)', conditionFit: false,
        });
        expect(predicate).toHaveBeenCalledTimes(1);
    });

    it('tracks each action independently through running and completion without changing older snapshots', async () => {
        let finish!: () => void;
        const action = new Action<Context>('work', 'Async work.', () => new Promise<void>((resolve) => { finish = resolve; }));
        const root = new Closure<Context>('root', 'Root.', () => true, [new Closure('nested', 'Nested.', () => true, [
            action, new Action('later', 'Later work.', jest.fn()),
        ])]);
        const state = new State(root);
        const result = state.step(context);
        const running = state.snapshot(context);
        expect(running.children[0]).toMatchObject({ onPath: true, current: false });
        expect(running.children[0].children[0]).toMatchObject({ status: 'Running', current: true, execution: { status: 'running' } });
        expect(running.children[0].children[1].execution?.status).toBe('idle');
        finish();
        await result;
        expect(state.snapshot(context).children[0].children[0].execution?.status).toBe('completed');
        expect(running.children[0].children[0].execution?.status).toBe('running');
        expect(new State(root).snapshot(context).children[0].children[0].execution?.status).toBe('idle');
        state.exitToRoot();
        expect(state.snapshot(context).children[0].children[0].execution?.status).toBe('idle');
    });

    it.each([false, true])('exposes failed execution and its error (async: %s)', async (async) => {
        const error = new Error('Storage unavailable');
        const run = jest.fn(() => { if (async) return Promise.reject(error); throw error; });
        const state = new State(new Closure<Context>('root', 'Root.', () => true, [new Action('persist', 'Save data.', run)]));
        if (async) await expect(state.step(context)).rejects.toBe(error);
        else expect(() => state.step(context)).toThrow(error);
        expect(state.snapshot(context).children[0]).toMatchObject({
            status: 'Failed', execution: { status: 'failed', error: 'Storage unavailable' },
        });
        state.step(context);
        expect(run).toHaveBeenCalledTimes(1);
    });

    it('copies condition trees and custom fields out of mutable inspection data', () => {
        const leaf = { description: 'Ready', conditionFit: true, inputMissing: false };
        const group = { ...leaf, description: 'Group', conditions: [leaf] };
        const fields = { Count: 1 };
        const state = new State(new Closure<Context>('root', 'Root.', () => true, [new Action(
            'work', 'Work.', jest.fn(), describeCondition(() => true, () => [group]), { inspect: () => ({ fields }) },
        )]));
        const before = state.snapshot(context);
        leaf.conditionFit = false;
        fields.Count = 2;
        expect(before.children[0]).toMatchObject({ fields: { Count: 1 }, conditions: [{ conditions: [{ conditionFit: true }] }] });
        expect(state.snapshot(context).children[0]).toMatchObject({ fields: { Count: 2 }, conditions: [{ conditions: [{ conditionFit: false }] }] });
    });
});
