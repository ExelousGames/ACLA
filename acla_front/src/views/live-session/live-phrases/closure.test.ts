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

it('selects the highest eligible closure priority and preserves declaration order for ties', () => {
    const throttle = new Closure<Context>('throttle', '', () => true, [], { priority: 1 });
    const turn = new Closure<Context>('turn', '', () => true, [], { priority: 2 });
    const brake = new Closure<Context>('brake', '', ({ ready }) => ready, [], { priority: 3 });
    const secondBrake = new Closure<Context>('second brake', '', ({ ready }) => ready, [], { priority: 3 });
    const root = new Closure<Context>('root', '', () => true, [throttle, turn, brake, secondBrake]);
    const ready = new State(root);
    ready.step(context);
    expect(ready.closure).toBe(brake);
    const unavailable = new State(root);
    unavailable.step({ ...context, ready: false });
    expect(unavailable.closure).toBe(turn);
    expect(root.priority).toBe(0);
    expect(ready.snapshot().children[2]).toMatchObject({ priority: 3, fields: { Priority: 3 } });
});

it('runs eligible actions by priority and preserves declaration order for ties', () => {
    const calls: string[] = [];
    const action = (name: string, priority?: number) => new Action<Context>(name, '', () => calls.push(name),
        () => true, { priority });
    const ordinary = action('ordinary');
    const root = new Closure<Context>('root', '', () => true, [
        ordinary, action('throttle', 1), action('turn', 2), action('brake', 3), action('second brake', 3),
    ]);
    const state = new State(root);
    root.children.forEach(() => state.step(context));
    expect(calls).toEqual(['brake', 'second brake', 'turn', 'throttle', 'ordinary']);
    expect(ordinary.priority).toBe(0);
    expect(state.snapshot().children[3]).toMatchObject({ priority: 3, fields: { Priority: 3 } });
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

it.each([false, true])('keeps the active closure when a higher-priority sibling becomes eligible (action completed: %s)', (completed) => {
    const work = new Action<Context>('work', 'Optional work.', jest.fn(), () => completed);
    const active = new Closure<Context>('active', 'Stay until explicitly exited.', () => true, [
        Action.exitToParent(({ exit }) => exit), work,
    ]);
    const eligible = jest.fn(({ ready }: Context) => ready);
    const sibling = new Closure<Context>('sibling', 'Wait for the active scope to exit.', eligible, [], { priority: 10 });
    const root = new Closure<Context>('root', 'Root.', () => true, [active, sibling]);
    const state = new State(root);
    state.step({ ...context, ready: false });
    eligible.mockClear();
    state.step(context);
    expect(state.closure).toBe(active);
    expect(eligible).not.toHaveBeenCalled();
    state.step({ ...context, exit: true });
    expect(state.path).toEqual([root]);
    expect(eligible).not.toHaveBeenCalled();
    state.step(context);
    expect(state.path).toEqual([root, sibling]);
});

it('waits for a nested closure to exit before evaluating its parent exit action', () => {
    const parentExit = jest.fn(({ ready }: Context) => ready);
    const nested = new Closure<Context>('nested', 'Own its exit.', () => true, [Action.exitToParent(({ exit }) => exit)]);
    const parent = new Closure<Context>('parent', 'Wait for the child.', () => true, [
        Action.exitToParent(parentExit, { priority: 10 }), nested,
    ]);
    const root = new Closure<Context>('root', 'Root.', () => true, [parent]);
    const state = new State(root);
    state.step({ ...context, ready: false });
    parentExit.mockClear();
    state.step(context);
    expect(state.path).toEqual([root, parent, nested]);
    expect(parentExit).not.toHaveBeenCalled();
    state.step({ ...context, exit: true });
    expect(state.path).toEqual([root, parent]);
    expect(parentExit).not.toHaveBeenCalled();
    state.step(context);
    expect(state.path).toEqual([root]);
    expect(parentExit).toHaveBeenCalledTimes(1);
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

it('exits one level, preserves parent actions, and resets nested actions for another visit', () => {
    const parentRun = jest.fn();
    const nestedRun = jest.fn();
    const parentAction = new Action<Context>('parent work', 'Run once in the parent.', parentRun);
    const nestedAction = new Action<Context>('nested work', 'Run once per nested visit.', nestedRun);
    const nested = new Closure<Context>('nested', 'Wait for an explicit exit.', ({ enter }) => enter, [
        nestedAction, Action.exitToParent(({ exit }) => exit),
    ]);
    const parent = new Closure<Context>('parent', 'Resume here after the nested exit.', () => true, [parentAction, nested]);
    const root = new Closure<Context>('root', 'Root.', () => true, [parent]);
    const state = new State(root);
    state.exitToParent();
    expect(state.path).toEqual([root]);
    state.step(context);
    state.step(context);
    state.step({ ...context, exit: true });
    expect(state.path).toEqual([root, parent]);
    expect(state.current).toBe(parent);
    expect(state.getExecution(parentAction).status).toBe('completed');
    expect(state.getExecution(nestedAction).status).toBe('idle');
    state.step(context);
    expect(nestedRun).toHaveBeenCalledTimes(2);
    expect(parentRun).toHaveBeenCalledTimes(1);
    state.step({ ...context, exit: true });
    expect(state.path).toEqual([root, parent]);
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
        expect(snapshot.status).toBe('Active');
        expect(snapshot.children.map((child) => child.status)).toEqual(['Not evaluated', 'Not evaluated']);
        expect(snapshot.children.map((child) => child.kind)).toEqual(['action', 'closure']);
        expect(snapshot.children[1].children[0]).toMatchObject({
            id: 'root/1/0', name: 'same name', description: 'Nested work.', kind: 'action', status: 'Not evaluated',
            execution: { status: 'idle' }, fields: {},
            conditions: [{ description: 'Work ready', conditionFit: null }], children: [],
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
        expect(state.snapshot(context).children[0]).toMatchObject({
            status: 'Not matched', conditions: [{ description: 'Custom condition (last checked)', conditionFit: false }],
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
            'work', 'Work.', jest.fn(), describeCondition(() => false, () => [group]), { inspect: () => ({ fields }) },
        )]));
        state.step(context);
        const before = state.snapshot(context);
        leaf.conditionFit = false;
        fields.Count = 2;
        expect(before.children[0]).toMatchObject({ status: 'Not matched', fields: { Count: 1 }, conditions: [{ conditions: [{ conditionFit: true }] }] });
        state.step(context);
        expect(state.snapshot(context).children[0]).toMatchObject({ fields: { Count: 2 }, conditions: [{ conditions: [{ conditionFit: false }] }] });
    });
});

it('does not measure or inspect descendants until their parent is entered, including during snapshots', () => {
    const measure = jest.fn(({ ready }: Context) => ready);
    const describe = jest.fn(({ ready }: Context) => [{ description: 'Ready', conditionFit: ready, inputMissing: false }]);
    const inspect = jest.fn(() => ({ fields: { Count: 1 } }));
    const action = new Action<Context>('work', 'Work.', jest.fn(), describeCondition(measure, describe), { inspect });
    const parent = new Closure<Context>('parent', 'Parent.', ({ enter }) => enter, [action]);
    const state = new State(new Closure<Context>('root', 'Root.', () => true, [parent]));
    state.step({ ...context, enter: false });
    state.snapshot(context);
    state.snapshot(context);
    expect(measure).not.toHaveBeenCalled();
    expect(describe).not.toHaveBeenCalled();
    expect(inspect).not.toHaveBeenCalled();
    expect(state.snapshot(context).children[0]).toMatchObject({
        status: 'Not matched', children: [{ status: 'Not evaluated', conditions: [{ conditionFit: null }] }],
    });
    state.step({ ...context, ready: false });
    expect(state.snapshot(context).children[0]).toMatchObject({
        status: 'Active', children: [{ status: 'Not matched', conditions: [{ conditionFit: false }] }],
    });
    expect(measure).toHaveBeenCalledTimes(1);
    expect(describe).toHaveBeenCalledTimes(1);
    state.snapshot(context);
    expect(measure).toHaveBeenCalledTimes(1);
    expect(describe).toHaveBeenCalledTimes(1);
    state.exitToRoot();
    expect(state.snapshot(context).children[0].children[0]).toMatchObject({
        status: 'Not evaluated', conditions: [{ conditionFit: null, inputMissing: false }],
    });
});
