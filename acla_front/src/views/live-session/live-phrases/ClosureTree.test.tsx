import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { Action, Closure, State, describeCondition } from './closure';
import ClosureTree from './ClosureTree';

it('distinguishes unchecked, failed and entered closures as the tree advances and resets', () => {
    const nested = new Closure<{ enter: boolean; ready: boolean }>('nested', 'Nested scope.', ({ ready }) => ready);
    const parent = new Closure<{ enter: boolean; ready: boolean }>('parent', 'Parent scope.', ({ enter }) => enter, [nested]);
    const state = new State(new Closure('root', 'Root.', () => true, [parent]));
    const view = render(<ClosureTree node={state.snapshot()} />);
    expect(within(screen.getByLabelText('parent closure')).getByText('Not evaluated')).toBeVisible();

    state.step({ enter: false, ready: true });
    view.rerender(<ClosureTree node={state.snapshot()} />);
    expect(within(screen.getByLabelText('parent closure')).getByText('Not matched')).toBeVisible();
    fireEvent.click(screen.getByLabelText('parent closure'));
    expect(within(screen.getByLabelText('nested closure')).getByText('Not evaluated')).toBeVisible();

    state.step({ enter: true, ready: false });
    view.rerender(<ClosureTree node={state.snapshot()} />);
    expect(screen.getByText('Active')).toBeVisible();
    expect(within(screen.getByLabelText('nested closure')).getByText('Not matched')).toBeVisible();

    state.step({ enter: true, ready: true });
    state.step({ enter: false, ready: false });
    view.rerender(<ClosureTree node={state.snapshot()} />);
    expect(within(screen.getByLabelText('nested closure')).getByText('Active')).toBeVisible();

    state.exitToRoot();
    view.rerender(<ClosureTree node={state.snapshot()} />);
    expect(screen.getAllByText('Not evaluated')).toHaveLength(2);
    fireEvent.click(screen.getByRole('button', { name: 'Go back one level' }));
    expect(within(screen.getByLabelText('parent closure')).getByText('Not evaluated')).toBeVisible();
});

it('renders arbitrary mixed nodes, condition groups and extra fields from a serialized recursive snapshot', () => {
    const run = jest.fn();
    const action = new Action<{ ready: boolean }>('record telemetry', 'Save a diagnostic sample.', run,
        describeCondition(({ ready }) => ready, ({ ready }) => [{
            description: 'Recording requirements', conditionFit: ready, inputMissing: false, conditions: [
                { description: 'Recorder available', conditionFit: ready, inputMissing: false },
                { description: 'Storage available', conditionFit: true, inputMissing: false, connector: 'and' },
            ],
        }]), { inspect: () => ({ fields: { Destination: 'diagnostics', Retries: 0, Enabled: false, Result: null } }) });
    let branch: Closure<{ ready: boolean }> = new Closure('level 4', 'Deepest scope.', () => true, [action]);
    for (let depth = 3; depth >= 1; depth--) branch = new Closure(`level ${depth}`, 'Nested scope.', () => true, [branch]);
    const state = new State(new Closure('root', 'Root.', () => true, [
        new Action('record telemetry', 'Work at root.', run, () => false), branch,
    ]));
    state.step({ ready: false });
    const view = render(<ClosureTree node={JSON.parse(JSON.stringify(state.snapshot({ ready: false })))} />);
    expect(screen.getByRole('button', { name: 'Go back one level' })).toBeDisabled();
    expect(screen.getByLabelText('root action 1: record telemetry')).toBeVisible();
    fireEvent.click(screen.getByLabelText('root action 1: record telemetry'));
    expect(screen.getByText('Work at root.')).toBeVisible();
    expect(screen.queryByLabelText('level 1 closure')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Go back one level' }));
    for (let depth = 1; depth <= 4; depth++) {
        const summary = screen.getByLabelText(`level ${depth} closure`);
        expect(summary).toBeVisible();
        fireEvent.click(summary);
    }
    const summary = screen.getByLabelText('level 4 action 1: record telemetry');
    expect(summary).toBeVisible();
    fireEvent.click(summary);
    const body = within(screen.getByRole('tabpanel', { name: 'Details' }));
    expect(body.getByText('Save a diagnostic sample.')).toBeVisible();
    expect(body.getByText('diagnostics')).toBeVisible();
    expect(body.getByText('0')).toBeVisible();
    expect(body.getByText('false')).toBeVisible();
    expect(body.getByText('—')).toBeVisible();
    fireEvent.click(screen.getByRole('tab', { name: 'Conditions' }));
    const conditions = within(screen.getByRole('list', { name: 'level 4 action 1: record telemetry conditions' }));
    expect(conditions.getByText('Recorder available')).toBeVisible();
    expect(conditions.getAllByText('Not met')).toHaveLength(2);
    expect(conditions.getByText('AND')).toBeVisible();
    expect(run).not.toHaveBeenCalled();
    // A data-only refresh retains the inspected directory and tab.
    state.step({ ready: true });
    view.rerender(<ClosureTree node={JSON.parse(JSON.stringify(state.snapshot({ ready: true })))} />);
    expect(conditions.getByText('Recorder available')).toBeVisible();
    expect(conditions.getAllByText('Met')).toHaveLength(3);
    expect(screen.getByRole('tab', { name: 'Conditions' })).toHaveAttribute('aria-selected', 'true');
    const breadcrumbs = within(screen.getByRole('navigation', { name: 'Phrase directory' }));
    expect(breadcrumbs.getByText('record telemetry')).toHaveAttribute('aria-current', 'page');
    fireEvent.click(screen.getByRole('button', { name: 'Go back one level' }));
    expect(screen.getByRole('heading', { name: 'level 4' })).toHaveFocus();
    expect(screen.getByLabelText('level 4 action 1: record telemetry')).toBeVisible();
    fireEvent.click(breadcrumbs.getByRole('button', { name: 'level 1' }));
    expect(screen.getByLabelText('level 2 closure')).toBeVisible();
    expect(screen.queryByLabelText('level 3 closure')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Go back one level' }));
    expect(screen.getByRole('button', { name: 'Go back one level' })).toBeDisabled();
    expect(screen.getByLabelText('root action 1: record telemetry')).toBeVisible();
});

it('displays independent action execution, current location and failure details', async () => {
    let reject!: (reason: Error) => void;
    const action = new Action<{}>('upload sample', 'Upload diagnostics.', () => new Promise<void>((_resolve, fail) => { reject = fail; }));
    const state = new State(new Closure<{}>('root', 'Root.', () => true, [action, new Action('later', 'Later work.', jest.fn())]));
    const pending = state.step({});
    const view = render(<ClosureTree node={state.snapshot({})} />);
    const summary = screen.getByLabelText('root action 1: upload sample');
    expect(within(summary).getByText('Running')).toBeVisible();
    expect(summary).toHaveAttribute('data-current', 'true');
    expect(within(screen.getByLabelText('root action 2: later')).getByText('Action · idle')).toBeVisible();
    fireEvent.click(summary);
    reject(new Error('Upload failed'));
    await expect(pending).rejects.toThrow('Upload failed');
    view.rerender(<ClosureTree node={state.snapshot({})} />);
    expect(screen.getByText('Failed')).toBeVisible();
    expect(screen.getByText('Upload failed')).toBeVisible();
});

it('supports keyboard tabs, empty directories and returning from a removed directory', () => {
    const state = new State(new Closure('root', 'Root details.', () => true, [new Closure('empty', 'Empty directory.', () => true)]));
    const snapshot = state.snapshot({});
    const view = render(<ClosureTree node={snapshot} />);
    const contents = screen.getByRole('tab', { name: 'Contents 1' });
    contents.focus();
    fireEvent.keyDown(contents, { key: 'ArrowRight' });
    expect(screen.getByRole('tab', { name: 'Conditions' })).toHaveFocus();
    expect(screen.getByRole('tabpanel', { name: 'Conditions' })).toBeVisible();
    expect(screen.queryByRole('list', { name: 'root children' })).not.toBeInTheDocument();
    fireEvent.keyDown(screen.getByRole('tab', { name: 'Conditions' }), { key: 'End' });
    expect(screen.getByRole('tab', { name: 'Details' })).toHaveFocus();
    expect(screen.getByText('Root details.')).toBeVisible();
    fireEvent.keyDown(screen.getByRole('tab', { name: 'Details' }), { key: 'Home' });
    fireEvent.click(screen.getByLabelText('empty closure'));
    expect(screen.getByText('No items in this closure.')).toBeVisible();
    view.rerender(<ClosureTree node={{ ...snapshot, children: [] }} />);
    expect(screen.getByRole('heading', { name: 'root' })).toBeVisible();
    expect(screen.getByRole('button', { name: 'Go back one level' })).toBeDisabled();
});

it('shows direct exit alternatives and live grouped conditions from serialized snapshots without inspecting predicates', () => {
    type Context = { released: boolean; safe?: boolean };
    const exitPredicate = jest.fn(({ released, safe }: Context) => released && safe === true);
    const parentExit = Action.exitToParent<Context>(describeCondition(exitPredicate, ({ released, safe }) => [{
        description: 'Release requirements', conditionFit: released && safe === true, inputMissing: safe === undefined,
        conditions: [
            { description: 'Speech released', conditionFit: released, inputMissing: false },
            { description: 'Safe to leave', conditionFit: safe === true, inputMissing: safe === undefined, connector: 'and' },
        ],
    }]));
    const rootExit = Action.exitToRoot<Context>(describeCondition(() => false, 'Emergency exit requested'));
    const ordinary = new Action<Context>('exit to root', 'An ordinary action with an exit-like name.', jest.fn(),
        describeCondition(() => false, 'Ordinary action requirement'));
    const nested = new Closure<Context>('nested', 'Nested scope.', () => false, [
        Action.exitToRoot(describeCondition(() => false, 'Nested exit requirement')),
    ]);
    const state = new State(new Closure<Context>('root', 'Root.', () => true, [
        new Closure('guide', 'Guide.', describeCondition(() => true, 'Guide is eligible'), [parentExit, rootExit, ordinary, nested]),
    ]));
    state.step({ released: false, safe: false });
    const snapshot = () => JSON.parse(JSON.stringify(state.snapshot()));
    const view = render(<ClosureTree node={snapshot()} />);
    fireEvent.click(screen.getByLabelText('guide closure'));
    fireEvent.click(screen.getByRole('tab', { name: 'Conditions' }));
    expect(screen.getByText('Entry conditions')).toBeVisible();
    expect(screen.getByRole('list', { name: 'guide conditions' })).toHaveTextContent('Guide is eligible');
    expect(screen.getByText('Exit conditions')).toBeVisible();
    const exits = within(screen.getByRole('list', { name: 'guide exit conditions' }));
    expect(exits.getByText('OR')).toBeVisible();
    const parent = within(exits.getByRole('group', { name: 'guide exit 1: exit to parent' }));
    expect(parent.getByRole('list', { name: 'guide exit 1: exit to parent conditions' })).toBeVisible();
    expect(parent.getByRole('group', { name: 'Condition group' })).toBeVisible();
    expect(parent.getByText('AND')).toBeVisible();
    expect(parent.getAllByText('Not met')).toHaveLength(3);
    expect(exits.getByRole('group', { name: 'guide exit 2: exit to root' })).toHaveTextContent('Emergency exit requested');
    expect(exits.queryByText('Ordinary action requirement')).not.toBeInTheDocument();
    expect(exits.queryByText('Nested exit requirement')).not.toBeInTheDocument();
    expect(exitPredicate).toHaveBeenCalledTimes(1);

    state.step({ released: true, safe: false });
    view.rerender(<ClosureTree node={snapshot()} />);
    expect(screen.getByRole('tab', { name: 'Conditions' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('heading', { name: 'guide' })).toBeVisible();
    expect(within(parent.getByText('Speech released').closest('li')!).getByText('Met')).toBeVisible();
    expect(exitPredicate).toHaveBeenCalledTimes(2);

    state.step({ released: true });
    view.rerender(<ClosureTree node={snapshot()} />);
    expect(within(parent.getByText('Safe to leave').closest('li')!).getByText('Missing input')).toBeVisible();
    expect(exitPredicate).toHaveBeenCalledTimes(3);
});

it('shows an empty exit state for closures and keeps action conditions separate', () => {
    const action = new Action('exit to root', 'An ordinary action.', jest.fn(), describeCondition(() => false, 'Action ready'));
    const state = new State(new Closure('root', 'Root.', () => true, [action]));
    render(<ClosureTree node={JSON.parse(JSON.stringify(state.snapshot()))} />);
    fireEvent.click(screen.getByRole('tab', { name: 'Conditions' }));
    expect(screen.getByText('Exit conditions')).toBeVisible();
    expect(screen.getByText('No exit conditions.')).toBeVisible();
    expect(screen.queryByRole('list', { name: 'root exit conditions' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('tab', { name: 'Contents 1' }));
    fireEvent.click(screen.getByLabelText('root action 1: exit to root'));
    fireEvent.click(screen.getByRole('tab', { name: 'Conditions' }));
    expect(screen.getByText('Action conditions')).toBeVisible();
    expect(screen.getByRole('list', { name: 'root action 1: exit to root conditions' })).toHaveTextContent('Action ready');
    expect(screen.queryByText('Entry conditions')).not.toBeInTheDocument();
    expect(screen.queryByText('Exit conditions')).not.toBeInTheDocument();
});
