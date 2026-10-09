import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { Action, Closure, State, describeCondition } from './closure';
import ClosureTree from './ClosureTree';

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
        new Action('record telemetry', 'Work at root.', run), branch,
    ]));
    const view = render(<ClosureTree node={JSON.parse(JSON.stringify(state.snapshot({ ready: false })))} />);
    expect(screen.getByLabelText('root action 1: record telemetry')).toBeVisible();
    fireEvent.click(screen.getByLabelText('root action 1: record telemetry'));
    expect(screen.getByText('Work at root.')).toBeVisible();
    for (let depth = 1; depth <= 4; depth++) {
        const summary = screen.getByLabelText(`level ${depth} closure`);
        expect(summary).toBeVisible();
        fireEvent.click(summary);
    }
    const summary = screen.getByLabelText('level 4 action 1: record telemetry');
    expect(summary).toBeVisible();
    fireEvent.click(summary);
    const body = within(summary.closest('details')!);
    expect(body.getByText('Save a diagnostic sample.')).toBeVisible();
    expect(body.getByText('diagnostics')).toBeVisible();
    expect(body.getByText('0')).toBeVisible();
    expect(body.getByText('false')).toBeVisible();
    expect(body.getByText('—')).toBeVisible();
    const conditions = within(body.getByRole('list', { name: 'level 4 action 1: record telemetry conditions' }));
    fireEvent.click(conditions.getByLabelText('level 4 action 1: record telemetry conditions group 1'));
    expect(conditions.getByText('Recorder available')).toBeVisible();
    expect(conditions.getAllByText('Not met')).toHaveLength(2);
    expect(conditions.getByText('AND')).toBeVisible();
    expect(run).not.toHaveBeenCalled();
    // A data-only refresh retains the user's open branches and updates condition status.
    view.rerender(<ClosureTree node={JSON.parse(JSON.stringify(state.snapshot({ ready: true })))} />);
    expect(conditions.getByText('Recorder available')).toBeVisible();
    expect(conditions.getAllByText('Met')).toHaveLength(3);
});

it('displays independent action execution, current location and failure details', async () => {
    let reject!: (reason: Error) => void;
    const action = new Action<{}>('upload sample', 'Upload diagnostics.', () => new Promise<void>((_resolve, fail) => { reject = fail; }));
    const state = new State(new Closure<{}>('root', 'Root.', () => true, [action, new Action('later', 'Later work.', jest.fn())]));
    const pending = state.step({});
    const view = render(<ClosureTree node={state.snapshot({})} />);
    const summary = screen.getByLabelText('root action 1: upload sample');
    expect(within(summary).getByText('Running')).toBeVisible();
    expect(summary.closest('details')).toHaveAttribute('data-current', 'true');
    expect(within(screen.getByLabelText('root action 2: later')).getByText('Action · idle')).toBeVisible();
    fireEvent.click(summary);
    reject(new Error('Upload failed'));
    await expect(pending).rejects.toThrow('Upload failed');
    view.rerender(<ClosureTree node={state.snapshot({})} />);
    expect(within(summary).getByText('Failed')).toBeVisible();
    expect(screen.getByText('Upload failed')).toBeVisible();
});
