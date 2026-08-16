import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { EditorView } from '@codemirror/view';
import { diagnosticCount } from '@codemirror/lint';
import { JsonataQueryEditor, type JsonataQueryEditorProps } from './JsonataQueryEditor';

const defaultProps = (): JsonataQueryEditorProps => ({
    id: 'jsonata-query',
    value: 'elements',
    labels: ['MSP', 'Wheel lock'],
    isDirty: false,
    isEvaluating: false,
    diagnostic: null,
    diagnosticFocusRequest: 0,
    onChange: jest.fn(),
    onApply: jest.fn(),
    onReset: jest.fn(),
});

const getEditorView = (): EditorView => {
    const textbox = screen.getByRole('textbox', { name: 'Query expression' });
    const editor = EditorView.findFromDOM(textbox);
    if (!editor) throw new Error('Expected CodeMirror to be mounted.');
    return editor;
};

describe('JsonataQueryEditor', () => {
    afterEach(() => {
        jest.useRealTimers();
    });

    it('renders the focused CodeMirror authoring panel with line numbers', async () => {
        const props = defaultProps();
        const { container } = render(<JsonataQueryEditor {...props} />);

        expect(screen.getByRole('region', { name: 'Edit query' })).toBeInTheDocument();
        expect(await screen.findByRole('textbox', { name: 'Query expression' }))
            .toHaveAttribute('aria-busy', 'false');
        expect(container.querySelector('.cm-lineNumbers')).toBeInTheDocument();
        expect(getEditorView().state.doc.toString()).toBe('elements');
    });

    it('routes Ctrl+Enter through Apply and suppresses duplicate submissions', async () => {
        let resolveApply!: () => void;
        const pendingApply = new Promise<void>((resolve) => { resolveApply = resolve; });
        const props = defaultProps();
        props.onApply = jest.fn(() => pendingApply);
        const view = render(<JsonataQueryEditor {...props} />);
        const textbox = await screen.findByRole('textbox', { name: 'Query expression' });

        fireEvent.keyDown(textbox, { key: 'Enter', ctrlKey: true });
        fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
        expect(props.onApply).toHaveBeenCalledTimes(1);

        view.rerender(<JsonataQueryEditor {...props} isEvaluating />);
        expect(screen.getByRole('textbox', { name: 'Query expression' }))
            .toHaveAttribute('aria-busy', 'true');
        expect(screen.getByRole('status')).toHaveTextContent('Evaluating query…');
        expect(screen.getByRole('button', { name: 'Applying…' })).toBeDisabled();
        expect(screen.getByRole('button', { name: 'Reset' })).toBeDisabled();

        await act(async () => resolveApply());
    });

    it('keeps draft changes editable and focuses an actionable Apply diagnostic', async () => {
        const props = defaultProps();
        const view = render(<JsonataQueryEditor {...props} />);
        await screen.findByRole('textbox', { name: 'Query expression' });
        const editor = getEditorView();

        act(() => editor.dispatch({
            changes: { from: 0, to: editor.state.doc.length, insert: 'elements[id = ]' },
        }));
        expect((props.onChange as jest.Mock).mock.calls.at(-1)?.[0]).toBe('elements[id = ]');

        view.rerender(<JsonataQueryEditor
            {...props}
            value="elements[id = ]"
            diagnostic={{
                code: 'S0211',
                message: 'The symbol cannot be used here.',
                position: 15,
                token: ']',
            }}
            diagnosticFocusRequest={1}
        />);

        const diagnostic = await screen.findByTestId('active-page-query-error');
        expect(diagnostic).toHaveTextContent('S0211');
        expect(diagnostic).toHaveTextContent('position 15');
        await waitFor(() => expect(diagnostic).toHaveFocus());
        expect(diagnosticCount(getEditorView().state)).toBe(1);
        expect(getEditorView().state.doc.toString()).toBe('elements[id = ]');
    });

    it('synchronizes Reset/template values without echoing them as user edits', async () => {
        const props = defaultProps();
        const view = render(<JsonataQueryEditor {...props} />);
        await screen.findByRole('textbox', { name: 'Query expression' });

        view.rerender(<JsonataQueryEditor {...props} value="elements[id = 'one']" />);

        expect(getEditorView().state.doc.toString()).toBe("elements[id = 'one']");
        expect(props.onChange).not.toHaveBeenCalled();
    });

    it('cancels deferred editor work when unmounted', () => {
        jest.useFakeTimers();
        const consoleError = jest.spyOn(console, 'error').mockImplementation(() => undefined);
        const view = render(<JsonataQueryEditor {...defaultProps()} />);

        view.unmount();
        act(() => jest.runOnlyPendingTimers());

        expect(consoleError).not.toHaveBeenCalled();
        consoleError.mockRestore();
    });
});
