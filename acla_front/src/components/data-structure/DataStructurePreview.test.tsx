import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import DataStructurePreview from './DataStructurePreview';

const expand = async (name: string, child: string) => {
    const summary = await screen.findByText(name, { selector: 'code' });
    fireEvent.click(summary);
    expect(await screen.findByText(child, { selector: 'code' })).toBeInTheDocument();
};

describe('data structure preview', () => {
    it('merges every record and array item into independently expandable nested structures', async () => {
        const readRecords = jest.fn(async (onChunk) => {
            onChunk([{ arbitrary_field: { entries: [{ first: 0 }, { later: false }] }, mixed: null, empty: [] }]);
            onChunk([{ arbitrary_field: { entries: [{ deep: { text: '' } }, 4] }, mixed: 'value', final_field: {} }]);
        });
        render(<DataStructurePreview readRecords={readRecords} />);

        expect(await screen.findByText(/Structure from 2 records/)).toBeInTheDocument();
        expect(screen.queryByText('arbitrary_field')).not.toBeInTheDocument();
        await expand('Record', 'arbitrary_field');
        const tree = screen.getByRole('region', { name: 'Recorded data structure' });
        expect(tree).toHaveTextContent('mixednull | string');
        expect(tree).toHaveTextContent('emptyarrayempty');
        expect(tree).toHaveTextContent('final_fieldobject0 fields');
        await expand('arbitrary_field', 'entries');
        await expand('entries', '[items]');
        expect(tree).toHaveTextContent('[items]number | object');
        await expand('[items]', 'first');
        expect(tree).toHaveTextContent('firstnumber');
        expect(tree).toHaveTextContent('laterboolean');
        await expand('deep', 'text');
        expect(tree).toHaveTextContent('textstring');

        fireEvent.click(screen.getByText('arbitrary_field'));
        await waitFor(() => expect(screen.queryByText('entries')).not.toBeInTheDocument());
        expect(screen.getByText('final_field')).toBeInTheDocument();
        expect(readRecords).toHaveBeenCalledTimes(1);
    });

    it('renders prototype-like property names as ordinary fields', async () => {
        render(<DataStructurePreview readRecords={async (onChunk) => {
            onChunk([JSON.parse('{"__proto__":{"constructor":false},"toString":0}')]);
        }} />);

        await expand('Record', '__proto__');
        await expand('__proto__', 'constructor');
        expect(screen.getByRole('region')).toHaveTextContent('constructorboolean');
        expect(screen.getByRole('region')).toHaveTextContent('toStringnumber');
    });

    it('shows read failures and allows retrying without stale partial results', async () => {
        const readRecords = jest.fn()
            .mockImplementationOnce(async (onChunk) => {
                onChunk([{ partial: true }]);
                throw new Error('File unavailable');
            })
            .mockImplementationOnce(async (onChunk) => { onChunk([{ complete: 1 }]); });
        render(<DataStructurePreview readRecords={readRecords} />);

        expect(await screen.findByRole('alert')).toHaveTextContent('File unavailable');
        fireEvent.click(screen.getByRole('button', { name: 'Retry reading fields' }));
        await expand('Record', 'complete');
        expect(screen.getByText('complete')).toBeInTheDocument();
        expect(screen.queryByText('partial')).not.toBeInTheDocument();
    });

    it('handles an empty recording', async () => {
        render(<DataStructurePreview readRecords={async () => undefined} />);
        expect(await screen.findByText('No recorded fields available.')).toBeInTheDocument();
    });

    it('stops a pending reader on its next chunk after closing the preview', async () => {
        let receiveChunk!: (records: unknown[]) => void;
        let finish!: () => void;
        const readRecords = jest.fn((onChunk) => {
            receiveChunk = onChunk;
            return new Promise<void>((resolve) => { finish = resolve; });
        });
        const view = render(<DataStructurePreview readRecords={readRecords} />);
        expect(screen.getByRole('status')).toHaveTextContent('Reading recorded fields');
        view.unmount();
        expect(() => receiveChunk([{ late: 1 }])).toThrow('Data structure read cancelled.');
        await act(async () => { finish(); });
    });
});
