import { useContext, useState } from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { Theme } from '@radix-ui/themes';
import apiService from 'services/api.service';
import { RacingSessionDetailedInfoDto } from 'data/live-analysis/live-analysis-type';
import { AnalysisContext } from '../analysis-context';
import SessionList from './session-list';

// react-scripts' Jest resolver does not support Radix's exported subpath.
jest.mock('radix-ui/internal', () => jest.requireActual('radix-ui/dist/internal.js'), { virtual: true });

jest.mock('services/api.service', () => ({
    __esModule: true,
    default: { post: jest.fn(), delete: jest.fn() },
}));
jest.mock('hooks/AuthProvider', () => ({
    useAuth: () => ({ userProfile: { id: 'user-1' } }),
}));

const mockPost = apiService.post as jest.Mock;
const mockDelete = apiService.delete as jest.Mock;
const firstSession: RacingSessionDetailedInfoDto = {
    SessionId: 'session-1', session_name: 'Sunday Race', map: 'Monza', car: 'GT3',
    user_id: 'user-1', points: [], data: [],
};

const Harness = ({ selected = firstSession, map = 'Monza' }: {
    selected?: RacingSessionDetailedInfoDto;
    map?: string;
}) => {
    const defaults = useContext(AnalysisContext);
    const [sessionSelected, setSession] = useState<RacingSessionDetailedInfoDto | null>(selected);
    return (
        <Theme>
            <AnalysisContext.Provider value={{ ...defaults, mapSelected: map, sessionSelected, setSession }}>
                <SessionList />
                <output aria-label="Selected session">{sessionSelected?.SessionId ?? 'none'}</output>
            </AnalysisContext.Provider>
        </Theme>
    );
};

beforeEach(() => {
    mockPost.mockReset().mockResolvedValue({ data: { list: [
        { sessionId: 'session-1', name: 'Sunday Race' },
        { sessionId: 'session-2', name: 'Practice' },
    ] } });
    mockDelete.mockReset().mockResolvedValue({ status: 200 });
});

const openDeleteDialog = async () => {
    fireEvent.click(await screen.findByRole('button', { name: 'Delete session Sunday Race' }));
    return screen.getByRole('alertdialog');
};

it('requires confirmation and leaves the session and selection unchanged on cancel', async () => {
    render(<Harness selected={{ ...firstSession, SessionId: 'session-2' }} />);
    const dialog = await openDeleteDialog();

    expect(within(dialog).getByText(/Delete “Sunday Race” and its recorded telemetry/)).toBeInTheDocument();
    expect(mockDelete).not.toHaveBeenCalled();
    expect(screen.getByLabelText('Selected session')).toHaveTextContent('session-2');
    fireEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }));

    expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Delete session Sunday Race' })).toBeInTheDocument();
    expect(mockDelete).not.toHaveBeenCalled();
});

it('removes the session after successful deletion and clears its selection', async () => {
    let resolveDelete!: (value: unknown) => void;
    mockDelete.mockReturnValue(new Promise((resolve) => { resolveDelete = resolve; }));
    render(<Harness />);
    const dialog = await openDeleteDialog();
    fireEvent.click(within(dialog).getByRole('button', { name: 'Delete session' }));

    expect(mockDelete).toHaveBeenCalledWith('racing-session/session-1');
    expect(within(dialog).getByRole('button', { name: 'Deleting...' })).toBeDisabled();
    expect(within(dialog).getByRole('button', { name: 'Cancel' })).toBeDisabled();
    expect(screen.getByLabelText('Selected session')).toHaveTextContent('session-1');
    fireEvent.keyDown(dialog, { key: 'Escape' });
    expect(screen.getByRole('alertdialog')).toBeInTheDocument();
    fireEvent.click(within(dialog).getByRole('button', { name: 'Deleting...' }));
    expect(mockDelete).toHaveBeenCalledTimes(1);

    await act(async () => { resolveDelete({ status: 200 }); });

    expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Delete session Sunday Race' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Delete session Practice' })).toBeInTheDocument();
    expect(screen.getByLabelText('Selected session')).toHaveTextContent('none');
});

it('preserves a different selected session when deleting', async () => {
    render(<Harness selected={{ ...firstSession, SessionId: 'session-2' }} />);
    const dialog = await openDeleteDialog();
    fireEvent.click(within(dialog).getByRole('button', { name: 'Delete session' }));

    await waitFor(() => expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument());
    expect(screen.getByLabelText('Selected session')).toHaveTextContent('session-2');
});

it('keeps a failed deletion open with an error and supports retry', async () => {
    mockDelete.mockRejectedValueOnce(new Error('Network unavailable'));
    render(<Harness />);
    const dialog = await openDeleteDialog();
    fireEvent.click(within(dialog).getByRole('button', { name: 'Delete session' }));

    expect(await within(dialog).findByRole('alert')).toHaveTextContent('Could not delete this session. Please try again.');
    expect(screen.getByLabelText('Selected session')).toHaveTextContent('session-1');
    expect(within(dialog).getByRole('button', { name: 'Delete session' })).toBeEnabled();
    fireEvent.click(within(dialog).getByRole('button', { name: 'Delete session' }));

    await waitFor(() => expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument());
    expect(mockDelete).toHaveBeenCalledTimes(2);
    expect(screen.queryByRole('button', { name: 'Delete session Sunday Race' })).not.toBeInTheDocument();
});

it('shows an empty state after deleting the last session', async () => {
    mockPost.mockResolvedValue({ data: { list: [{ sessionId: 'session-1', name: 'Sunday Race' }] } });
    render(<Harness />);
    const dialog = await openDeleteDialog();
    fireEvent.click(within(dialog).getByRole('button', { name: 'Delete session' }));

    expect(await screen.findByText('No recorded sessions for this track.')).toBeInTheDocument();
});

it('removes a stale entry when the session has already been deleted', async () => {
    mockDelete.mockRejectedValue({ status: 404, data: { message: 'Session not found' } });
    render(<Harness />);
    const dialog = await openDeleteDialog();
    fireEvent.click(within(dialog).getByRole('button', { name: 'Delete session' }));

    await waitFor(() => expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument());
    expect(screen.queryByRole('button', { name: 'Delete session Sunday Race' })).not.toBeInTheDocument();
    expect(screen.getByLabelText('Selected session')).toHaveTextContent('none');
});

it('ignores an old list request after switching tracks', async () => {
    let resolveList!: (value: unknown) => void;
    mockPost.mockReturnValueOnce(new Promise((resolve) => { resolveList = resolve; }));
    mockPost.mockResolvedValueOnce({ data: { list: [{ sessionId: 'session-2', name: 'Practice' }] } });
    const view = render(<Harness />);
    view.rerender(<Harness map="Spa" />);
    await screen.findByRole('button', { name: 'Delete session Practice' });

    await act(async () => {
        resolveList({ data: { list: [{ sessionId: 'session-1', name: 'Sunday Race' }] } });
    });

    expect(screen.queryByRole('button', { name: 'Delete session Sunday Race' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Delete session Practice' })).toBeInTheDocument();
});
