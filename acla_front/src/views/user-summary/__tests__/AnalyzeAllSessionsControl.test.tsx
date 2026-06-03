import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import AnalyzeAllSessionsControl from '../AnalyzeAllSessionsControl';
import apiService from 'services/api.service';

jest.mock('@radix-ui/themes', () => ({
    Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    Flex: ({ children, ...props }: any) => <div {...props}>{children}</div>,
    Spinner: () => <span>Loading</span>,
    Text: ({ children, ...props }: any) => <span {...props}>{children}</span>,
}));

jest.mock('@radix-ui/react-icons', () => ({
    UpdateIcon: () => <span>Update</span>,
}));

jest.mock('services/api.service', () => ({
    __esModule: true,
    default: {
        get: jest.fn(),
        post: jest.fn(),
    },
}));

const mockedApi = apiService as jest.Mocked<typeof apiService>;

describe('AnalyzeAllSessionsControl', () => {
    beforeEach(() => {
        jest.clearAllMocks();
    });

    it('queues analysis when clicked', async () => {
        mockedApi.get.mockResolvedValue({ data: null, status: 200 });
        mockedApi.post.mockResolvedValue({
            data: { id: 'job-1', status: 'queued', progress: { message: 'Queued' } },
            status: 201,
        } as any);

        render(<AnalyzeAllSessionsControl onCompleted={jest.fn()} />);

        await userEvent.click(await screen.findByRole('button', { name: /analyze all sessions/i }));

        expect(mockedApi.post).toHaveBeenCalledWith('/userinfo/summary/analyze-all');
        expect(await screen.findByText('Analysis queued')).toBeInTheDocument();
    });

    it('shows a duplicate active job message on conflict', async () => {
        mockedApi.get.mockResolvedValueOnce({ data: null, status: 200 } as any);
        mockedApi.get.mockResolvedValueOnce({
            data: { id: 'job-1', status: 'running', progress: { message: 'Analyzing sessions' } },
            status: 200,
        } as any);
        mockedApi.post.mockRejectedValue({ status: 409 });

        render(<AnalyzeAllSessionsControl onCompleted={jest.fn()} />);

        await waitFor(() => expect(mockedApi.get).toHaveBeenCalled());
        await userEvent.click(screen.getByRole('button', { name: /analyze all sessions/i }));

        expect(await screen.findByText('Analysis is already queued or running for this user')).toBeInTheDocument();
    });
});
