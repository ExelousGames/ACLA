import React from 'react';
import { act, render, screen } from '@testing-library/react';

jest.mock('contexts/EnvironmentContext', () => ({
    useEnvironment: jest.fn(),
}));

import DesktopGameProvider, {
    DESKTOP_GAME_POLL_INTERVAL_MS,
    useDesktopGame,
} from 'contexts/DesktopGameContext';
import { useEnvironment } from 'contexts/EnvironmentContext';

const mockedUseEnvironment = useEnvironment as jest.Mock;
const detectDesktopGame = jest.fn();

const TestConsumer = () => {
    const { detectedGame, detectionStatus, error } = useDesktopGame();
    return (
        <div>
            <span data-testid="game">{detectedGame ?? 'none'}</span>
            <span data-testid="status">{detectionStatus}</span>
            <span data-testid="error">{error ?? 'none'}</span>
        </div>
    );
};

const providerTree = (
    <DesktopGameProvider>
        <TestConsumer />
    </DesktopGameProvider>
);

const renderProvider = (strict = false) => render(
    strict ? <React.StrictMode>{providerTree}</React.StrictMode> : providerTree,
);

const flushPromises = async () => {
    await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
    });
};

describe('DesktopGameContext', () => {
    beforeEach(() => {
        jest.useFakeTimers();
        mockedUseEnvironment.mockReturnValue('electron');
        detectDesktopGame.mockReset();
        (window as any).electronAPI = { detectDesktopGame };
    });

    afterEach(() => {
        jest.useRealTimers();
        delete (window as any).electronAPI;
    });

    it('checks immediately and follows polling, error, and recovery transitions', async () => {
        detectDesktopGame
            .mockResolvedValueOnce({ supported: true, detectedGame: null })
            .mockResolvedValueOnce({ supported: true, detectedGame: 'acc' })
            .mockRejectedValueOnce(new Error('tasklist failed'))
            .mockResolvedValueOnce({ supported: true, detectedGame: 'iracing' });

        renderProvider();

        expect(screen.getByTestId('status')).toHaveTextContent('checking');
        await flushPromises();
        expect(detectDesktopGame).toHaveBeenCalledTimes(1);
        expect(screen.getByTestId('status')).toHaveTextContent('not-detected');

        await act(async () => {
            jest.advanceTimersByTime(DESKTOP_GAME_POLL_INTERVAL_MS);
        });
        await flushPromises();
        expect(screen.getByTestId('game')).toHaveTextContent('acc');
        expect(screen.getByTestId('status')).toHaveTextContent('detected');

        await act(async () => {
            jest.advanceTimersByTime(DESKTOP_GAME_POLL_INTERVAL_MS);
        });
        await flushPromises();
        expect(screen.getByTestId('game')).toHaveTextContent('none');
        expect(screen.getByTestId('status')).toHaveTextContent('error');
        expect(screen.getByTestId('error')).toHaveTextContent('tasklist failed');

        await act(async () => {
            jest.advanceTimersByTime(DESKTOP_GAME_POLL_INTERVAL_MS);
        });
        await flushPromises();
        expect(screen.getByTestId('game')).toHaveTextContent('iracing');
        expect(screen.getByTestId('status')).toHaveTextContent('detected');
        expect(screen.getByTestId('error')).toHaveTextContent('none');
    });

    it('does not overlap slow detection requests', async () => {
        let resolveFirstRequest!: (value: { supported: boolean; detectedGame: null }) => void;
        detectDesktopGame
            .mockImplementationOnce(() => new Promise((resolve) => {
                resolveFirstRequest = resolve;
            }))
            .mockResolvedValue({ supported: true, detectedGame: null });

        renderProvider(true);
        await flushPromises();
        act(() => {
            jest.advanceTimersByTime(DESKTOP_GAME_POLL_INTERVAL_MS * 3);
        });
        expect(detectDesktopGame).toHaveBeenCalledTimes(1);

        resolveFirstRequest({ supported: true, detectedGame: null });
        await flushPromises();
        await act(async () => {
            jest.advanceTimersByTime(DESKTOP_GAME_POLL_INTERVAL_MS);
        });
        expect(detectDesktopGame).toHaveBeenCalledTimes(2);
    });

    it('reports web and non-Windows desktop environments as unsupported', async () => {
        mockedUseEnvironment.mockReturnValue('web');
        const webView = renderProvider();
        expect(screen.getByTestId('status')).toHaveTextContent('unsupported');
        expect(detectDesktopGame).not.toHaveBeenCalled();
        webView.unmount();

        mockedUseEnvironment.mockReturnValue('electron');
        detectDesktopGame.mockResolvedValue({ supported: false, detectedGame: null });
        renderProvider();
        await flushPromises();

        expect(screen.getByTestId('status')).toHaveTextContent('unsupported');
        act(() => {
            jest.advanceTimersByTime(DESKTOP_GAME_POLL_INTERVAL_MS * 2);
        });
        expect(detectDesktopGame).toHaveBeenCalledTimes(1);
    });

    it('cleans up its polling interval on unmount', async () => {
        const clearIntervalSpy = jest.spyOn(window, 'clearInterval');
        detectDesktopGame.mockResolvedValue({ supported: true, detectedGame: null });
        const view = renderProvider();
        await flushPromises();

        view.unmount();
        expect(clearIntervalSpy).toHaveBeenCalled();
        act(() => {
            jest.advanceTimersByTime(DESKTOP_GAME_POLL_INTERVAL_MS * 2);
        });
        expect(detectDesktopGame).toHaveBeenCalledTimes(1);
        clearIntervalSpy.mockRestore();
    });
});
