import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import WindowFrame from './WindowFrame';

describe('WindowFrame', () => {
    let onMaximizedChange: ((maximized: boolean) => void) | undefined;
    const unsubscribe = jest.fn();
    const controls = {
        minimize: jest.fn(),
        toggleMaximize: jest.fn(),
        close: jest.fn(),
        isMaximized: jest.fn(),
        onMaximizedChange: jest.fn(),
    };

    beforeEach(() => {
        jest.clearAllMocks();
        onMaximizedChange = undefined;
        controls.minimize.mockResolvedValue({ success: true });
        controls.toggleMaximize.mockResolvedValue({ success: true, isMaximized: true });
        controls.close.mockResolvedValue({ success: true });
        controls.isMaximized.mockResolvedValue(false);
        controls.onMaximizedChange.mockImplementation((callback: (maximized: boolean) => void) => {
            onMaximizedChange = callback;
            return unsubscribe;
        });
        Object.defineProperty(window, 'electronAPI', {
            configurable: true,
            value: { windowControls: controls },
        });
    });

    afterEach(() => {
        delete (window as unknown as { electronAPI?: unknown }).electronAPI;
    });

    it('connects the custom buttons to Electron and reflects maximized state', async () => {
        const { unmount } = render(<WindowFrame />);

        await waitFor(() => expect(controls.isMaximized).toHaveBeenCalledTimes(1));

        fireEvent.click(screen.getByRole('button', { name: 'Minimize window' }));
        fireEvent.click(screen.getByRole('button', { name: 'Maximize window' }));
        fireEvent.click(screen.getByRole('button', { name: 'Close window' }));

        expect(controls.minimize).toHaveBeenCalledTimes(1);
        expect(controls.toggleMaximize).toHaveBeenCalledTimes(1);
        expect(controls.close).toHaveBeenCalledTimes(1);

        await waitFor(() => expect(screen.getByRole('button', { name: 'Restore window' })).toBeInTheDocument());

        act(() => onMaximizedChange?.(false));
        expect(screen.getByRole('button', { name: 'Maximize window' })).toBeInTheDocument();

        unmount();
        expect(unsubscribe).toHaveBeenCalledTimes(1);
    });
});
