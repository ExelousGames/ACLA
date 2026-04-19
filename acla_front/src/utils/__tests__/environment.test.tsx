import { detectEnvironment, Environment } from '../environment';

describe('detectEnvironment', () => {
    const originalProcess = window.process;
    const originalNavigator = window.navigator;

    afterEach(() => {
        // Restore originals
        Object.defineProperty(window, 'process', { value: originalProcess, writable: true });
    });

    it('should return "web" by default', () => {
        expect(detectEnvironment()).toBe('web');
    });

    it('should return "electron" when window.process.type is "renderer"', () => {
        Object.defineProperty(window, 'process', {
            value: { type: 'renderer' },
            writable: true,
        });
        expect(detectEnvironment()).toBe('electron');
    });

    it('should return "electron" when navigator userAgent contains "electron"', () => {
        Object.defineProperty(window, 'navigator', {
            value: { userAgent: 'Mozilla/5.0 Electron/28.0.0' },
            writable: true,
            configurable: true,
        });
        expect(detectEnvironment()).toBe('electron');
        // Restore
        Object.defineProperty(window, 'navigator', {
            value: originalNavigator,
            writable: true,
            configurable: true,
        });
    });
});
