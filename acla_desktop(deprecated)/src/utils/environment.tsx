export type Environment = 'electron' | 'web';

export const detectEnvironment = (): Environment => {
    // Check for Electron environment
    if (typeof window !== 'undefined' && (window as any).process?.type === 'renderer') {
        return 'electron';
    }
    if (typeof process !== 'undefined' && process.versions?.electron) {
        return 'electron';
    }
    if (typeof navigator === 'object' && navigator.userAgent?.toLowerCase().includes('electron')) {
        return 'electron';
    }

    // Default to web
    return 'web';
};