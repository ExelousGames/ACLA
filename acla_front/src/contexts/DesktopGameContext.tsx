import React, { createContext, ReactNode, useContext, useEffect, useState } from 'react';
import { useEnvironment } from './EnvironmentContext';

export type DesktopGame = 'ac' | 'acc' | 'iracing';
export type DesktopGameDetectionStatus =
    | 'checking'
    | 'detected'
    | 'not-detected'
    | 'unsupported'
    | 'error';

export interface DesktopGameContextValue {
    detectedGame: DesktopGame | null;
    detectionStatus: DesktopGameDetectionStatus;
    error: string | null;
}

export const DESKTOP_GAME_POLL_INTERVAL_MS = 2000;

const DesktopGameContext = createContext<DesktopGameContextValue | undefined>(undefined);

const unsupportedState: DesktopGameContextValue = {
    detectedGame: null,
    detectionStatus: 'unsupported',
    error: null,
};

const getErrorMessage = (error: unknown): string => {
    if (error instanceof Error) {
        return error.message;
    }
    return String(error);
};

const DesktopGameProvider = ({ children }: { children: ReactNode }) => {
    const environment = useEnvironment();
    const [detection, setDetection] = useState<DesktopGameContextValue>(() => (
        environment === 'electron'
            ? { detectedGame: null, detectionStatus: 'checking', error: null }
            : unsupportedState
    ));

    useEffect(() => {
        if (environment !== 'electron') {
            setDetection(unsupportedState);
            return;
        }

        const detectDesktopGame = window.electronAPI?.detectDesktopGame;
        if (typeof detectDesktopGame !== 'function') {
            setDetection({
                detectedGame: null,
                detectionStatus: 'error',
                error: 'Desktop game detection is unavailable.',
            });
            return;
        }

        let disposed = false;
        let requestInFlight = false;
        let intervalId: number | null = null;

        const poll = async () => {
            if (disposed || requestInFlight) {
                return;
            }

            requestInFlight = true;
            try {
                const result = await detectDesktopGame();
                if (disposed) {
                    return;
                }

                if (!result.supported) {
                    setDetection(unsupportedState);
                    if (intervalId !== null) {
                        window.clearInterval(intervalId);
                        intervalId = null;
                    }
                    return;
                }

                setDetection({
                    detectedGame: result.detectedGame,
                    detectionStatus: result.detectedGame ? 'detected' : 'not-detected',
                    error: null,
                });
            } catch (error) {
                if (!disposed) {
                    setDetection({
                        detectedGame: null,
                        detectionStatus: 'error',
                        error: getErrorMessage(error),
                    });
                }
            } finally {
                requestInFlight = false;
            }
        };

        void Promise.resolve().then(poll);
        intervalId = window.setInterval(() => {
            void poll();
        }, DESKTOP_GAME_POLL_INTERVAL_MS);

        return () => {
            disposed = true;
            if (intervalId !== null) {
                window.clearInterval(intervalId);
            }
        };
    }, [environment]);

    return (
        <DesktopGameContext.Provider value={detection}>
            {children}
        </DesktopGameContext.Provider>
    );
};

export default DesktopGameProvider;

export const useDesktopGame = (): DesktopGameContextValue => {
    const context = useContext(DesktopGameContext);
    if (!context) {
        throw new Error('useDesktopGame must be used within a DesktopGameProvider');
    }
    return context;
};
