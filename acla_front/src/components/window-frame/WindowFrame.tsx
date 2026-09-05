import { useEffect, useState } from 'react';
import './WindowFrame.css';

type WindowControlApi = {
    minimize: () => Promise<unknown>;
    toggleMaximize: () => Promise<{ isMaximized?: boolean } | void>;
    close: () => Promise<unknown>;
    isMaximized: () => Promise<boolean>;
    onMaximizedChange: (callback: (isMaximized: boolean) => void) => () => void;
};

const getWindowControls = (): WindowControlApi | undefined => (
    (window as unknown as { electronAPI?: { windowControls?: WindowControlApi } })
        .electronAPI?.windowControls
);

const MinimizeIcon = () => (
    <svg aria-hidden="true" viewBox="0 0 12 12">
        <path d="M2 6.5h8" />
    </svg>
);

const MaximizeIcon = ({ maximized }: { maximized: boolean }) => (
    <svg aria-hidden="true" viewBox="0 0 12 12">
        {maximized ? (
            <>
                <path d="M3.5 4.5h5v5h-5z" />
                <path d="M5 4.5V3h4v4H8.5" />
            </>
        ) : (
            <path d="M2.5 2.5h7v7h-7z" />
        )}
    </svg>
);

const CloseIcon = () => (
    <svg aria-hidden="true" viewBox="0 0 12 12">
        <path d="m3 3 6 6M9 3 3 9" />
    </svg>
);

const WindowFrame = () => {
    const controls = getWindowControls();
    const [isMaximized, setIsMaximized] = useState(false);

    useEffect(() => {
        if (!controls) return undefined;

        let mounted = true;
        void controls.isMaximized()
            .then((maximized) => {
                if (mounted) setIsMaximized(Boolean(maximized));
            })
            .catch(() => undefined);

        const unsubscribe = controls.onMaximizedChange((maximized) => {
            if (mounted) setIsMaximized(Boolean(maximized));
        });

        return () => {
            mounted = false;
            unsubscribe?.();
        };
    }, [controls]);

    if (!controls) return null;

    const runControl = (action: () => Promise<unknown>) => {
        void action().catch(() => undefined);
    };

    const toggleMaximize = () => {
        void controls.toggleMaximize()
            .then((result) => {
                if (result && typeof result.isMaximized === 'boolean') {
                    setIsMaximized(result.isMaximized);
                }
            })
            .catch(() => undefined);
    };

    return (
        <header className="window-frame" aria-label="Application window controls">
            <div className="window-frame__brand" aria-label="Kestrel Motorsport Analyst">
                <span className="window-frame__mark" aria-hidden="true">
                    <span className="window-frame__mark-wing" />
                    <span className="window-frame__mark-core" />
                </span>
                <span className="window-frame__brand-name">Kestrel</span>
                <span className="window-frame__brand-divider" aria-hidden="true" />
                <span className="window-frame__product">Motorsport Analyst</span>
            </div>

            <div className="window-frame__status" aria-hidden="true">
                <span className="window-frame__status-line" />
                <span className="window-frame__status-dot" />
                <span>Desktop telemetry suite</span>
                <span className="window-frame__status-line window-frame__status-line--right" />
            </div>

            <div className="window-frame__controls">
                <button
                    className="window-frame__control"
                    type="button"
                    aria-label="Minimize window"
                    title="Minimize"
                    onClick={() => runControl(controls.minimize)}
                >
                    <MinimizeIcon />
                </button>
                <button
                    className="window-frame__control"
                    type="button"
                    aria-label={isMaximized ? 'Restore window' : 'Maximize window'}
                    title={isMaximized ? 'Restore' : 'Maximize'}
                    onClick={toggleMaximize}
                >
                    <MaximizeIcon maximized={isMaximized} />
                </button>
                <button
                    className="window-frame__control window-frame__control--close"
                    type="button"
                    aria-label="Close window"
                    title="Close"
                    onClick={() => runControl(controls.close)}
                >
                    <CloseIcon />
                </button>
            </div>
        </header>
    );
};

export default WindowFrame;
