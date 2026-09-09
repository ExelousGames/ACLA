import React from 'react';
import type { TtsPack } from './tts-service';

export interface TtsHandle {
    play(): void;
    stop(): void;
}

export interface TtsProps {
    pack: TtsPack;
    onEnded?: () => void;
    onError?: (error: Error) => void;
}

/** Playback belongs to the component that owns the pack; synthesis never plays audio. */
export const Tts = React.forwardRef<TtsHandle, TtsProps>(({ pack, onEnded, onError }, ref) => {
    const audioRef = React.useRef<HTMLAudioElement>(null);
    const playbackRef = React.useRef(0);
    const errorCallbackRef = React.useRef(onError);
    errorCallbackRef.current = onError;

    React.useEffect(() => {
        const audio = audioRef.current;
        return () => {
            playbackRef.current += 1;
            audio?.pause();
        };
    }, [pack.audioDataUrl]);

    React.useImperativeHandle(ref, () => ({
        play: () => {
            const audio = audioRef.current;
            if (!audio) return;
            const playback = ++playbackRef.current;
            const reportError = (error: unknown) => {
                if (playback !== playbackRef.current) return;
                errorCallbackRef.current?.(error instanceof Error ? error : new Error(String(error)));
            };
            try {
                audio.currentTime = 0;
                void audio.play().catch(reportError);
            } catch (error) {
                reportError(error);
            }
        },
        stop: () => {
            playbackRef.current += 1;
            audioRef.current?.pause();
        },
    }), []);

    return <audio
        ref={audioRef}
        src={pack.audioDataUrl}
        preload="auto"
        onEnded={onEnded}
        onError={() => errorCallbackRef.current?.(new Error('The speech audio could not be played.'))}
    />;
});

Tts.displayName = 'Tts';
