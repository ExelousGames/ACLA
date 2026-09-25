import React from 'react';
import { audioManager, type PlaybackHandle, type PlaybackOutcome } from 'services/audio';
import type { TtsPack } from './tts-service';

export interface TtsHandle {
    play(): void;
    stop(): void;
}

export interface TtsProps {
    pack: TtsPack;
    priority?: number;
    volume?: number;
    onEnded?: () => void;
    onError?: (error: Error) => void;
    onComplete?: (outcome: PlaybackOutcome) => void;
}

/** The owner retains the pack and decides when an explicit replay is appropriate. */
export const Tts = React.forwardRef<TtsHandle, TtsProps>((props, ref) => {
    const playbackRef = React.useRef<PlaybackHandle | null>(null);
    const propsRef = React.useRef(props);
    propsRef.current = props;
    const stop = React.useCallback(() => {
        const playback = playbackRef.current;
        playbackRef.current = null;
        playback?.stop();
    }, []);

    React.useEffect(() => stop, [props.pack.audioDataUrl, stop]);
    React.useImperativeHandle(ref, () => ({
        play: () => {
            stop();
            const { pack, priority = 50, volume } = propsRef.current;
            const playback = audioManager.play({
                url: pack.audioDataUrl, type: 'voice', priority, volume,
                onComplete: (outcome) => {
                    if (playbackRef.current !== playback) return;
                    playbackRef.current = null;
                    if (outcome.status === 'completed') propsRef.current.onEnded?.();
                    if (outcome.status === 'failed') propsRef.current.onError?.(outcome.error);
                    propsRef.current.onComplete?.(outcome);
                },
            });
            playbackRef.current = playback;
        },
        stop,
    }), [stop]);

    return null;
});

Tts.displayName = 'Tts';
