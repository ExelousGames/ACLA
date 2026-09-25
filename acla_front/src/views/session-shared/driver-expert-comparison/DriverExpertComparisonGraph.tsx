import React from 'react';
import { Tts, type TtsHandle, type TtsPack } from 'components/tts';
import {
    DriverExpertComparisonPresentation,
    type DriverExpertComparisonPresentationProps,
} from './DriverExpertComparisonPresentation';

// The overlay imports the presentation directly. Local narration stays here.
export * from './DriverExpertComparisonPresentation';
export interface DriverExpertComparisonGraphProps extends DriverExpertComparisonPresentationProps {
    voice?: TtsPack;
    audioPriority?: number;
    audioVolume?: number;
}

export const DriverExpertComparisonGraph: React.FC<DriverExpertComparisonGraphProps> = ({
    voice, audioPriority, audioVolume, onReplayStarted, onReplayComplete, onReplayStopped, ...props
}) => {
    const tts = React.useRef<TtsHandle>(null);
    const callbacks = React.useRef({ onReplayStarted, onReplayComplete });
    callbacks.current = { onReplayStarted, onReplayComplete };
    const run = React.useMemo(() => ({
        visualSettled: false, audioSettled: true, completed: false, active: true,
    }), [props.data.samples, voice]);
    const currentRun = React.useRef(run);
    const runKey = React.useRef(0);
    if (currentRun.current !== run) runKey.current += 1;
    currentRun.current = run;
    const [failedRun, setFailedRun] = React.useState<typeof run>();
    const [speakingRun, setSpeakingRun] = React.useState<typeof run>();
    React.useEffect(() => {
        run.active = true;
        const playback = tts.current;
        return () => { run.active = false; playback?.stop(); };
    }, [run]);

    const completeIfSettled = () => {
        if (currentRun.current !== run || !run.active || run.completed
            || !run.visualSettled || !run.audioSettled) return;
        run.completed = true;
        callbacks.current.onReplayComplete?.();
    };

    return <>
        {voice && <Tts key={runKey.current} ref={tts} pack={voice} priority={audioPriority} volume={audioVolume}
            onComplete={(outcome) => {
                if (currentRun.current !== run || !run.active) return;
                run.audioSettled = true;
                setSpeakingRun(undefined);
                setFailedRun(outcome.status === 'failed' ? run : undefined);
                completeIfSettled();
            }} />}
        {failedRun === run && <span role="status">Narration unavailable</span>}
        <DriverExpertComparisonPresentation key={runKey.current} {...props}
            completedStatusLabel={speakingRun === run ? 'Finishing narration' : props.completedStatusLabel}
            onReplayStarted={() => {
                run.completed = run.visualSettled = false;
                run.audioSettled = !voice;
                setSpeakingRun(voice ? run : undefined);
                setFailedRun(undefined);
                tts.current?.play();
                callbacks.current.onReplayStarted?.();
            }}
            onReplayStopped={() => { tts.current?.stop(); onReplayStopped?.(); }}
            onReplayComplete={() => {
                run.visualSettled = true;
                completeIfSettled();
            }} />
    </>;
};

export default DriverExpertComparisonGraph;
