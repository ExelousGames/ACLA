import React, { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import {
    NamedOperationComponentHandle,
    OPERATION_COMPONENT_NAMES,
    useOptionalOperationComponentRefDirectory,
    useRegisterOperationComponentRef,
} from 'contexts/OperationComponentRefContext';
import type { LiveSessionHandle } from '../LiveSessionView';
import type { TtsPack } from 'components/tts';
import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import { audioManager, type PlaybackHandle } from 'services/audio';
import { PHRASE_DEFINITIONS, PhraseEngine, PhraseSnapshot, type PhraseEvent } from './phrase-engine';
import { LivePhraseOverlay } from './LivePhraseOverlay';
import { copyPhraseMap, copyPhraseReference, MAX_DRIVING_CAUTION, REFERENCE_DRIVING_AGGRESSION } from './phrase-driving-reference';
import { createPhraseMapContext } from './phrase-map-context';
import type { ReferenceCornerBounds } from './phrase-reference-guidance';
import ReferenceRangeMap from './graphs/ReferenceRangeMap';
import ClosureTree from './ClosureTree';
import { Phrase, PHRASE_SILENCE_LEVEL, type PhraseInstance } from './phrase';
import { getPhraseSpeechInstances, loadAllPhraseSpeech } from './phrase-speech';
import './LivePhrases.css';

// Prepare the time-sensitive, short cues first.
const speechCatalog = getPhraseSpeechInstances();
const eventInstance = (event: PhraseEvent): PhraseInstance => event.instance
    ?? speechCatalog.find((instance) => instance.id.startsWith(`${event.ruleId}:say:`) && instance.speech.text === event.sentence)
    ?? new Phrase(event.ruleId, event.sentence).createInstance();

export interface LivePhrasesHandle extends NamedOperationComponentHandle {
    getSnapshot(): PhraseSnapshot;
    getPhraseCatalog(): typeof PHRASE_DEFINITIONS;
}

const LivePhrases = forwardRef<LivePhrasesHandle, { name: string }>(({ name }, forwardedRef) => {
    const directory = useOptionalOperationComponentRefDirectory();
    const root = directory?.findComponentRef<LiveSessionHandle>(OPERATION_COMPONENT_NAMES.LIVE_SESSION)?.current ?? null;
    const [enabled, setEnabled] = useState(false);
    const [drivingAggression, setDrivingAggression] = useState(REFERENCE_DRIVING_AGGRESSION);
    const [rangeBaseline, setRangeBaseline] = useState<TopLapReferenceEntry | null>(null);
    const [rangeCorners, setRangeCorners] = useState<readonly ReferenceCornerBounds[]>([]);
    const rangeReference = useMemo(() => copyPhraseReference(rangeBaseline, drivingAggression, rangeCorners),
        [rangeBaseline, drivingAggression, rangeCorners]);
    const aggressionRef = useRef(drivingAggression);
    aggressionRef.current = drivingAggression;
    const updateAggression = useRef<((value: number) => void) | null>(null);
    const [snapshot, setSnapshot] = useState(() => new PhraseEngine().evaluate(Date.now()));
    const [overlayError, setOverlayError] = useState<string | null>(null);
    const [voiceError, setVoiceError] = useState<string | null>(null);
    const [speechStatus, setSpeechStatus] = useState<'Disabled' | 'Preparing' | 'Ready' | 'Incomplete'>('Disabled');
    const [speechLevel, setSpeechLevel] = useState(0);
    const voicePacks = useRef(new Map<string, TtsPack>());
    const snapshotRef = useRef(snapshot);
    snapshotRef.current = snapshot;

    // Keep the range preview available while detection and speech are disabled.
    useEffect(() => {
        const update = () => {
            setRangeBaseline(copyPhraseReference(root?.getLiveReferenceLap() ?? null));
            setRangeCorners(createPhraseMapContext(copyPhraseMap(root?.getLiveCircuitMap() ?? null)).cornerBounds);
        };
        const unsubscribe = root?.subscribeLiveCircuitMap(update);
        update();
        return unsubscribe;
    }, [root]);

    useEffect(() => {
        const engine = new PhraseEngine();
        // Prepared WAVs contain only the first through last audible frames.
        const updateSpeechDurations = () => engine.setSpeechDurations(new Map(Array.from(voicePacks.current,
            ([key, pack]) => [key, pack.durationMs])));
        updateSpeechDurations();
        engine.setDrivingAggression(aggressionRef.current, Date.now());
        setOverlayError(null);
        setVoiceError(null);
        setSpeechLevel(0);
        if (!enabled) {
            setSpeechStatus('Disabled');
            setSnapshot(engine.evaluate(Date.now()));
            return;
        }
        setSpeechStatus('Preparing');
        const overlay = directory ? new LivePhraseOverlay(directory, name, (error) => {
            setOverlayError(error instanceof Error ? error.message : String(error));
        }) : null;
        const controller = new AbortController();
        let playback: PlaybackHandle | null = null;
        let pendingVoice: PhraseEvent | null = null;
        let pendingInstance: PhraseInstance | null = null;
        let speechPreparationFinished = false;
        let lastVoiceEventId = 0;
        const discardPendingVoice = () => {
            const event = pendingVoice;
            pendingVoice = null;
            pendingInstance = null;
            if (event?.speech) engine.speechFinished(event.id, Date.now(), true);
        };
        const playReadyVoice = () => {
            if (controller.signal.aborted || !pendingVoice || !pendingInstance) return;
            const pack = voicePacks.current.get(pendingInstance.cacheKey);
            if (!pack) {
                if (speechPreparationFinished) discardPendingVoice();
                return;
            }
            // Consume the event even if live chat rejects or interrupts its audio.
            const event = pendingVoice;
            pendingVoice = null;
            pendingInstance = null;
            let started = false;
            playback = audioManager.play({
                url: pack.audioDataUrl, type: 'voice', priority: 25, // Live chat defaults to 50.
                onLevel: (level, positionMs) => {
                    if (controller.signal.aborted) return;
                    setSpeechLevel(level);
                    const bounds = pack.speechBounds;
                    if (!started && (bounds ? positionMs >= bounds.startMs
                        : bounds !== null && level > PHRASE_SILENCE_LEVEL)) {
                        started = true;
                        publish(engine.speechStarted(event.id, Date.now()));
                    }
                },
                onComplete: (outcome) => {
                    if (controller.signal.aborted) return;
                    setSpeechLevel(0);
                    if (outcome.status === 'failed') setVoiceError(outcome.error.message);
                    if (event.speech) publish(engine.speechFinished(event.id, Date.now(), outcome.status !== 'completed'));
                },
            });
        };
        // All variants are requested here; telemetry and playback never request speech.
        void loadAllPhraseSpeech(speechCatalog, voicePacks.current, controller.signal, () => {
            updateSpeechDurations();
            playReadyVoice();
        }).then(({ complete, errors }) => {
            if (controller.signal.aborted) return;
            speechPreparationFinished = true;
            setSpeechStatus(complete ? 'Ready' : 'Incomplete');
            if (errors.length) setVoiceError(errors.map((error) => error instanceof Error ? error.message : String(error)).join('; '));
            playReadyVoice();
        }).catch((error) => {
            if (!controller.signal.aborted) {
                setSpeechStatus('Incomplete');
                setVoiceError(error instanceof Error ? error.message : String(error));
            }
        });
        const dispose = () => {
            updateAggression.current = null;
            controller.abort();
            pendingVoice = null;
            pendingInstance = null;
            playback?.stop();
            overlay?.dispose();
        };
        const publish = (next: PhraseSnapshot) => {
            setSnapshot((previous) => (
                JSON.stringify(previous) === JSON.stringify(next) ? previous : next
            ));
            overlay?.update(next, root?.getLiveCircuitMap() ?? null, root?.getLiveReferenceLap() ?? null);
            if (next.events.length === 0) {
                lastVoiceEventId = 0;
                discardPendingVoice();
                pendingInstance = null;
                playback?.stop();
                playback = null;
            }
            for (const event of next.events) {
                if (pendingVoice?.id === event.id) pendingVoice = event;
                if (event.id <= lastVoiceEventId) continue;
                lastVoiceEventId = event.id;
                pendingVoice = event;
                pendingInstance = eventInstance(event);
                // Entry conditions already passed in the engine. Playback only needs
                // the prepared clip; the shared audio manager handles interruption.
                playReadyVoice();
            }
        };
        updateAggression.current = (value) => publish(engine.setDrivingAggression(value, Date.now()));
        publish(engine.evaluate(Date.now()));
        if (!root) return dispose;
        const updateVision = () => publish(engine.receiveVision(root.getTrackVisionDetection(), Date.now()));
        const unsubscribeVision = root.subscribeTrackVision(updateVision);
        updateVision();
        const updateMap = () => publish(engine.receiveMap(root.getLiveCircuitMap(), Date.now(), root.getLiveReferenceLap()));
        const unsubscribeMap = root.subscribeLiveCircuitMap(updateMap);
        updateMap();
        const unsubscribeTelemetry = root.subscribeTelemetry((event) => {
            publish(engine.receiveTelemetry(event, Date.now(), root.getLiveCornerSummaries()));
        });
        // Expire inputs even if the simulator or screen capture stops sending events.
        const timer = setInterval(() => publish(engine.evaluate(Date.now())), 250);
        return () => {
            clearInterval(timer);
            unsubscribeVision();
            unsubscribeMap();
            unsubscribeTelemetry();
            dispose();
        };
    }, [directory, enabled, name, root]);

    const handle = useMemo<LivePhrasesHandle>(() => ({
        getComponentName: () => name,
        getSnapshot: () => snapshotRef.current,
        getPhraseCatalog: () => PHRASE_DEFINITIONS,
    }), [name]);
    useImperativeHandle(forwardedRef, () => handle, [handle]);
    const registeredHandle = useRef(handle);
    registeredHandle.current = handle;
    useRegisterOperationComponentRef(registeredHandle);

    return (
        <section className="live-phrases" aria-label="Live phrases">
            <header>
                <div className="live-phrases__heading">
                    <h2>Live phrases <span>Closures</span></h2>
                    <button type="button" className="live-phrases__toggle" onClick={() => setEnabled((current) => !current)}>
                        {enabled ? 'Disable' : 'Enable'} detection
                    </button>
                </div>
            </header>
            <div className="live-phrases__aggression">
                <label>
                    <span>Driving aggression</span>
                    <input type="range" min="0" max={MAX_DRIVING_CAUTION} step="1" value={drivingAggression}
                        aria-valuetext={drivingAggression === 0 ? '0% braking adjustment (Reference)' : drivingAggression === 100
                            ? '100% braking adjustment (Direction guide)' : `${drivingAggression}% braking adjustment`}
                        onChange={(event) => {
                            const value = Number(event.target.value);
                            setDrivingAggression(value);
                            updateAggression.current?.(value);
                        }} />
                </label>
                <div className="live-phrases__aggression-scale"><span>0% · Reference</span><span>{drivingAggression}% braking adjustment</span><span>100% · Direction guide</span></div>
                <ReferenceRangeMap reference={rangeReference} corners={rangeCorners} />
            </div>
            {overlayError && <p role="alert">Unable to display live phrase in the overlay: {overlayError}</p>}
            {voiceError && <p role="alert">Unable to prepare or play live phrase speech: {voiceError}</p>}
            <div className="live-phrases__sources" role="status">
                <span data-ready={enabled}>Detection: {enabled ? 'Enabled' : 'Disabled'}</span>
                <span data-ready={speechStatus === 'Ready'}>Speech: {speechStatus}</span>
                <meter aria-label="Speech level" min={0} max={1} value={speechLevel} />
                <span data-ready={snapshot.telemetryReady}>Telemetry: {snapshot.telemetryReady ? 'Live' : 'Waiting for live data'}</span>
                <span data-ready={snapshot.mapReady}>Live Map: {snapshot.mapReady ? 'Ready' : 'Waiting for a tagged circuit map'}</span>
                <span data-ready={snapshot.referenceLoaded}>Reference copy: {snapshot.referenceLoaded ? 'Loaded from Live Map' : 'Waiting for Live Map reference'}</span>
                {snapshot.mapContext.phase && <span>Current section: {snapshot.mapContext.cornerSpeed ? `${snapshot.mapContext.cornerSpeed} corner · ` : ''}{snapshot.mapContext.phase}</span>}
                {snapshot.mapContext.cornerShape && <span>Corner shape: {snapshot.mapContext.cornerShape}</span>}
                {snapshot.mapContext.sequenceId && <span>Corner sequence: {snapshot.mapContext.sequenceShape ?? 'Shape unavailable'} · {snapshot.mapContext.sequenceCornerIndex ? `${snapshot.mapContext.sequenceCornerIndex} of ` : ''}{snapshot.mapContext.sequenceCornerCount} corners</span>}
            </div>
            {!snapshot.mapReady && <p className="live-phrases__hint">Open Live Map in Add Visualization. In Circuit Maps, tag corners with corner and slow or fast, and tag straights with straight or long straight.</p>}
            <section aria-label="Sentence catalog">
                <p className="live-phrases__state" aria-label="Closure state">State: {snapshot.state.path.join(' → ')}</p>
                <ClosureTree node={snapshot.root} />
            </section>
            <section aria-label="Triggered sentences" className="live-phrases__output">
                <h3>Triggered sentences</h3>
                {snapshot.events.length === 0
                    ? <p>{enabled ? 'No phrases yet. Waiting for live driving data and a matching reference cue or overtaking opportunity.' : 'Enable detection to start live phrase guidance.'}</p>
                    : <ol>{snapshot.events.slice().reverse().map((event) => (
                        <li key={event.id}>
                            <time dateTime={new Date(event.timestamp).toISOString()}>
                                {new Date(event.timestamp).toLocaleTimeString(undefined, {
                                    hour: 'numeric', minute: 'numeric', second: 'numeric', fractionalSecondDigits: 3,
                                })}
                            </time>
                            <span>{event.sentence}</span>
                        </li>
                    ))}</ol>}
            </section>
        </section>
    );
});

LivePhrases.displayName = 'LivePhrases';
export default LivePhrases;
