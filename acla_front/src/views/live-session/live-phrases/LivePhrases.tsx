import React, { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import {
    NamedOperationComponentHandle,
    OPERATION_COMPONENT_NAMES,
    useOptionalOperationComponentRefDirectory,
    useRegisterOperationComponentRef,
} from 'contexts/OperationComponentRefContext';
import type { LiveSessionHandle } from '../LiveSessionView';
import { synthesizeTts, type TtsPack } from 'components/tts';
import { audioManager, type PlaybackHandle } from 'services/audio';
import { getPhraseActions, PHRASE_DEFINITIONS, PhraseEngine, PhraseSnapshot, type PhraseEvent } from './phrase-engine';
import { LivePhraseOverlay, LIVE_PHRASE_DISPLAY_MS } from './LivePhraseOverlay';
import ClosureTree from './ClosureTree';
import './LivePhrases.css';

const speechActions = PHRASE_DEFINITIONS.flatMap(getPhraseActions);

export interface LivePhrasesHandle extends NamedOperationComponentHandle {
    getSnapshot(): PhraseSnapshot;
    getPhraseCatalog(): typeof PHRASE_DEFINITIONS;
}

const LivePhrases = forwardRef<LivePhrasesHandle, { name: string }>(({ name }, forwardedRef) => {
    const directory = useOptionalOperationComponentRefDirectory();
    const root = directory?.findComponentRef<LiveSessionHandle>(OPERATION_COMPONENT_NAMES.LIVE_SESSION)?.current ?? null;
    const [enabled, setEnabled] = useState(false);
    const [snapshot, setSnapshot] = useState(() => new PhraseEngine().evaluate(Date.now()));
    const [overlayError, setOverlayError] = useState<string | null>(null);
    const [voiceError, setVoiceError] = useState<string | null>(null);
    const [speechStatus, setSpeechStatus] = useState<'Disabled' | 'Preparing' | 'Ready' | 'Incomplete'>('Disabled');
    const voicePacks = useRef(new Map<string, TtsPack>());
    const snapshotRef = useRef(snapshot);
    snapshotRef.current = snapshot;

    useEffect(() => {
        const engine = new PhraseEngine();
        setOverlayError(null);
        setVoiceError(null);
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
        let lastVoiceEventId = 0;
        const playReadyVoice = () => {
            if (controller.signal.aborted || !pendingVoice) return;
            if (Date.now() >= pendingVoice.timestamp + LIVE_PHRASE_DISPLAY_MS) {
                pendingVoice = null;
                return;
            }
            const pack = voicePacks.current.get(pendingVoice.sentence);
            if (!pack) return;
            // Consume the event even if live chat rejects or interrupts its audio.
            pendingVoice = null;
            playback?.stop();
            playback = audioManager.play({
                url: pack.audioDataUrl, type: 'voice', priority: 25, // Live chat defaults to 50.
                onComplete: (outcome) => {
                    if (!controller.signal.aborted && outcome.status === 'failed') setVoiceError(outcome.error.message);
                },
            });
        };
        // Prepare sequentially through the system TTS service; retain completed clips across toggles.
        void (async () => {
            for (const { sentence } of speechActions) {
                if (controller.signal.aborted) return;
                if (voicePacks.current.has(sentence)) continue;
                try {
                    const pack = await synthesizeTts({ text: sentence, speed: 1.2 }, controller.signal);
                    if (controller.signal.aborted) return;
                    voicePacks.current.set(sentence, pack);
                    playReadyVoice();
                } catch (error) {
                    if (controller.signal.aborted) return;
                    setVoiceError(error instanceof Error ? error.message : String(error));
                }
            }
            setSpeechStatus(speechActions.every(({ sentence }) => voicePacks.current.has(sentence)) ? 'Ready' : 'Incomplete');
        })();
        const dispose = () => {
            controller.abort();
            pendingVoice = null;
            playback?.stop();
            overlay?.dispose();
        };
        const publish = (next: PhraseSnapshot) => {
            setSnapshot((previous) => (
                JSON.stringify(previous) === JSON.stringify(next) ? previous : next
            ));
            overlay?.update(next);
            if (next.events.length === 0) {
                lastVoiceEventId = 0;
                pendingVoice = null;
                playback?.stop();
                playback = null;
            }
            for (const event of next.events) {
                if (event.id <= lastVoiceEventId) continue;
                lastVoiceEventId = event.id;
                pendingVoice = event;
            }
            // Never speak a delayed preparation after its driving conditions have passed.
            if (!next.telemetryReady || !next.closures.some((rule) => rule.id === pendingVoice?.ruleId && rule.status === 'Active')) {
                pendingVoice = null;
            }
            playReadyVoice();
        };
        publish(engine.evaluate(Date.now()));
        if (!root) return dispose;
        const updateVision = () => publish(engine.receiveVision(root.getTrackVisionDetection(), Date.now()));
        const unsubscribeVision = root.subscribeTrackVision(updateVision);
        updateVision();
        const updateMap = () => publish(engine.receiveMap(root.getLiveCircuitMap(), Date.now()));
        const unsubscribeMap = root.subscribeLiveCircuitMap(updateMap);
        updateMap();
        const unsubscribeTelemetry = root.subscribeTelemetry((event) => {
            publish(engine.receiveTelemetry(event, Date.now()));
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
                <p>Overtaking guidance combines Track Vision BEV positions with Live Map corner shapes, consecutive corners and your progress through the corner.</p>
                <p>Triggered phrases pop out automatically when the Assistant overlay is enabled.</p>
                <p>Enabling detection prepares spoken guidance. Live chat audio takes priority over spoken phrases.</p>
            </header>
            {overlayError && <p role="alert">Unable to display live phrase in the overlay: {overlayError}</p>}
            {voiceError && <p role="alert">Unable to prepare or play live phrase speech: {voiceError}</p>}
            <div className="live-phrases__sources" role="status">
                <span data-ready={enabled}>Detection: {enabled ? 'Enabled' : 'Disabled'}</span>
                <span data-ready={speechStatus === 'Ready'}>Speech: {speechStatus}</span>
                <span data-ready={snapshot.telemetryReady}>Telemetry: {snapshot.telemetryReady ? 'Live' : 'Waiting for live data'}</span>
                <span data-ready={snapshot.visionReady}>Track Vision: {snapshot.visionReady ? 'Live' : 'Unavailable or stale'}</span>
                <span data-ready={snapshot.mapReady}>Live Map: {snapshot.mapReady ? 'Ready' : 'Waiting for a tagged circuit map'}</span>
                {snapshot.mapContext.phase && <span>Current section: {snapshot.mapContext.cornerSpeed ? `${snapshot.mapContext.cornerSpeed} corner · ` : ''}{snapshot.mapContext.phase}</span>}
                {snapshot.mapContext.cornerShape && <span>Corner shape: {snapshot.mapContext.cornerShape}</span>}
                {snapshot.mapContext.sequenceId && <span>Corner sequence: {snapshot.mapContext.sequenceShape ?? 'Shape unavailable'} · {snapshot.mapContext.sequenceCornerIndex ? `${snapshot.mapContext.sequenceCornerIndex} of ` : ''}{snapshot.mapContext.sequenceCornerCount} corners</span>}
            </div>
            {!snapshot.visionReady && <p className="live-phrases__hint">Open Track Vision in Add Visualization, share your forward-facing driving view, and apply the camera calibration to enable overtaking guidance.</p>}
            {!snapshot.mapReady && <p className="live-phrases__hint">Open Live Map in Add Visualization. In Circuit Maps, tag corners with corner and slow or fast, and tag straights with straight or long straight.</p>}
            <section aria-label="Sentence catalog">
                <p className="live-phrases__state" aria-label="Closure state">State: {snapshot.state.path.join(' → ')}</p>
                <ClosureTree node={snapshot.root} />
                <details className="live-phrases__disclosure live-phrases__help">
                    <summary>How conditions are evaluated</summary>
                    <p className="live-phrases__hint">AND requires both conditions; OR allows either alternative. Groups are evaluated first; within each group, AND is evaluated before OR.</p>
                    <p className="live-phrases__hint">Live Map tags identify slow and fast corners; centerline geometry estimates their shapes. Tag an enclosing area with consecutive corners to link the corner segments inside it. Lap position estimates entry, middle and exit. Track Vision BEV estimates visible positions using camera calibration and a flat road, but cannot confirm overlap, a clear passing lane or an opponent’s intent. Guidance depends on those conditions being met. Telemetry expires after 1.5 s and vision after 2 s; missing inputs do not match a condition, but an OR alternative can still match. Live telemetry is always required.</p>
                </details>
            </section>
            <section aria-label="Triggered sentences" className="live-phrases__output">
                <h3>Triggered sentences</h3>
                {snapshot.events.length === 0
                    ? <p>{enabled ? 'No phrases yet. Waiting for an opponent ahead, a tagged Live Map section, and live driving data.' : 'Enable detection to start live phrase guidance.'}</p>
                    : <ol>{snapshot.events.slice().reverse().map((event) => (
                        <li key={event.id}><time dateTime={new Date(event.timestamp).toISOString()}>{new Date(event.timestamp).toLocaleTimeString()}</time><span>{event.sentence}</span></li>
                    ))}</ol>}
            </section>
        </section>
    );
});

LivePhrases.displayName = 'LivePhrases';
export default LivePhrases;
