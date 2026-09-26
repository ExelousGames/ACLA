import React, { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import {
    NamedOperationComponentHandle,
    OPERATION_COMPONENT_NAMES,
    useOptionalOperationComponentRefDirectory,
    useRegisterOperationComponentRef,
} from 'contexts/OperationComponentRefContext';
import type { LiveSessionHandle } from '../LiveSessionView';
import { describeConditions, PHRASE_RULES, PhraseEngine, PhraseSnapshot } from './phrase-engine';
import './LivePhrases.css';

export interface LivePhrasesHandle extends NamedOperationComponentHandle {
    getSnapshot(): PhraseSnapshot;
    getPhraseCatalog(): typeof PHRASE_RULES;
}

const LivePhrases = forwardRef<LivePhrasesHandle, { name: string }>(({ name }, forwardedRef) => {
    const directory = useOptionalOperationComponentRefDirectory();
    const root = directory?.findComponentRef<LiveSessionHandle>(OPERATION_COMPONENT_NAMES.LIVE_SESSION)?.current ?? null;
    const [snapshot, setSnapshot] = useState(() => new PhraseEngine().evaluate(Date.now()));
    const snapshotRef = useRef(snapshot);
    snapshotRef.current = snapshot;

    useEffect(() => {
        const engine = new PhraseEngine();
        const publish = (next: PhraseSnapshot) => setSnapshot((previous) => (
            JSON.stringify(previous) === JSON.stringify(next) ? previous : next
        ));
        publish(engine.evaluate(Date.now()));
        if (!root) return;
        const updateVision = () => publish(engine.receiveVision(root.getTrackVisionDetection(), Date.now()));
        const unsubscribeVision = root.subscribeTrackVision(updateVision);
        updateVision();
        const unsubscribeTelemetry = root.subscribeTelemetry((event) => {
            publish(engine.receiveTelemetry(event, Date.now()));
        });
        // Expire inputs even if the simulator or screen capture stops sending events.
        const timer = setInterval(() => publish(engine.evaluate(Date.now())), 250);
        return () => {
            clearInterval(timer);
            unsubscribeVision();
            unsubscribeTelemetry();
        };
    }, [root]);

    const handle = useMemo<LivePhrasesHandle>(() => ({
        getComponentName: () => name,
        getSnapshot: () => snapshotRef.current,
        getPhraseCatalog: () => PHRASE_RULES,
    }), [name]);
    useImperativeHandle(forwardedRef, () => handle, [handle]);
    const registeredHandle = useRef(handle);
    registeredHandle.current = handle;
    useRegisterOperationComponentRef(registeredHandle);

    return (
        <section className="live-phrases" aria-label="Live phrases">
            <header>
                <h2>Live phrases <span>Local rules</span></h2>
                <p>Live Phrases selects sentences from the driver and opponent positions reported by Track Vision.</p>
            </header>
            <div className="live-phrases__sources" role="status">
                <span data-ready={snapshot.telemetryReady}>Telemetry: {snapshot.telemetryReady ? 'Live' : 'Waiting for live data'}</span>
                <span data-ready={snapshot.visionReady}>Track Vision: {snapshot.visionReady ? 'Live' : 'Unavailable or stale'}</span>
            </div>
            {!snapshot.visionReady && <p className="live-phrases__hint">Open Track Vision in Add Visualization, share your forward-facing driving view, and apply the camera calibration to enable corner-position phrases.</p>}
            <section aria-label="Sentence catalog">
                <h3>All possible sentences <span>({PHRASE_RULES.length})</span></h3>
                <p className="live-phrases__hint">Every possible sentence is listed below, including inactive rules. All conditions must hold for the listed duration. A phrase appears once per match, with an 8 s cooldown and 0.5 s clear period before repeating.</p>
                <ol className="live-phrases__catalog">
                    {PHRASE_RULES.map((rule, index) => {
                        const state = snapshot.rules[index];
                        return <li key={rule.id}>
                            <div className="live-phrases__rule-heading"><span>{rule.category}</span><span data-active={state.status === 'Active'}>{state.status}</span></div>
                            <strong>{rule.sentence}</strong>
                            <p>{describeConditions(rule)}</p>
                            <small>Hold for {rule.holdMs / 1000} s · Cooldown 8 s</small>
                            {state.missing.length > 0 && <small className="live-phrases__missing">Waiting for: {state.missing.join(', ')}</small>}
                        </li>;
                    })}
                </ol>
                <p className="live-phrases__hint">Track Vision analyzes the screen and reports positions. Telemetry confirms live driving and speed; G-forces are not used. Telemetry must be at most 1.5 s old and Track Vision results at most 2 s old. Unknown positions produce no position phrase.</p>
            </section>
            <section aria-label="Triggered sentences" className="live-phrases__output">
                <h3>Triggered sentences</h3>
                {snapshot.events.length === 0
                    ? <p>No phrases yet. Waiting for a visible corner, track edges, an opponent ahead, and live driving data.</p>
                    : <ol>{snapshot.events.slice().reverse().map((event) => (
                        <li key={event.id}><time dateTime={new Date(event.timestamp).toISOString()}>{new Date(event.timestamp).toLocaleTimeString()}</time><span>{event.sentence}</span></li>
                    ))}</ol>}
            </section>
        </section>
    );
});

LivePhrases.displayName = 'LivePhrases';
export default LivePhrases;
