import React, { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import {
    NamedOperationComponentHandle,
    OPERATION_COMPONENT_NAMES,
    useOptionalOperationComponentRefDirectory,
    useRegisterOperationComponentRef,
} from 'contexts/OperationComponentRefContext';
import type { LiveSessionHandle } from '../LiveSessionView';
import { PHRASE_RULES, PhraseEngine, PhraseSnapshot } from './phrase-engine';
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
                <p>Overtaking guidance combines Track Vision positions with Live Map corner shapes, consecutive corners and your progress through the corner.</p>
            </header>
            <div className="live-phrases__sources" role="status">
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
                <h3>All possible sentences <span>({PHRASE_RULES.length})</span></h3>
                <p className="live-phrases__hint">All {PHRASE_RULES.length} guides are listed below. The first matching guide takes priority; its conditions must hold for 0.8 s. Each guide has an 8 s cooldown and requires a 0.5 s clear period before repeating.</p>
                <ol className="live-phrases__catalog">
                    {PHRASE_RULES.map((rule, index) => {
                        const state = snapshot.rules[index];
                        return <li key={rule.id}>
                            <div className="live-phrases__rule-heading"><span>{rule.category}</span><span data-active={state.status === 'Active'}>{state.status}</span></div>
                            <strong>{rule.sentence}</strong>
                            <ul className="live-phrases__conditions" aria-label={`${rule.category} conditions`}>
                                {state.conditions.map((condition, conditionIndex) => (
                                    <li key={conditionIndex} className="live-phrases__condition" data-condition-fit={condition.conditionFit} data-input-missing={condition.inputMissing}>
                                        <span>{condition.description}</span>
                                        <span className="live-phrases__condition-status">{condition.inputMissing ? 'Missing input' : condition.conditionFit ? 'Met' : 'Not met'}</span>
                                    </li>
                                ))}
                            </ul>
                            <small>Hold for {rule.holdMs / 1000} s · Cooldown 8 s</small>
                            {state.missing.length > 0 && <small className="live-phrases__missing">Waiting for: {state.missing.join(', ')}</small>}
                        </li>;
                    })}
                </ol>
                <p className="live-phrases__hint">Live Map tags identify slow and fast corners; centerline geometry estimates their shapes. Tag an enclosing area with consecutive corners to link the corner segments inside it. Lap position estimates entry, middle and exit. Track Vision reports visible positions, but cannot confirm overlap, a clear passing lane or an opponent’s intent. Guidance depends on those conditions being met. Telemetry expires after 1.5 s and vision after 2 s; missing inputs withhold the affected guides.</p>
            </section>
            <section aria-label="Triggered sentences" className="live-phrases__output">
                <h3>Triggered sentences</h3>
                {snapshot.events.length === 0
                    ? <p>No phrases yet. Waiting for an opponent ahead, a tagged Live Map section, and live driving data.</p>
                    : <ol>{snapshot.events.slice().reverse().map((event) => (
                        <li key={event.id}><time dateTime={new Date(event.timestamp).toISOString()}>{new Date(event.timestamp).toLocaleTimeString()}</time><span>{event.sentence}</span></li>
                    ))}</ol>}
            </section>
        </section>
    );
});

LivePhrases.displayName = 'LivePhrases';
export default LivePhrases;
