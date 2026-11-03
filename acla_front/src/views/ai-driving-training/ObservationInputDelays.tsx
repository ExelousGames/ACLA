import React from 'react';
import { VISION_TIMING_STAGES, VisionTimingSnapshot } from 'views/live-session/track-vision/vision-timing';
import { DRIVING_MODEL_TIMING_STAGES, DrivingModelTimingSnapshot } from './driving-model-timing';

export interface ObservationDelaySnapshot {
    checkedAt: number;
    telemetryReceivedAt?: number;
    capturedAt?: number;
    pairedTelemetryReceivedAt?: number;
    publishedTiming?: VisionTimingSnapshot;
    latestTiming?: VisionTimingSnapshot | null;
    observationMs: number;
    observationReady: boolean;
    modelTiming?: DrivingModelTimingSnapshot;
}

const DISPLAY_TIMING_STAGES = VISION_TIMING_STAGES.map((stage, index) => ({ ...stage, index }))
    .filter(({ id }) => !['segment', 'segmentReadback', 'depth', 'depthReadback', 'track', 'scene', 'birdsEye'].includes(id));

const milliseconds = (value?: number) => value !== undefined && Number.isFinite(value) && value >= 0
    ? `${Math.round(value)} ms` : 'Unavailable';
const age = (now: number, timestamp?: number) => timestamp !== undefined && Number.isFinite(timestamp) ? now - timestamp : undefined;

function Freshness({ value, limit }: { value?: number; limit: number }) {
    const valid = value !== undefined && Number.isFinite(value);
    const stale = valid && (value < 0 || value > limit);
    return <dd data-stale={stale}>{valid && value < 0 ? 'Clock mismatch' : milliseconds(value)} / {limit} ms limit
        {valid && ` · ${stale ? 'Stale' : 'Fresh'}`}</dd>;
}

export default function ObservationInputDelays({ snapshot, telemetryMaxAgeMs, visionMaxAgeMs }: {
    snapshot?: ObservationDelaySnapshot; telemetryMaxAgeMs: number; visionMaxAgeMs: number;
}) {
    const now = snapshot?.checkedAt ?? Date.now();
    const timing = snapshot?.latestTiming;
    const model = snapshot?.modelTiming;
    const stageIndex = VISION_TIMING_STAGES.findIndex(({ id }) => id === timing?.currentStage);
    return <section className="driving-card driving-input-delays" aria-label="Input profiling">
        <h2>Input profiling</h2>
        <p>Telemetry freshness, input processing and driving model timings for diagnosing delays.</p>
        <dl className="driving-details">
            <dt>Live telemetry age</dt><Freshness value={age(now, snapshot?.telemetryReceivedAt)} limit={telemetryMaxAgeMs} />
            <dt>Telemetry age at image capture</dt><Freshness
                value={snapshot?.capturedAt === undefined ? undefined : age(snapshot.capturedAt, snapshot.pairedTelemetryReceivedAt)}
                limit={telemetryMaxAgeMs} />
            <dt>Paired telemetry age now</dt><dd>{milliseconds(age(now, snapshot?.pairedTelemetryReceivedAt))}</dd>
            <dt>Image age for AI</dt><Freshness value={age(now, snapshot?.capturedAt)} limit={visionMaxAgeMs} />
            <dt>Published frame processing</dt><dd>{milliseconds(snapshot?.publishedTiming?.totalMs)}</dd>
            <dt>Time since frame published</dt><dd>{milliseconds(age(now, snapshot?.publishedTiming?.finishedAt))}</dd>
            <dt>Capture interval</dt><dd>{milliseconds(timing?.captureIntervalMs)}</dd>
            <dt>Observation validation / assembly</dt><dd>{milliseconds(snapshot?.observationMs)}
                {snapshot && !snapshot.observationReady && ' · Inputs invalid'}</dd>
        </dl>
        <table className="driving-delay-stages">
            <caption>Latest capture processing</caption>
            <thead><tr><th scope="col">Stage</th><th scope="col">Delay</th></tr></thead>
            <tbody>{DISPLAY_TIMING_STAGES.map(({ id, label, index }) => <tr key={id}>
                <th scope="row">{label}</th>
                <td>{timing?.currentStage === id ? 'Running'
                    : timing?.stages[id] !== undefined ? milliseconds(timing.stages[id])
                        : !timing ? 'Unavailable' : timing.state === 'processing' && index > stageIndex ? 'Pending' : 'Not run'}</td>
            </tr>)}</tbody>
            <tfoot><tr><th scope="row">Total frame processing</th><td>{timing?.state === 'processing' ? 'Running'
                : `${milliseconds(timing?.totalMs)}${timing ? ` · ${timing.state === 'complete' ? 'Complete' : 'Incomplete'}` : ''}`}</td></tr></tfoot>
        </table>
        <table className="driving-delay-stages">
            <caption>Driving model processing</caption>
            <thead><tr><th scope="col">Stage</th><th scope="col">Delay</th></tr></thead>
            <tbody>{DRIVING_MODEL_TIMING_STAGES.map(({ id, label }) => <tr key={id}>
                <th scope="row">{label}</th>
                <td>{model?.currentStage === id ? 'Running'
                    : model?.stages[id] !== undefined ? milliseconds(model.stages[id])
                        : !model ? 'Unavailable' : model.state === 'processing' ? 'Not run yet' : 'Not run'}</td>
            </tr>)}</tbody>
            <tfoot><tr><th scope="row">Total driving model cycle</th><td>{model?.state === 'processing' ? 'Running'
                : `${milliseconds(model?.totalMs)}${model ? ` · ${model.state === 'complete' ? 'Complete' : 'Incomplete'}` : ''}`}</td></tr></tfoot>
        </table>
        <dl className="driving-details">
            <dt>Replay count</dt><dd>{model?.replayCount ?? 'Unavailable'}</dd>
            <dt>History length</dt><dd>{model?.historyLength ?? 'Unavailable'}</dd>
        </dl>
        <p className="driving-note">{model
            ? `Latest model cycle: ${model.training ? 'Training' : 'Evaluation'} · ${model.source === 'ai' ? 'AI' : 'Human'} control.
                Observation age: ${milliseconds(age(now, model.capturedAt))}.${model.finishedAt === undefined ? ''
                    : ` Finished ${milliseconds(age(now, model.finishedAt))} ago.`}`
            : 'Waiting for a driving model cycle on fresh inputs.'}</p>
        <p className="driving-note">Training CPU preparation / submission measures elapsed CPU-side work through the last submission,
            including learning from AI actions or human demonstrations and delayed plan replays.
            GPU completion wait measures the subsequent queue wait, including queued work and scheduling delays.
            Inference includes camera encoding, policy evaluation and output readback.
            These elapsed times are separate from GPU timestamp queries used to measure individual passes.
            Replay count counts delayed plan samples submitted this cycle. History length counts populated history bins
            in the current input, excluding the current observation.
            The latest cycle remains visible while stopped or waiting for fresh inputs.</p>
        <p className="driving-note">Updates every 250 ms, including when inputs are invalid. Stage times include queue waits;
            the frame total includes all shared Track Vision processing, including stages not listed here.
            Game/car, speed, velocity, yaw, gear/RPM and controls share the paired telemetry timestamp.
            Simulator rendering and transport before the app receives telemetry or copies the screen are not measured.</p>
    </section>;
}
