import React, { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import type { DesktopGame } from 'contexts/DesktopGameContext';
import { capturePolicyCameraImage, PolicyCameraImage } from 'views/live-session/live-session-recording/policy-camera';
import type { DrivingModelConfig } from './driving-model-config';
import type { DrivingPolicy } from './driving-policy';
import { cameraChannelStats, DrivingCameraInspection } from './driving-camera-inspection';
import { parseRecordedDrivingSession, RecordedDrivingSession } from './recorded-driving-session';
import './RecordedFrameViewer.css';

export interface RecordedDrivingFrame {
    session: RecordedDrivingSession;
    videoSeconds: number;
    offsetSeconds: number;
    camera: PolicyCameraImage;
}
export interface RecordedFrameViewerHandle {
    replay(consume: (frame: RecordedDrivingFrame) => Promise<void>, signal: AbortSignal): Promise<void>;
}
interface Props {
    configuration: DrivingModelConfig;
    policy: DrivingPolicy | null;
    disabled: boolean;
    onFrame: (frame: RecordedDrivingFrame | null) => void;
    onBusyChange?: (busy: boolean) => void;
}
interface Example { seconds: number; score: number; thumbnail: string }
interface Gallery { layer: number; examples: Example[][]; sampled: number; interval: number; complete: boolean }
const videoFile = (file: File) => file.type.startsWith('video/') || /\.(mp4|webm|mkv|mov|m4v|avi)$/i.test(file.name);
const telemetryFile = (file: File) => /\.jsonl$/i.test(file.name);
const fileLabel = (file: File) => file.webkitRelativePath || file.name;
const abortError = () => new DOMException('Recording operation cancelled.', 'AbortError');
const checkAbort = (signal: AbortSignal) => { if (signal.aborted) throw abortError(); };
const message = (error: unknown) => error instanceof Error ? error.message : 'Could not read the recording.';
const timestamp = (seconds: number) => `${Math.floor(seconds / 60)}:${(seconds % 60).toFixed(2).padStart(5, '0')}`;

function seekVideo(video: HTMLVideoElement, seconds: number, signal: AbortSignal): Promise<void> {
    checkAbort(signal);
    if (!video.seeking && Math.abs(video.currentTime - seconds) < 0.0001 && video.readyState >= 2) return Promise.resolve();
    return new Promise((resolve, reject) => {
        const finish = (error?: Error) => {
            clearTimeout(timeout);
            video.removeEventListener('seeked', seeked);
            video.removeEventListener('error', failed);
            signal.removeEventListener('abort', aborted);
            error ? reject(error) : resolve();
        };
        const seeked = () => finish();
        const failed = () => finish(new Error('The recorded video could not be decoded. Try an MP4 or WebM recording.'));
        const aborted = () => finish(abortError());
        const timeout = setTimeout(() => finish(new Error('Timed out seeking the recorded video.')), 15000);
        video.addEventListener('seeked', seeked, { once: true });
        video.addEventListener('error', failed, { once: true });
        signal.addEventListener('abort', aborted, { once: true });
        try { video.currentTime = seconds; } catch (error) { finish(error instanceof Error ? error : new Error(String(error))); }
    });
}

function originalFrame(video: HTMLVideoElement): HTMLCanvasElement {
    if (video.readyState < 2 || !video.videoWidth || !video.videoHeight || video.seeking) {
        throw new Error('Wait for the recorded video frame to finish loading.');
    }
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    if (!context) throw new Error('Recorded frame pixels are unavailable.');
    context.drawImage(video, 0, 0);
    return canvas;
}

function thumbnail(source: HTMLCanvasElement): string {
    const canvas = document.createElement('canvas');
    canvas.width = 320; canvas.height = Math.max(1, Math.round(320 * source.height / source.width));
    const context = canvas.getContext('2d');
    if (!context) throw new Error('Recorded frame preview is unavailable.');
    context.drawImage(source, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.8);
}

const RecordedFrameViewer = forwardRef<RecordedFrameViewerHandle, Props>(function RecordedFrameViewer(props, ref) {
    const current = useRef(props); current.current = props;
    const video = useRef<HTMLVideoElement>(null);
    const mounted = useRef(true);
    const heatmap = useRef<HTMLCanvasElement>(null);
    const operation = useRef<AbortController | null>(null);
    const busyRef = useRef('');
    const [busy, setBusy] = useState('');
    const [videoFiles, setVideoFiles] = useState<File[]>([]);
    const [telemetryFiles, setTelemetryFiles] = useState<File[]>([]);
    const [folderSelected, setFolderSelected] = useState(false);
    const [selectedVideo, setSelectedVideo] = useState<File | null>(null);
    const [selectedTelemetry, setSelectedTelemetry] = useState<File | null>(null);
    const [url, setUrl] = useState('');
    const [game, setGame] = useState<DesktopGame>('acc');
    const [session, setSession] = useState<RecordedDrivingSession | null>(null);
    const [loadingTelemetry, setLoadingTelemetry] = useState(false);
    const sessionRef = useRef(session); sessionRef.current = session;
    const [offset, setOffset] = useState(0);
    const offsetRef = useRef(offset); offsetRef.current = offset;
    const [time, setTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [error, setError] = useState('');
    const [inspection, setInspection] = useState<DrivingCameraInspection | null>(null);
    const [inspectedImage, setInspectedImage] = useState('');
    const [inspectedTime, setInspectedTime] = useState(0);
    const [layerIndex, setLayerIndex] = useState(0);
    const [channel, setChannel] = useState(0);
    const [interval, setIntervalSeconds] = useState(1);
    const [gallery, setGallery] = useState<Gallery | null>(null);
    const [progress, setProgress] = useState('');
    const locked = props.disabled || Boolean(busy) || loadingTelemetry;
    const folderError = folderSelected && (!videoFiles.length || !telemetryFiles.length)
        ? 'Choose a folder containing a video and telemetry .jsonl file, or select the missing file individually.' : '';

    const changeBusy = (value: string) => {
        busyRef.current = value;
        if (mounted.current) { setBusy(value); current.current.onBusyChange?.(Boolean(value)); }
    };
    const resetInspection = () => { setInspection(null); setInspectedImage(''); setGallery(null); setProgress(''); };
    const capture = (): RecordedDrivingFrame => {
        if (!sessionRef.current || !video.current) throw new Error('Select a recorded video and telemetry JSONL first.');
        return { session: sessionRef.current, videoSeconds: video.current.currentTime, offsetSeconds: offsetRef.current,
            camera: capturePolicyCameraImage(originalFrame(video.current)) };
    };
    const publishFrame = () => {
        if (busyRef.current) return;
        if (video.current) setTime(video.current.currentTime);
        if (!sessionRef.current || !video.current || video.current.readyState < 2 || video.current.seeking) {
            current.current.onFrame(null); return;
        }
        try { current.current.onFrame(capture()); }
        catch (reason) { setError(message(reason)); current.current.onFrame(null); }
    };

    useEffect(() => {
        setDuration(0); setTime(0); setOffset(0); offsetRef.current = 0; setError(''); resetInspection();
        current.current.onFrame(null);
        if (!selectedVideo) { setUrl(''); return; }
        const next = URL.createObjectURL(selectedVideo);
        setUrl(next);
        return () => URL.revokeObjectURL(next);
        // File identity owns the video URL and the matching inspection results.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [selectedVideo]);
    useEffect(() => {
        let cancelled = false;
        setSession(null); sessionRef.current = null; current.current.onFrame(null); setError('');
        setOffset(0); offsetRef.current = 0; resetInspection();
        if (!selectedTelemetry) { setLoadingTelemetry(false); return; }
        setLoadingTelemetry(true);
        void selectedTelemetry.text().then((text) => {
            const loaded = parseRecordedDrivingSession(text, game);
            if (!cancelled) { sessionRef.current = loaded; setSession(loaded); publishFrame(); }
        }).catch((reason) => { if (!cancelled) setError(message(reason)); })
            .finally(() => { if (!cancelled) setLoadingTelemetry(false); });
        return () => { cancelled = true; };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [selectedTelemetry, game]);
    useEffect(() => {
        if (busyRef.current === 'Inspecting frame' || busyRef.current === 'Scanning recording') operation.current?.abort();
        resetInspection(); setLayerIndex(0); setChannel(0);
        // Model learning invalidates activation rankings from older weights.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [props.policy, props.policy?.steps]);
    useEffect(() => {
        mounted.current = true;
        return () => { mounted.current = false; operation.current?.abort(); current.current.onFrame(null); current.current.onBusyChange?.(false); };
    }, []);

    const runOperation = async (label: string, work: (signal: AbortSignal) => Promise<void>, external?: AbortSignal) => {
        if (busyRef.current) throw new Error('Wait for the current recording operation to finish.');
        const controller = new AbortController();
        const abort = () => controller.abort();
        if (external?.aborted) throw abortError();
        external?.addEventListener('abort', abort, { once: true });
        operation.current = controller; changeBusy(label); setError('');
        video.current?.pause();
        try { await work(controller.signal); }
        catch (reason) { if (controller.signal.aborted) throw abortError(); throw reason; }
        finally {
            external?.removeEventListener('abort', abort);
            if (operation.current === controller) { operation.current = null; changeBusy(''); if (mounted.current) publishFrame(); }
        }
    };
    useImperativeHandle(ref, () => ({
        replay: async (consume, signal) => {
            await runOperation('Replaying recording', async (active) => {
                const player = video.current, recording = sessionRef.current;
                if (!player || !recording || !Number.isFinite(player.duration) || player.duration <= 0) {
                    throw new Error('Load a recorded video and telemetry before replaying.');
                }
                const first = Math.max(0, offsetRef.current), last = Math.min(player.duration - 0.001, recording.durationSeconds + offsetRef.current);
                if (last < first) throw new Error('Video and telemetry do not overlap. Adjust the telemetry offset.');
                const previousTime = player.currentTime;
                const step = Math.max(0.1, current.current.configuration.runtime.controlIntervalMs / 1000);
                try {
                    for (let at = first; at <= last + 1e-7; at += step) {
                        checkAbort(active); await seekVideo(player, Math.min(at, last), active); checkAbort(active);
                        setTime(player.currentTime); await consume(capture()); checkAbort(active);
                    }
                } finally {
                    if (!active.aborted) await seekVideo(player, previousTime, active);
                }
            }, signal);
        },
    }));

    const reportError = (reason: unknown) => { if (mounted.current && !(reason instanceof Error && reason.name === 'AbortError')) setError(message(reason)); };
    const inspect = async (seconds?: number) => {
        const model = current.current.policy;
        if (!model || !video.current) return;
        try {
            await runOperation('Inspecting frame', async (signal) => {
                if (seconds !== undefined) await seekVideo(video.current!, seconds, signal);
                const source = originalFrame(video.current!), at = video.current!.currentTime;
                const result = await model.inspectCameraAsync(capturePolicyCameraImage(source)); checkAbort(signal);
                setInspection(result); setInspectedImage(source.toDataURL('image/jpeg', 0.9)); setInspectedTime(at);
                setLayerIndex((index) => Math.min(index, result.layers.length - 1));
            });
        } catch (reason) { reportError(reason); }
    };
    const scan = async () => {
        const model = current.current.policy, player = video.current;
        if (!model || !player || !Number.isFinite(player.duration) || player.duration <= 0) return;
        const selectedLayer = layerIndex, sampleInterval = interval, previousTime = player.currentTime, modelSteps = model.steps;
        let result: Gallery = { layer: selectedLayer, examples: [], sampled: 0, interval: sampleInterval, complete: false };
        try {
            await runOperation('Scanning recording', async (signal) => {
                const count = Math.ceil(player.duration / sampleInterval);
                setGallery(null);
                for (let index = 0; index < count; index++) {
                    checkAbort(signal); await seekVideo(player, index * sampleInterval, signal);
                    const source = originalFrame(player);
                    const response = await model.inspectCameraAsync(capturePolicyCameraImage(source)); checkAbort(signal);
                    const layer = response.layers[selectedLayer];
                    if (!layer) throw new Error('The selected camera layer is unavailable in this model.');
                    if (!result.examples.length) result.examples = Array.from({ length: layer.channels }, () => []);
                    let preview = '';
                    result.examples.forEach((examples, ch) => {
                        const score = cameraChannelStats(layer, ch).peak;
                        if (score <= 0 || (examples.length >= 3 && score <= examples[examples.length - 1].score)) return;
                        if (!preview) preview = thumbnail(source);
                        examples.push({ seconds: player.currentTime, score, thumbnail: preview });
                        examples.sort((a, b) => b.score - a.score || a.seconds - b.seconds);
                        examples.splice(3);
                    });
                    result.sampled++;
                    setProgress(`Scanned ${result.sampled} / ${count} frames`);
                }
                result.complete = true;
                await seekVideo(player, previousTime, signal);
            });
        } catch (reason) { reportError(reason); }
        finally { if (mounted.current && result.sampled && current.current.policy === model && model.steps === modelSteps) setGallery(result); }
    };

    const layer = inspection?.layers[layerIndex];
    const selectedChannel = layer ? Math.min(channel, layer.channels - 1) : 0;
    const stats = layer ? cameraChannelStats(layer, selectedChannel) : null;
    useEffect(() => {
        const canvas = heatmap.current;
        if (!canvas || !layer) return;
        const context = canvas.getContext('2d');
        if (!context) return;
        canvas.width = layer.width; canvas.height = layer.height;
        const pixels = context.createImageData(layer.width, layer.height);
        let maximum = 0;
        for (let i = selectedChannel; i < layer.values.length; i += layer.channels) maximum = Math.max(maximum, Math.abs(layer.values[i]));
        for (let pixel = 0; pixel < layer.width * layer.height; pixel++) {
            const value = maximum ? layer.values[pixel * layer.channels + selectedChannel] / maximum : 0;
            pixels.data[pixel * 4] = value > 0 ? Math.round(value * 255) : 0;
            pixels.data[pixel * 4 + 1] = Math.round(Math.abs(value) * 120);
            pixels.data[pixel * 4 + 2] = value < 0 ? Math.round(-value * 255) : 0;
            pixels.data[pixel * 4 + 3] = 255;
        }
        context.putImageData(pixels, 0, 0);
    }, [layer, selectedChannel]);

    const chooseFolder = (files: FileList | null) => {
        const candidates = Array.from(files ?? []), videos = candidates.filter(videoFile), telemetry = candidates.filter(telemetryFile);
        setFolderSelected(true);
        setVideoFiles(videos); setTelemetryFiles(telemetry);
        setSelectedVideo(videos.length === 1 ? videos[0] : null);
        setSelectedTelemetry(telemetry.length === 1 ? telemetry[0] : null);
    };
    const layers = inspection?.layers ?? props.configuration.network.cameraLayers.map((item, index) => ({ ...item, index }));
    return <section className="driving-card recorded-frame-viewer" aria-label="Recorded-frame viewer">
        <h3>Recorded session</h3>
        <div className="recorded-file-inputs">
            <label>Local session folder<input type="file" aria-label="Local session folder" multiple disabled={locked}
                {...{ webkitdirectory: '', directory: '' } as React.InputHTMLAttributes<HTMLInputElement>}
                onChange={(event) => chooseFolder(event.target.files)} /></label>
            <label>Recorded video<input type="file" accept="video/*,.mkv,.avi" disabled={locked}
                onChange={(event) => { const file = event.target.files?.[0] ?? null; setVideoFiles(file ? [file] : []); setSelectedVideo(file); }} /></label>
            <label>Telemetry JSONL<input type="file" accept=".jsonl" disabled={locked}
                onChange={(event) => { const file = event.target.files?.[0] ?? null; setTelemetryFiles(file ? [file] : []); setSelectedTelemetry(file); }} /></label>
        </div>
        {videoFiles.length > 1 && <label className="recorded-setting">Video in folder<select value={selectedVideo ? videoFiles.indexOf(selectedVideo) : ''} disabled={locked}
            onChange={(event) => setSelectedVideo(event.target.value === '' ? null : videoFiles[Number(event.target.value)])}>
            <option value="">Select a video</option>{videoFiles.map((file, index) => <option key={index} value={index}>{fileLabel(file)}</option>)}</select></label>}
        {telemetryFiles.length > 1 && <label className="recorded-setting">Telemetry in folder<select value={selectedTelemetry ? telemetryFiles.indexOf(selectedTelemetry) : ''} disabled={locked}
            onChange={(event) => setSelectedTelemetry(event.target.value === '' ? null : telemetryFiles[Number(event.target.value)])}>
            <option value="">Select telemetry</option>{telemetryFiles.map((file, index) => <option key={index} value={index}>{fileLabel(file)}</option>)}</select></label>}
        <div className="recorded-settings">
            <label>Game for JSONL without game metadata<select value={game} disabled={locked} onChange={(event) => setGame(event.target.value as DesktopGame)}>
                <option value="acc">Assetto Corsa Competizione</option><option value="ac">Assetto Corsa</option><option value="iracing">iRacing</option></select></label>
            <label>First telemetry at video time (seconds)<input type="number" step="0.01" value={offset} disabled={locked}
                onChange={(event) => { const value = event.target.valueAsNumber; if (Number.isFinite(value)) { offsetRef.current = value; setOffset(value); publishFrame(); } }} /></label>
        </div>
        <p className="driving-note">A positive offset means telemetry starts later in the video. Choose matching files from the same session; adjust this offset to align their motion.</p>
        {loadingTelemetry && <p role="status">Reading telemetry…</p>}
        {session && <p>{selectedTelemetry?.name} · {session.game} · {session.rows.length.toLocaleString()} samples · {session.durationSeconds.toFixed(2)} seconds</p>}
        {(error || folderError) && <p className="driving-error" role="alert">{error || folderError}</p>}
        {url && <>
            <video ref={video} className="recorded-video" src={url} controls={!locked} preload="auto" aria-label="Original recorded video"
                onLoadedMetadata={(event) => setDuration(Number.isFinite(event.currentTarget.duration) ? event.currentTarget.duration : 0)}
                onLoadedData={publishFrame} onTimeUpdate={publishFrame} onSeeked={publishFrame}
                onSeeking={() => { if (!busyRef.current) current.current.onFrame(null); }}
                onError={() => { setError('The recorded video could not be decoded. Try an MP4 or WebM recording.'); current.current.onFrame(null); }} />
            <label className="recorded-timeline">Recorded frame · {timestamp(time)} / {timestamp(duration)}
                <input type="range" min="0" max={Math.max(0, duration - 0.001)} step="0.01" value={time} disabled={locked || !duration}
                    onChange={(event) => { if (video.current) { video.current.pause(); video.current.currentTime = Number(event.target.value); setTime(Number(event.target.value)); } }} /></label>
            <h3>Camera model inspection</h3>
            <p>Inspect the loaded driving model’s camera CNN. Feature maps show learned activations; they are diagnostic views, not a recognition accuracy score. {props.policy ? `Model: ${props.policy.steps.toLocaleString()} training steps.` : 'Open a model with camera weights to inspect it.'}</p>
            <div className="recorded-settings">
                <label>Feature layer<select value={layerIndex} disabled={locked} onChange={(event) => { setLayerIndex(Number(event.target.value)); setChannel(0); }}>
                    {layers.map((item, index) => <option key={index} value={index}>Layer {index + 1} · {item.channels} channels</option>)}</select></label>
                <label>Feature channel<select value={selectedChannel} disabled={locked || !layer} onChange={(event) => setChannel(Number(event.target.value))}>
                    {Array.from({ length: layer?.channels ?? 1 }, (_, index) => <option key={index} value={index}>Channel {index + 1}</option>)}</select></label>
            </div>
            <div className="driving-buttons"><button disabled={locked || !props.policy || !duration} onClick={() => { void inspect(); }}>Inspect current frame</button></div>
            {inspection && layer && <>
                <div className="recorded-feature-images">
                    <figure><img src={inspectedImage} alt={`Original recorded frame at ${timestamp(inspectedTime)}`} /><figcaption>Original image · {timestamp(inspectedTime)}</figcaption></figure>
                    <figure><canvas ref={heatmap} aria-label={`Layer ${layerIndex + 1}, channel ${selectedChannel + 1} activation map`} /><figcaption>Layer {layerIndex + 1} / channel {selectedChannel + 1} · {layer.width} × {layer.height}</figcaption></figure>
                </div>
                <p className="driving-note">Mean {stats?.mean.toFixed(5)} · Peak {stats?.peak.toFixed(5)}. Orange: positive activation; blue: negative activation. Brightness is scaled within this channel.</p>
            </>}
            <h3>Strongest real examples per channel</h3>
            <p>Scan the video for the three frames with the highest positive peak activation in each channel of the selected layer. Rankings cover sampled frames only.</p>
            <label className="recorded-setting">Sample every (seconds)<input type="number" min="0.1" max="60" step="0.1" value={interval} disabled={locked}
                onChange={(event) => { const value = event.target.valueAsNumber; if (Number.isFinite(value)) setIntervalSeconds(Math.max(0.1, Math.min(60, value))); }} /></label>
            <div className="driving-buttons"><button disabled={locked || !props.policy || !duration} onClick={() => { void scan(); }}>Find strongest examples</button>
                {busy && <button className="driving-stop" onClick={() => operation.current?.abort()}>Cancel {busy === 'Replaying recording' ? 'replay' : 'inspection'}</button>}</div>
            {(busy || progress) && <p role="status">{busy}{busy && progress ? ' · ' : ''}{progress}</p>}
            {gallery && <>
                <p>{gallery.complete ? 'Scan complete' : 'Partial scan'} · Layer {gallery.layer + 1} · {gallery.sampled} frames sampled every {gallery.interval} seconds.</p>
                <div className="recorded-channel-gallery">{gallery.examples.map((examples, index) => <section key={index} aria-label={`Strongest examples for channel ${index + 1}`}>
                    <h4>Channel {index + 1}</h4>
                    {!examples.length ? <p>No positive activation in sampled frames.</p> : <div className="recorded-example-row">{examples.map((example) => <button key={example.seconds} disabled={locked}
                        onClick={() => { setLayerIndex(gallery.layer); setChannel(index); void inspect(example.seconds); }}>
                        <img src={example.thumbnail} alt={`Real example for channel ${index + 1} at ${timestamp(example.seconds)}`} />
                        <span>{timestamp(example.seconds)} · {example.score.toFixed(5)}</span>
                    </button>)}</div>}
                </section>)}</div>
            </>}
        </>}
    </section>;
});
export default RecordedFrameViewer;
