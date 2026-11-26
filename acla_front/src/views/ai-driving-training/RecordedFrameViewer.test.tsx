import React from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import trainingLabels from 'data/recorded-training-labels.json';
import { encodeVideoTelemetryStart, VIDEO_TIMING_BYTES } from 'data/session-video-timing';
import RecordedFrameViewer, { RecordedFrameViewerHandle, type RecordedReplayFrame } from './RecordedFrameViewer';
import { cloneConfig } from './driving-model-config';
import type { DrivingPolicy } from './driving-policy';
import { parseRecordedDrivingSessionStream, type RecordedDrivingRow, type RecordedDrivingSession } from './recorded-driving-session';
import type { TrainingAnnotation, TrainingSelection } from './RecordedTrainingRanges';

jest.mock('./recorded-driving-session', () => ({
    ...jest.requireActual('./recorded-driving-session'),
    parseRecordedDrivingSessionStream: jest.fn(),
}));
const parse = jest.mocked(parseRecordedDrivingSessionStream);
const telemetryStream = (name: string) => ({ name } as unknown as ReadableStream<Uint8Array>);
const recordedRow = (index: number, timeSeconds: number, throttle: number, brake: number, gear: number): RecordedDrivingRow => ({
    index, timeSeconds, segment: 0, sample: { Physics_gas: throttle, Physics_brake: brake, Physics_gear: gear },
});
const recording: RecordedDrivingSession = { id: 'local-1', game: 'acc',
    rows: [recordedRow(0, 0, 0.2, 0.6, 2), recordedRow(1, 0.125, 0.45, 0.3, 3),
        recordedRow(2, 0.25, 0.7, 0.15, 4), recordedRow(3, 0.3, 0.8, 0.1, 4)],
    startTimeSeconds: 100, durationSeconds: 0.3, timestampField: 'Graphics_analyst_session_time' };
const result = (peak: number) => ({ layers: [{ index: 0, width: 2, height: 1, channels: 2,
    values: new Float32Array([peak, 0, peak / 2, -0.1]) }] });
const originalFetch = window.fetch;

beforeEach(() => {
    parse.mockReset().mockResolvedValue(recording);
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: jest.fn(() => 'blob:recorded-video') });
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: jest.fn() });
    jest.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => undefined);
    jest.spyOn(HTMLCanvasElement.prototype, 'toDataURL').mockReturnValue('data:image/jpeg;base64,preview');
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (this: HTMLCanvasElement) {
        return { drawImage: jest.fn(), fillRect: jest.fn(), fillStyle: '', putImageData: jest.fn(),
            getImageData: () => ({ data: new Uint8ClampedArray(this.width * this.height * 4) }),
            createImageData: (width: number, height: number) => ({ data: new Uint8ClampedArray(width * height * 4) }),
        } as unknown as CanvasRenderingContext2D;
    } as any);
});
afterEach(() => { delete window.recordedTrainingRanges; window.fetch = originalFetch; jest.restoreAllMocks(); });

function mockVideo(video: HTMLVideoElement, duration: number) {
    let time = 0;
    Object.defineProperties(video, {
        duration: { configurable: true, value: duration }, videoWidth: { configurable: true, value: 16 },
        videoHeight: { configurable: true, value: 9 }, readyState: { configurable: true, value: 2 },
        currentTime: { configurable: true, get: () => time, set: (value: number) => {
            time = value;
            Promise.resolve().then(() => video.dispatchEvent(new Event('seeked')));
        } },
    });
}
function mockTimeline() {
    const timeline = screen.getByRole('slider', { name: /Recorded frame/ });
    jest.spyOn(timeline, 'getBoundingClientRect').mockReturnValue({
        x: 100, y: 0, top: 0, right: 1100, bottom: 20, left: 100, width: 1000, height: 20, toJSON: () => ({}),
    });
    return timeline;
}
function pointer(timeline: HTMLElement, type: string, clientX: number, pointerType = 'mouse') {
    const event = new MouseEvent(type, { bubbles: true, cancelable: true, clientX });
    Object.defineProperties(event, {
        pointerId: { value: 1 }, isPrimary: { value: true }, pointerType: { value: pointerType },
    });
    fireEvent(timeline, event);
}
function openEditor() {
    fireEvent.click(screen.getByRole('button', { name: 'Add new training annotation' }));
}
async function loadVideo(duration = 2.5, file = new File(['video'], 'drive.mp4', { type: 'video/mp4' })) {
    openEditor();
    fireEvent.change(screen.getByLabelText('Recorded video'), { target: { files: [file] } });
    const video = await screen.findByLabelText('Original recorded video') as HTMLVideoElement;
    mockVideo(video, duration);
    fireEvent.loadedMetadata(video); fireEvent.loadedData(video);
    return video;
}
async function loadTelemetry(name = 'drive.jsonl') {
    const file = new File(['telemetry'], name);
    Object.defineProperty(file, 'stream', { value: () => telemetryStream(name) });
    fireEvent.change(screen.getByLabelText('Telemetry JSONL'), { target: { files: [file] } });
    await waitFor(() => expect(screen.queryByText('Reading telemetry…')).not.toBeInTheDocument());
}
function savedAnnotation(name: string, videoFileName: string, telemetryFileName: string,
    offsetSeconds: number | undefined, ranges: TrainingSelection['ranges']): TrainingAnnotation {
    return { id: name, name, videoUrl: `acla-training-media://recording/${videoFileName}`,
        telemetryUrl: `acla-training-media://recording/${telemetryFileName}`,
        selection: { version: 2, videoFileName, telemetryFileName, ranges, ...(offsetSeconds === undefined ? {} : { offsetSeconds }) } };
}
function mockAnnotationPicker(annotations: TrainingAnnotation[]) {
    const open = jest.fn(async (file: File) => {
        const annotation = annotations.find((item) => item.name === file.name);
        if (!annotation) throw new Error('Put the video in the same folder as this annotation.');
        return annotation;
    });
    window.recordedTrainingRanges = { read: jest.fn(async () => null), write: jest.fn(), update: jest.fn(), open, scan: jest.fn() };
    window.fetch = jest.fn(async (url: string) => ({ ok: true, body: telemetryStream(url.split('/').pop()!) })) as jest.Mock;
    return open;
}
function pickAnnotations(...names: string[]) {
    fireEvent.change(screen.getByLabelText('Edit saved annotations'), { target: {
        files: names.map((name) => new File(['{}'], name)),
    } });
}
function folderFile(relativePath: string) {
    const file = new File([], relativePath.split('/').pop()!);
    Object.defineProperty(file, 'webkitRelativePath', { value: relativePath });
    return file;
}
async function loadAnnotationPreview(duration: number) {
    const preview = await screen.findByLabelText('Original recorded video') as HTMLVideoElement;
    mockVideo(preview, duration);
    fireEvent.loadedMetadata(preview); fireEvent.loadedData(preview);
    await waitFor(() => expect(screen.queryByText('Reading telemetry…')).not.toBeInTheDocument());
    return preview;
}

it('pairs local frames using timing saved in the video and revokes its video URL', async () => {
    const onFrame = jest.fn();
    const view = render(<RecordedFrameViewer configuration={cloneConfig()} policy={null} disabled={false} onFrame={onFrame} />);
    const timing = encodeVideoTelemetryStart(100.125);
    const file = new File(['video', timing], 'drive.mp4', { type: 'video/mp4' });
    const readTail = jest.spyOn(file, 'slice').mockReturnValue({ arrayBuffer: async () => timing.buffer } as Blob);
    const video = await loadVideo(2.5, file);
    await loadTelemetry();
    expect(parse).toHaveBeenCalledWith(telemetryStream('drive.jsonl'), 'acc', expect.any(AbortSignal));
    expect(readTail).toHaveBeenCalledWith(-VIDEO_TIMING_BYTES);
    expect(onFrame).toHaveBeenLastCalledWith(expect.objectContaining({ session: recording, videoSeconds: 0, telemetrySeconds: 100.125,
        camera: expect.objectContaining({ width: 384, height: 216, rgb: expect.any(Uint8Array) }) }));
    const timeline = within(screen.getByRole('region', { name: 'Recorded telemetry timeline' }));
    expect(timeline.getByLabelText('Throttle at playhead')).toHaveTextContent('45%');
    expect(timeline.getByLabelText('Brake at playhead')).toHaveTextContent('30%');
    expect(timeline.getByLabelText('Gear at playhead')).toHaveTextContent('2');
    expect(screen.queryByLabelText('Telemetry offset (ms)')).not.toBeInTheDocument();
    await act(async () => { video.currentTime = 0.125; });
    expect(onFrame).toHaveBeenLastCalledWith(expect.objectContaining({ telemetrySeconds: 100.25 }));
    expect(timeline.getByLabelText('Throttle at playhead')).toHaveTextContent('70%');
    expect(timeline.getByLabelText('Gear at playhead')).toHaveTextContent('3');
    expect(screen.getByRole('button', { name: 'Inspect current frame' })).toBeDisabled();
    view.unmount();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:recorded-video');
    expect(onFrame).toHaveBeenLastCalledWith(null);
});

it('changes recorded video playback speed without moving the playhead and keeps it across videos', async () => {
    jest.mocked(URL.createObjectURL).mockImplementation((file) => `blob:${(file as File).name}`);
    render(<RecordedFrameViewer configuration={cloneConfig()} policy={null} disabled={false} onFrame={jest.fn()} />);
    const video = await loadVideo();
    const slider = screen.getByRole('slider', { name: 'Playback speed' });
    expect(slider).toHaveAttribute('min', '0.25');
    expect(slider).toHaveAttribute('max', '2');
    expect(slider).toHaveAttribute('step', '0.25');
    expect(slider).toHaveValue('1');
    expect(video.playbackRate).toBe(1);
    await act(async () => { video.currentTime = 0.75; });

    fireEvent.change(slider, { target: { value: '0.5' } });
    expect(video.playbackRate).toBe(0.5);
    expect(video.currentTime).toBe(0.75);
    fireEvent.change(slider, { target: { value: '2' } });
    expect(video.playbackRate).toBe(2);
    expect(video.currentTime).toBe(0.75);

    const next = new File(['video'], 'next.mp4', { type: 'video/mp4' });
    fireEvent.change(screen.getByLabelText('Recorded video'), { target: { files: [next] } });
    fireEvent.change(screen.getByLabelText('Video in folder'), { target: { value: '1' } });
    const replacement = screen.getByLabelText('Original recorded video') as HTMLVideoElement;
    expect(replacement).toHaveAttribute('src', 'blob:next.mp4');
    expect(replacement.playbackRate).toBe(2);
    expect(screen.getByRole('slider', { name: 'Playback speed' })).toHaveValue('2');

    act(() => { replacement.playbackRate = 1.25; });
    expect(screen.getByRole('slider', { name: 'Playback speed' })).toHaveValue('1.25');
});

it('still seeks across the recording on a quick press, a regular drag or a slider value change', async () => {
    render(<RecordedFrameViewer configuration={cloneConfig()} policy={null} disabled={false} onFrame={jest.fn()} />);
    const video = await loadVideo(600);
    const timeline = mockTimeline();

    await act(async () => {
        pointer(timeline, 'pointerdown', 354);
        pointer(timeline, 'pointerup', 354);
    });
    expect(video.currentTime).toBeCloseTo(150, 1);

    await act(async () => {
        pointer(timeline, 'pointerdown', 354);
        pointer(timeline, 'pointermove', 846);
        pointer(timeline, 'pointerup', 846);
    });
    expect(video.currentTime).toBeCloseTo(450, 1);
    expect(screen.queryByText(/Fine adjustment active/)).not.toBeInTheDocument();

    await act(async () => { fireEvent.change(timeline, { target: { value: '450.01' } }); });
    expect(video.currentTime).toBeCloseTo(450.01, 2);
});

it('keeps the telemetry plot, full recording progress and video in sync while zooming and scrubbing', async () => {
    parse.mockResolvedValue({ ...recording, durationSeconds: 599, rows: [
        recordedRow(0, 0, 0, 1, 2), recordedRow(1, 0.125, 0.1, 0.9, 3),
        recordedRow(2, 225, 0.8, 0.2, 4), recordedRow(3, 300, 0.4, 0.6, 3), recordedRow(4, 599, 1, 0, 5),
    ] });
    render(<RecordedFrameViewer configuration={cloneConfig()} policy={null} disabled={false} onFrame={jest.fn()} />);
    const video = await loadVideo(600);
    await loadTelemetry();
    const timeline = within(screen.getByRole('region', { name: 'Recorded telemetry timeline' }));
    const plot = timeline.getByTestId('recorded-telemetry-plot');
    const playhead = timeline.getByTestId('recorded-telemetry-playhead');
    const progress = screen.getByRole('slider', { name: /Recorded frame/ });
    const zoomIn = timeline.getByRole('button', { name: 'Zoom in telemetry timeline' });
    const zoomOut = timeline.getByRole('button', { name: 'Zoom out telemetry timeline' });
    const fit = timeline.getByRole('button', { name: 'Fit full recording' });
    jest.spyOn(plot, 'getBoundingClientRect').mockReturnValue({
        x: 100, y: 0, top: 0, right: 1100, bottom: 200, left: 100, width: 1000, height: 200, toJSON: () => ({}),
    });
    expect(timeline.getByText('Throttle')).toBeVisible();
    expect(timeline.getByText('Brake')).toBeVisible();
    expect(timeline.getByText('Gear')).toBeVisible();
    for (const channel of ['throttle', 'brake', 'gear']) {
        expect(timeline.getByTestId(`recorded-telemetry-${channel}-trace`)).toHaveAttribute('d', expect.stringContaining('M'));
    }
    expect(zoomOut).toBeDisabled();

    await act(async () => {
        pointer(plot, 'pointerdown', 600);
        pointer(plot, 'pointerup', 600);
    });
    expect(video.currentTime).toBeCloseTo(300, 2);
    expect(progress).toHaveValue('300');
    expect(parseFloat(playhead.style.left)).toBeCloseTo(50, 2);

    fireEvent.click(zoomIn);
    expect(video.currentTime).toBeCloseTo(300, 2);
    expect(timeline.getByText('2×')).toBeVisible();
    expect(zoomOut).toBeEnabled();
    await act(async () => {
        pointer(plot, 'pointerdown', 550);
        pointer(plot, 'pointermove', 350);
        pointer(plot, 'pointerup', 350);
    });
    expect(video.currentTime).toBeCloseTo(225, 2);
    expect(progress).toHaveValue('225');
    expect(parseFloat(playhead.style.left)).toBeCloseTo(25, 2);

    video.currentTime = 375;
    fireEvent.timeUpdate(video);
    await waitFor(() => expect(progress).toHaveValue('375'));
    expect(parseFloat(playhead.style.left)).toBeCloseTo(75, 2);

    fireEvent.change(progress, { target: { value: '500' } });
    expect(video.currentTime).toBe(500);
    expect(parseFloat(playhead.style.left)).toBeCloseTo(100 * 2 / 3, 2);
    fireEvent.click(timeline.getByText('6:15.00'), { clientX: 350 });
    expect(video.currentTime).toBe(375);
    expect(progress).toHaveValue('375');

    fireEvent.click(zoomOut);
    expect(timeline.getByText('1×')).toBeVisible();
    expect(video.currentTime).toBe(375);
    fireEvent.click(zoomIn);
    fireEvent.click(fit);
    expect(zoomOut).toBeDisabled();
    expect(progress).toHaveValue('375');
    expect(parseFloat(playhead.style.left)).toBeCloseTo(62.5, 2);
});

it.each(['pointerup', 'pointercancel'])('fine-tunes the playhead after a hold and exits on %s', async (end) => {
    render(<RecordedFrameViewer configuration={cloneConfig()} policy={null} disabled={false} onFrame={jest.fn()} />);
    const video = await loadVideo(600);
    const timeline = mockTimeline();
    expect(timeline).toHaveAccessibleDescription('Hold the progress bar, then drag to fine-tune.');

    jest.useFakeTimers();
    try {
        await act(async () => { pointer(timeline, 'pointerdown', 600, 'touch'); });
        const start = video.currentTime;
        expect(start).toBeCloseTo(300, 1);
        act(() => { jest.advanceTimersByTime(349); });
        expect(screen.queryByText(/Fine adjustment active/)).not.toBeInTheDocument();
        act(() => { jest.advanceTimersByTime(1); });
        expect(screen.getByText(/Fine adjustment active/)).toBeInTheDocument();
        expect(video.currentTime).toBe(start);

        await act(async () => { pointer(timeline, 'pointermove', 620, 'touch'); });
        expect(video.currentTime).toBeCloseTo(start + 0.1, 2);
        await act(async () => { pointer(timeline, 'pointermove', 580, 'touch'); });
        expect(video.currentTime).toBeCloseTo(start - 0.1, 2);

        await act(async () => { pointer(timeline, end, 580, 'touch'); });
        expect(screen.queryByText(/Fine adjustment active/)).not.toBeInTheDocument();
        await act(async () => { pointer(timeline, 'pointermove', 750, 'touch'); });
        expect(video.currentTime).toBeCloseTo(start - 0.1, 2);
    } finally {
        jest.useRealTimers();
    }
});

it.each([[108, 88, 0], [1092, 1112, 599.999]])('keeps a fine adjustment from %s px within the recording', async (startX, outsideX, limit) => {
    render(<RecordedFrameViewer configuration={cloneConfig()} policy={null} disabled={false} onFrame={jest.fn()} />);
    const video = await loadVideo(600);
    const timeline = mockTimeline();

    jest.useFakeTimers();
    try {
        await act(async () => { pointer(timeline, 'pointerdown', startX); });
        act(() => { jest.advanceTimersByTime(350); });
        expect(screen.getByText(/Fine adjustment active/)).toBeInTheDocument();

        await act(async () => { pointer(timeline, 'pointermove', outsideX); });
        expect(video.currentTime).toBeCloseTo(limit, 3);
        await act(async () => { pointer(timeline, 'pointermove', startX); });
        expect(video.currentTime).toBeCloseTo(limit + Math.sign(startX - outsideX) * 0.1, 2);
        pointer(timeline, 'pointerup', startX);
    } finally {
        jest.useRealTimers();
    }
});

it('previews, saves and replays an edited offset for a legacy annotation and supports resetting to zero', async () => {
    const ranges = [{ start: 0.1, end: 0.3, label: 'Qualifying' }];
    const annotation = savedAnnotation('drive.json', 'drive.mp4', 'drive.jsonl', undefined, ranges);
    mockAnnotationPicker([annotation]);
    const update = jest.mocked(window.recordedTrainingRanges!.update).mockImplementation(async (_, draft) => ({
        ...annotation, selection: { ...annotation.selection, ...draft },
    }));
    const onFrame = jest.fn();
    const ref = React.createRef<RecordedFrameViewerHandle>();
    render(<RecordedFrameViewer ref={ref} configuration={cloneConfig()} policy={null} disabled={false} onFrame={onFrame} />);
    pickAnnotations(annotation.name);
    const video = await loadAnnotationPreview(2.5);
    await act(async () => { video.currentTime = 0.2; });
    const offset = screen.getByRole('spinbutton', { name: 'Telemetry offset (ms)' });
    expect(offset).toHaveValue(null);
    expect(offset).toHaveAccessibleDescription('Positive values read earlier telemetry.');
    expect(onFrame.mock.lastCall?.[0].telemetrySeconds).toBeCloseTo(100.2, 8);

    const timeline = within(screen.getByRole('region', { name: 'Recorded telemetry timeline' }));
    await act(async () => { video.currentTime = 0.25; });
    expect(timeline.getByLabelText('Throttle at playhead')).toHaveTextContent('70%');
    fireEvent.change(offset, { target: { value: '125' } });
    expect(timeline.getByLabelText('Throttle at playhead')).toHaveTextContent('45%');
    await act(async () => { video.currentTime = 0.2; });

    fireEvent.change(offset, { target: { value: '93' } });
    expect(onFrame.mock.lastCall?.[0].telemetrySeconds).toBeCloseTo(100.107, 8);
    const save = screen.getByRole('button', { name: 'Save training annotation' });
    fireEvent.click(save);
    await screen.findByText('Saved drive.json beside the video.');
    expect(update).toHaveBeenLastCalledWith(annotation.id, { ranges, offsetSeconds: 0.093 });

    const consume = jest.fn(async (_frame: RecordedReplayFrame) => undefined);
    await act(async () => ref.current!.measure(consume, new AbortController().signal));
    expect(consume.mock.calls[0][0].telemetrySeconds).toBeCloseTo(100.007, 8);

    fireEvent.change(offset, { target: { value: '0' } });
    fireEvent.click(save);
    await screen.findByText('Saved drive.json beside the video.');
    expect(update).toHaveBeenLastCalledWith(annotation.id, { ranges, offsetSeconds: 0 });
    expect(offset).toHaveValue(0);
});

it('allows negative manual alignment when embedded timing uses a different clock', async () => {
    const onFrame = jest.fn();
    parse.mockResolvedValue({ ...recording, timestampField: 'Graphics_game_time' });
    render(<RecordedFrameViewer configuration={cloneConfig()} policy={null} disabled={false} onFrame={onFrame} />);
    const timing = encodeVideoTelemetryStart(100.125);
    const file = new File(['video', timing], 'drive.mp4', { type: 'video/mp4' });
    jest.spyOn(file, 'slice').mockReturnValue({ arrayBuffer: async () => timing.buffer } as Blob);
    const video = await loadVideo(2.5, file);
    await loadTelemetry();
    await act(async () => { video.currentTime = 0.1; });
    expect(onFrame.mock.lastCall?.[0].telemetrySeconds).toBeCloseTo(100.1, 8);
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Telemetry offset (ms)' }), { target: { value: '-25' } });
    expect(onFrame.mock.lastCall?.[0].telemetrySeconds).toBeCloseTo(100.125, 8);
});

it('does not apply a saved or manually edited offset to replacement telemetry', async () => {
    const first = savedAnnotation('drive.json', 'drive.mp4', 'drive.jsonl', 0.093, null);
    const write = jest.fn(async (video: File, telemetry: File, draft: { offsetSeconds?: number }) =>
        savedAnnotation('drive.json', video.name, telemetry.name, draft.offsetSeconds, null));
    window.recordedTrainingRanges = { read: jest.fn(async () => first.selection), write, update: jest.fn(), open: jest.fn(), scan: jest.fn() };
    const onFrame = jest.fn();
    render(<RecordedFrameViewer configuration={cloneConfig()} policy={null} disabled={false} onFrame={onFrame} />);
    const video = await loadVideo();
    await loadTelemetry();
    await act(async () => { video.currentTime = 0.2; });
    const offset = () => screen.getByRole('spinbutton', { name: 'Telemetry offset (ms)' });
    expect(offset()).toHaveValue(93);
    fireEvent.change(offset(), { target: { value: '110' } });
    expect(onFrame.mock.lastCall?.[0].telemetrySeconds).toBeCloseTo(100.09, 8);

    await loadTelemetry('alternative.jsonl');
    expect(offset()).toHaveValue(null);
    expect(onFrame.mock.lastCall?.[0].telemetrySeconds).toBeCloseTo(100.2, 8);
    fireEvent.click(screen.getByRole('button', { name: 'Save training annotation' }));
    await screen.findByText('Saved drive.json beside the video.');
    expect(write).toHaveBeenLastCalledWith(expect.any(File), expect.objectContaining({ name: 'alternative.jsonl' }), { ranges: null });
});

it('requires explicit file choices when a folder contains multiple recordings', () => {
    render(<RecordedFrameViewer configuration={cloneConfig()} policy={null} disabled={false} onFrame={() => undefined} />);
    openEditor();
    fireEvent.change(screen.getByLabelText('Working folder'), { target: { files: [
        new File(['a'], 'a.mp4'), new File(['b'], 'b.mp4'), new File(['a'], 'a.jsonl'), new File(['b'], 'b.jsonl'),
    ] } });
    expect(screen.getByLabelText('Video in folder')).toHaveValue('');
    expect(screen.getByLabelText('Telemetry in folder')).toHaveValue('');
    expect(screen.queryByLabelText('Original recorded video')).not.toBeInTheDocument();
    expect(parse).not.toHaveBeenCalled();
});

it('keeps missing-folder-file errors visible after loading the video and clears them when the pair is complete', async () => {
    render(<RecordedFrameViewer configuration={cloneConfig()} policy={null} disabled={false} onFrame={() => undefined} />);
    openEditor();
    fireEvent.change(screen.getByLabelText('Working folder'), { target: { files: [new File(['video'], 'drive.mp4')] } });
    await screen.findByLabelText('Original recorded video');
    expect(screen.getByRole('alert')).toHaveTextContent('select the missing file individually');
    await loadTelemetry();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
});

it('discovers pairs from the working folder before opening an editor and updates the training list when rescanned', async () => {
    const first = savedAnnotation('drive.mp4.training-ranges.json', 'drive.mp4', 'drive.jsonl', 0, null);
    const second = savedAnnotation('turns.json', 'track.webm', 'controls.jsonl', 0.1, [{ start: 0.2, end: 0.4 }]);
    const open = mockAnnotationPicker([]);
    const scan = jest.mocked(window.recordedTrainingRanges!.scan);
    scan.mockResolvedValueOnce([first, second, { error: 'Missing telemetry.' }, null]).mockResolvedValueOnce([first]);
    const firstJson = folderFile(`practice/${first.name}`);
    const secondJson = folderFile(`practice/stint/${second.name}`);
    const missingJson = folderFile('practice/stint/broken.mp4.training-ranges.json');
    const otherJson = folderFile('practice/settings.json');
    const video = folderFile('practice/drive.mp4');
    const telemetry = folderFile('practice/drive.jsonl');
    const readTelemetry = jest.fn(() => telemetryStream(telemetry.name));
    Object.defineProperty(telemetry, 'stream', { value: readTelemetry });
    const onReplayReadyChange = jest.fn();
    render(<RecordedFrameViewer configuration={cloneConfig()} policy={null} disabled={false}
        onFrame={jest.fn()} onReplayReadyChange={onReplayReadyChange} />);
    const folder = screen.getByLabelText('Working folder');
    expect(folder).toHaveAttribute('webkitdirectory');
    fireEvent.change(folder, { target: { files: [firstJson, secondJson, missingJson, otherJson, video, telemetry,
        folderFile('practice/stint/track.webm'), folderFile('practice/stint/broken.mp4'), folderFile('practice/other/track.webm'),
    ] } });

    await screen.findByText('practice · 2 annotated recording pairs found');
    expect(scan).toHaveBeenCalledWith([firstJson, secondJson, missingJson, otherJson]);
    expect(open).not.toHaveBeenCalled();
    expect(window.fetch).not.toHaveBeenCalled();
    expect(screen.getByText('track.webm · controls.jsonl')).toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveTextContent('stint/broken.mp4.training-ranges.json: Missing telemetry.');
    expect(onReplayReadyChange).toHaveBeenLastCalledWith(true);
    fireEvent.click(screen.getByRole('checkbox', { name: /^drive\.mp4\.training-ranges\.json/ }));
    openEditor();
    expect(within(screen.getByLabelText('Video in folder')).getAllByRole('option').map((option) => option.textContent))
        .toEqual(['Select a video to review', 'practice/other/track.webm']);
    expect(readTelemetry).not.toHaveBeenCalled();

    fireEvent.change(folder, { target: { files: [firstJson, video, telemetry] } });
    await screen.findByText('practice · 1 annotated recording pair found');
    expect(screen.getByRole('checkbox', { name: /^drive\.mp4\.training-ranges\.json/ })).not.toBeChecked();
    expect(screen.queryByRole('button', { name: 'Edit stint/turns.json' })).not.toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Video in folder')).not.toBeInTheDocument();
    expect(onReplayReadyChange).toHaveBeenLastCalledWith(false);
});

it('removes a folder video from the add-new picker as soon as its annotation is saved', async () => {
    mockAnnotationPicker([]);
    const bridge = window.recordedTrainingRanges!;
    jest.mocked(bridge.write).mockResolvedValue(savedAnnotation('drive.mp4.training-ranges.json', 'drive.mp4', 'drive.jsonl', 0, null));
    const telemetry = folderFile('practice/drive.jsonl');
    Object.defineProperty(telemetry, 'stream', { value: () => telemetryStream(telemetry.name) });
    render(<RecordedFrameViewer configuration={cloneConfig()} policy={null} disabled={false} onFrame={jest.fn()} />);
    fireEvent.change(screen.getByLabelText('Working folder'), { target: { files: [folderFile('practice/drive.mp4'), telemetry] } });
    await screen.findByText('practice · 0 annotated recording pairs found');
    openEditor();
    await loadAnnotationPreview(2.5);
    fireEvent.click(screen.getByRole('button', { name: 'Save training annotation' }));

    await screen.findByText('Saved drive.mp4.training-ranges.json beside the video.');
    expect(screen.getByRole('button', { name: 'Edit drive.mp4.training-ranges.json' })).toHaveAttribute('aria-pressed', 'true');
    openEditor();
    expect(screen.queryByLabelText('Video in folder')).not.toBeInTheDocument();
    expect(screen.getByText('All videos in the working folder already have training annotations.')).toBeVisible();
});

it('shows actual feature-map values and ranks real sampled frames for every channel', async () => {
    const inspect = jest.fn().mockResolvedValueOnce(result(2)).mockResolvedValueOnce(result(1))
        .mockResolvedValueOnce(result(3)).mockResolvedValueOnce(result(2));
    const model = { steps: 20, inspectCameraAsync: inspect } as unknown as DrivingPolicy;
    render(<RecordedFrameViewer configuration={cloneConfig()} policy={model} disabled={false} onFrame={() => undefined} />);
    await loadVideo();
    fireEvent.click(screen.getByRole('button', { name: 'Inspect current frame' }));
    await screen.findByLabelText('Layer 1, channel 1 activation map');
    expect(screen.getByText(/Mean 1.50000 · Peak 2.00000/)).toBeInTheDocument();
    expect(screen.getByAltText('Original recorded frame at 0:00.00')).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Feature channel'), { target: { value: '1' } });
    expect(screen.getByLabelText('Layer 1, channel 2 activation map')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Find strongest examples' }));
    await screen.findByText(/Scan complete · Layer 1 · 3 frames sampled every 1 seconds/);
    const strongest = within(screen.getByLabelText('Strongest examples for channel 1')).getAllByRole('button');
    expect(strongest.map((button) => button.textContent)).toEqual(['0:01.00 · 3.00000', '0:02.00 · 2.00000', '0:00.00 · 1.00000']);
    expect(within(screen.getByLabelText('Strongest examples for channel 2')).getByText('No positive activation in sampled frames.')).toBeInTheDocument();
    expect(inspect).toHaveBeenCalledTimes(4);
    await loadTelemetry();
    expect(screen.queryByLabelText('Strongest examples for channel 1')).not.toBeInTheDocument();
    expect(screen.queryByAltText('Original recorded frame at 0:00.00')).not.toBeInTheDocument();
});

it('replays only the overlap, awaits each training consumer, and stops on cancellation', async () => {
    const annotation = savedAnnotation('drive.json', 'drive.mp4', 'drive.jsonl', 0.25, null);
    mockAnnotationPicker([{ ...annotation, videoTelemetryStartSeconds: 99.5,
        selection: { ...annotation.selection, fullRecordingLabel: 'Warmup' } }]);
    const ref = React.createRef<RecordedFrameViewerHandle>();
    const onBusyChange = jest.fn();
    render(<RecordedFrameViewer ref={ref} configuration={cloneConfig()} policy={null} disabled={false}
        onFrame={() => undefined} onBusyChange={onBusyChange} />);
    pickAnnotations('drive.json');
    await loadAnnotationPreview(2.5);
    const abort = new AbortController();
    let finish!: () => void;
    const consume = jest.fn(() => new Promise<void>((resolve) => { finish = resolve; }));
    const onProgress = jest.fn();
    let pending!: Promise<void>;
    await act(async () => { pending = ref.current!.replay(consume, abort.signal, onProgress); await Promise.resolve(); });
    expect(consume).toHaveBeenCalledTimes(1);
    expect(consume).toHaveBeenCalledWith(expect.objectContaining({ videoSeconds: 0.5, telemetrySeconds: 100,
        recordingId: annotation.id, clipLabel: 'Warmup' }));
    expect(onBusyChange).toHaveBeenLastCalledWith(true);
    await act(async () => {
        abort.abort(); finish();
        await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    });
    expect(consume).toHaveBeenCalledTimes(1);
    expect(onProgress).toHaveBeenCalledTimes(1);
    expect(onProgress).toHaveBeenCalledWith(expect.objectContaining({ processedFrames: 0 }));
    expect(onBusyChange).toHaveBeenLastCalledWith(false);
});

it('marks, edits and removes saved ranges on the timeline, and restores each video’s selection', async () => {
    let stored: TrainingAnnotation | undefined;
    const write = jest.fn(async (video: File, telemetry: File, { ranges }: Pick<TrainingSelection, 'ranges'>) => {
        stored = savedAnnotation('drive.mp4.training-ranges.json', video.name, telemetry.name, undefined, ranges);
        return stored;
    });
    window.recordedTrainingRanges = { read: jest.fn(async () => stored?.selection ?? null), write, update: jest.fn(), open: jest.fn(), scan: jest.fn() };
    const props = { configuration: cloneConfig(), policy: null, disabled: false, onFrame: jest.fn() };
    const view = render(<RecordedFrameViewer {...props} />);
    const video = await loadVideo();
    await loadTelemetry();
    const start = () => screen.getByLabelText('Range start (seconds)');
    const end = () => screen.getByLabelText('Range end (seconds)');
    const label = () => screen.getByRole('combobox', { name: 'Clip label (optional)' });
    expect(within(label()).getAllByRole<HTMLOptionElement>('option').map((option) => option.value)).toEqual(['', ...trainingLabels]);
    const fullLabel = screen.getByRole('combobox', { name: 'Full recording label (optional)' });
    for (const select of [label(), fullLabel]) {
        expect(within(select).queryByRole('option', { name: 'Qualifying' })).not.toBeInTheDocument();
        expect(within(select).queryByRole('option', { name: 'Recovery' })).not.toBeInTheDocument();
        expect(within(select).queryByRole('option', { name: 'Race pace' })).not.toBeInTheDocument();
        expect(within(select).queryByRole('option', { name: 'Shifting' })).not.toBeInTheDocument();
        expect(within(select).queryByRole('option', { name: 'Cornering' })).not.toBeInTheDocument();
        expect(within(select).getByRole('option', { name: 'Recover off track' })).toBeInTheDocument();
        expect(within(select).getByRole('option', { name: 'Shift down' })).toBeEnabled();
        expect(within(select).getByRole('option', { name: 'Shift up' })).toBeEnabled();
        expect(within(select).getByRole('option', { name: 'Switch lane left' })).toBeEnabled();
        expect(within(select).getByRole('option', { name: 'Switch lane right' })).toBeEnabled();
    }
    fireEvent.click(screen.getByRole('button', { name: 'Mark start here' }));
    await act(async () => { video.currentTime = 0.4; });
    fireEvent.click(screen.getByRole('button', { name: 'Mark end here' }));
    fireEvent.change(label(), { target: { value: 'Hairpin' } });
    fireEvent.click(screen.getByRole('button', { name: 'Add range' }));
    expect(label()).toHaveValue('');
    fireEvent.click(screen.getByRole('button', { name: 'Mark start here' }));
    fireEvent.change(end(), { target: { value: '1.5' } });
    fireEvent.click(screen.getByRole('button', { name: 'Add range' }));
    const timeline = () => within(screen.getByRole('region', { name: 'Label timeline' }));
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    await act(async () => { fireEvent.click(timeline().getByRole('button', { name: 'Range 2: Unlabeled, 0:00.40 – 0:01.50' })); });
    fireEvent.click(screen.getByRole('button', { name: 'Edit range 2' }));
    expect(start()).toHaveValue(0.4);
    fireEvent.change(start(), { target: { value: '0.3' } });
    fireEvent.click(screen.getByRole('button', { name: 'Update range' }));
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    const save = () => screen.getByRole('button', { name: 'Save training annotation' });
    fireEvent.click(save());
    await waitFor(() => expect(screen.queryByText('Unsaved annotation changes')).not.toBeInTheDocument());
    expect(write).toHaveBeenCalledWith(expect.objectContaining({ name: 'drive.mp4' }), expect.objectContaining({ name: 'drive.jsonl' }), {
        ranges: [{ start: 0, end: 0.4, label: 'Hairpin' }, { start: 0.3, end: 1.5 }] });
    expect(screen.getByRole('checkbox', { name: /drive.mp4.training-ranges.json/ })).toBeChecked();
    view.unmount();

    render(<RecordedFrameViewer {...props} />);
    const reopenedVideo = await loadVideo();
    await loadTelemetry();
    const hairpin = timeline().getByRole('button', { name: 'Range 1: Hairpin, 0:00.00 – 0:00.40' });
    const unlabeled = timeline().getByRole('button', { name: 'Range 2: Unlabeled, 0:00.30 – 0:01.50' });
    expect(hairpin).toBeVisible();
    expect(parseFloat(hairpin.style.left)).toBe(0);
    expect(parseFloat(hairpin.style.width)).toBeCloseTo(16);
    expect(parseFloat(unlabeled.style.left)).toBeCloseTo(12);
    expect(parseFloat(unlabeled.style.width)).toBeCloseTo(48);

    await act(async () => { fireEvent.click(unlabeled); });
    expect(unlabeled).toHaveAttribute('aria-pressed', 'true');
    expect(reopenedVideo.currentTime).toBeCloseTo(0.3);
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Jump to end of range 2' })); });
    expect(reopenedVideo.currentTime).toBeCloseTo(1.5);
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Jump to start of range 2' })); });
    expect(reopenedVideo.currentTime).toBeCloseTo(0.3);
    fireEvent.click(screen.getByRole('button', { name: 'Remove range 2' }));
    await act(async () => { fireEvent.click(hairpin); });
    fireEvent.click(screen.getByRole('button', { name: 'Remove range 1' }));
    expect(screen.getByText(/No footage selected/)).toBeInTheDocument();
    fireEvent.click(save());
    await waitFor(() => expect(screen.queryByText('Unsaved annotation changes')).not.toBeInTheDocument());
    expect(write).toHaveBeenLastCalledWith(expect.any(File), expect.any(File), { ranges: [] });
});

it('labels gear changes that overlap an existing clip and saves them in order', async () => {
    const hairpin = { start: 0.5, end: 1.75, label: 'Hairpin' };
    const annotation = { ...savedAnnotation('drive.json', 'drive.mp4', 'drive.jsonl', undefined, [hairpin]), videoTelemetryStartSeconds: 99.5 };
    mockAnnotationPicker([annotation]);
    const update = jest.mocked(window.recordedTrainingRanges!.update).mockImplementation(async (_, draft) => ({
        ...annotation, selection: { ...annotation.selection, ranges: draft.ranges },
    }));
    const rows = [[0, 3], [0.125, 4], [0.25, 4], [0.5, 4], [0.625, 3], [0.9375, 3], [1, 4], [1.5, 4]]
        .map(([timeSeconds, gear], index) => ({ timeSeconds, index, segment: 0,
            sample: { Graphics_status: 2, Physics_gear: gear } }));
    parse.mockResolvedValue({ ...recording, rows, durationSeconds: 1.5 });
    const configuration = cloneConfig();
    configuration.runtime.controlIntervalMs = 125;
    render(<RecordedFrameViewer configuration={configuration} policy={null} disabled={false} onFrame={jest.fn()} />);
    pickAnnotations(annotation.name);
    const video = await loadAnnotationPreview(2.5);
    const timeline = within(screen.getByRole('region', { name: 'Label timeline' }));
    const addShiftDown = screen.getByRole('button', { name: 'Add shift down' });
    const addShiftUp = screen.getByRole('button', { name: 'Add shift up' });

    expect(video.currentTime).toBe(0);
    await act(async () => { fireEvent.click(addShiftDown); });
    expect(video.currentTime).toBe(1.125);
    expect(timeline.getByRole('button', { name: 'Range 2: Shift down, 0:01.00 – 0:01.13' })).toBeVisible();
    expect(screen.getByText('Unsaved annotation changes')).toBeVisible();
    expect(update).not.toHaveBeenCalled();

    fireEvent.click(addShiftDown);
    expect(screen.getByRole('alert')).toHaveTextContent('No unmarked shift down');
    expect(timeline.getAllByRole('button', { name: /Shift down/ })).toHaveLength(1);

    await act(async () => { fireEvent.click(addShiftUp); });
    expect(video.currentTime).toBe(1.5);
    expect(timeline.getByRole('button', { name: 'Range 3: Shift up, 0:01.44 – 0:01.56' })).toBeVisible();

    await act(async () => { fireEvent.click(addShiftUp); });
    expect(video.currentTime).toBe(0.625);
    expect(timeline.getByRole('button', { name: 'Range 2: Shift up, 0:00.50 – 0:00.63' })).toBeVisible();
    expect(timeline.getByRole('button', { name: 'Range 3: Shift down, 0:01.00 – 0:01.13' })).toBeVisible();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Save training annotation' }));
    await screen.findByText('Saved drive.json beside the video.');
    expect(update).toHaveBeenCalledWith(annotation.id, { ranges: [
        hairpin,
        { start: 0.5, end: 0.625, label: 'Shift up' },
        { start: 1, end: 1.125, label: 'Shift down' },
        { start: 1.4375, end: 1.5625, label: 'Shift up' },
    ] });
});

it('groups clips by label while keeping identical ranges independently selectable', async () => {
    const annotation = savedAnnotation('drive.json', 'drive.mp4', 'drive.jsonl', 0, [
        { start: 0.1, end: 0.3, label: 'Hairpin' },
        { start: 0.1, end: 0.3, label: 'Hairpin' },
        { start: 0.4, end: 0.6, label: 'Shift down' },
        { start: 1.1, end: 1.4, label: 'Hairpin' },
    ]);
    mockAnnotationPicker([annotation]);
    render(<RecordedFrameViewer configuration={cloneConfig()} policy={null} disabled={false} onFrame={jest.fn()} />);
    pickAnnotations(annotation.name);
    await loadAnnotationPreview(2.5);
    const timeline = within(screen.getByRole('region', { name: 'Label timeline' }));
    const firstCorner = timeline.getByRole('button', { name: /^Range 1: Hairpin/ });
    const duplicate = timeline.getByRole('button', { name: /^Range 2: Hairpin/ });
    const laterCorner = timeline.getByRole('button', { name: /^Range 4: Hairpin/ });
    const shift = timeline.getByRole('button', { name: /^Range 3: Shift down/ });

    const ruler = within(timeline.getByRole('group', { name: 'Elapsed video time' }));
    expect(ruler.getByText('0:02.50')).toBeVisible();
    expect(firstCorner.parentElement).toBe(duplicate.parentElement);
    expect(firstCorner.parentElement).toBe(laterCorner.parentElement);
    expect(firstCorner.parentElement).not.toBe(shift.parentElement);
    expect(firstCorner.style.getPropertyValue('--clip-row')).toBe('0');
    expect(duplicate.style.getPropertyValue('--clip-row')).toBe('1');
    expect(laterCorner.style.getPropertyValue('--clip-row')).toBe('0');

    await act(async () => { fireEvent.click(duplicate); });
    expect(duplicate).toHaveAttribute('aria-pressed', 'true');
    expect(firstCorner).toHaveAttribute('aria-pressed', 'false');
    fireEvent.click(screen.getByRole('button', { name: 'Edit range 2' }));
    fireEvent.change(screen.getByLabelText('Range end (seconds)'), { target: { value: '0.5' } });
    fireEvent.click(screen.getByRole('button', { name: 'Update range' }));
    expect(firstCorner).toHaveAccessibleName('Range 1: Hairpin, 0:00.10 – 0:00.30');
    expect(duplicate).toHaveAccessibleName('Range 2: Hairpin, 0:00.10 – 0:00.50');
    fireEvent.click(screen.getByRole('button', { name: 'Remove range 2' }));
    expect(firstCorner).toBeInTheDocument();
    expect(duplicate).not.toBeInTheDocument();
});

it('zooms and fits the clip timeline without changing clip times', async () => {
    const annotation = savedAnnotation('drive.json', 'drive.mp4', 'drive.jsonl', 0, [
        { start: 0.5, end: 1, label: 'Hairpin' },
    ]);
    mockAnnotationPicker([annotation]);
    render(<RecordedFrameViewer configuration={cloneConfig()} policy={null} disabled={false} onFrame={jest.fn()} />);
    pickAnnotations(annotation.name);
    await loadAnnotationPreview(2.5);
    const timeline = within(screen.getByRole('region', { name: 'Label timeline' }));
    const axis = timeline.getByRole('group', { name: 'Elapsed video time' });
    const ruler = within(axis);
    const viewport = timeline.getByRole('group', { name: 'Label timeline viewport' });
    const zoomIn = timeline.getByRole('button', { name: 'Zoom in timeline' });
    const zoomOut = timeline.getByRole('button', { name: 'Zoom out timeline' });
    const fit = timeline.getByRole('button', { name: 'Fit timeline to view' });
    const zoomLevel = timeline.getByLabelText('Timeline zoom level');
    const clip = timeline.getByRole('button', { name: 'Range 1: Hairpin, 0:00.50 – 0:01.00' });
    const grid = clip.closest<HTMLElement>('.recorded-label-timeline-grid');
    Object.defineProperty(viewport, 'clientWidth', { value: 600 });
    Object.defineProperties(axis, {
        offsetLeft: { value: 160 },
        offsetWidth: { get: () => 440 * Number(grid?.style.getPropertyValue('--timeline-zoom')) },
    });
    await act(async () => { fireEvent.click(clip); });

    expect(zoomLevel).toHaveTextContent('1×');
    expect(zoomOut).toBeDisabled();
    expect(fit).toBeDisabled();
    expect(ruler.queryByText('0:00.31')).not.toBeInTheDocument();
    fireEvent.click(zoomIn);
    expect(zoomLevel).toHaveTextContent('2×');
    expect(grid?.style.getPropertyValue('--timeline-zoom')).toBe('2');
    expect(ruler.getByText('0:00.31')).toBeVisible();
    expect(clip).toHaveStyle({ left: '20%', width: '20%' });
    expect(viewport.scrollLeft).toBeCloseTo(88);

    for (const level of [4, 8, 16, 32]) {
        fireEvent.click(zoomIn);
        expect(zoomLevel).toHaveTextContent(`${level}×`);
    }
    expect(zoomIn).toBeDisabled();
    fireEvent.click(zoomOut);
    expect(zoomLevel).toHaveTextContent('16×');
    fireEvent.click(fit);
    expect(zoomLevel).toHaveTextContent('1×');
    expect(zoomOut).toBeDisabled();
    expect(viewport.scrollLeft).toBe(0);
    expect(ruler.queryByText('0:00.31')).not.toBeInTheDocument();
});

it('updates a clip label in a saved annotation and preserves its training checkbox', async () => {
    const first = savedAnnotation('first.json', 'first.mp4', 'first.jsonl', 0, null);
    const prior = savedAnnotation('second.json', 'second.mp4', 'second.jsonl', 0.1,
        [{ start: 0.2, end: 0.4, label: 'Cornering' }]);
    const second: TrainingAnnotation = { ...prior, selection: { ...prior.selection, game: 'iracing' } };
    const resized = { ...second, selection: { ...second.selection, ranges: [{ start: 0.2, end: 0.5, label: 'Cornering' }] } };
    const edited = { ...second, selection: { ...second.selection, ranges: [{ start: 0.2, end: 0.5, label: 'Shift down' }] } };
    mockAnnotationPicker([first, second]);
    const update = jest.mocked(window.recordedTrainingRanges!.update).mockResolvedValueOnce(resized).mockResolvedValueOnce(edited);
    render(<RecordedFrameViewer configuration={cloneConfig()} policy={null} disabled={false} onFrame={jest.fn()} />);
    pickAnnotations(first.name, second.name);
    await loadAnnotationPreview(2.5);
    const secondCheckbox = screen.getByRole('checkbox', { name: /^second\.json/ });
    fireEvent.click(secondCheckbox);
    fireEvent.click(screen.getByRole('button', { name: 'Edit second.json' }));
    await loadAnnotationPreview(2.5);
    expect(screen.getByLabelText('Recorded video')).not.toBeVisible();
    const label = () => screen.getByRole('combobox', { name: 'Clip label (optional)' });
    const timeline = within(screen.getByRole('region', { name: 'Label timeline' }));
    const editRange = async () => {
        const legacyClip = timeline.getByRole('button', { name: /^Range 1: Cornering/ });
        await act(async () => { fireEvent.click(legacyClip); });
        fireEvent.click(screen.getByRole('button', { name: 'Edit range 1' }));
    };
    expect(within(label()).queryByRole('option', { name: 'Cornering (legacy)' })).not.toBeInTheDocument();
    await editRange();
    expect(label()).toHaveValue('Cornering');
    expect(within(label()).getByRole('option', { name: 'Cornering (legacy)' })).toBeDisabled();
    fireEvent.change(screen.getByLabelText('Range end (seconds)'), { target: { value: '0.5' } });
    fireEvent.click(screen.getByRole('button', { name: 'Update range' }));
    fireEvent.click(screen.getByRole('button', { name: 'Save training annotation' }));
    await screen.findByText('Saved second.json beside the video.');
    expect(update).toHaveBeenLastCalledWith(second.id, { ranges: resized.selection.ranges, game: 'iracing' });

    await editRange();
    fireEvent.change(label(), { target: { value: 'Shift down' } });
    fireEvent.click(screen.getByRole('button', { name: 'Update range' }));
    fireEvent.click(screen.getByRole('button', { name: 'Save training annotation' }));
    await screen.findByText('Saved second.json beside the video.');
    expect(update).toHaveBeenLastCalledWith(second.id, { ranges: edited.selection.ranges, game: 'iracing' });
    expect(timeline.getByRole('button', { name: 'Range 1: Shift down, 0:00.20 – 0:00.50' })).toBeVisible();
    expect(window.recordedTrainingRanges!.write).not.toHaveBeenCalled();
    expect(secondCheckbox).not.toBeChecked();
    expect(screen.getByRole('button', { name: 'Save training annotation' })).toBeDisabled();
});

it('selects and clears a fixed label for the full recording', async () => {
    const annotation = savedAnnotation('drive.json', 'drive.mp4', 'drive.jsonl', 0, null);
    const labeled = { ...annotation, selection: { ...annotation.selection, fullRecordingLabel: 'Shift up' } };
    mockAnnotationPicker([annotation]);
    const update = jest.mocked(window.recordedTrainingRanges!.update).mockResolvedValueOnce(labeled).mockResolvedValueOnce(annotation);
    render(<RecordedFrameViewer configuration={cloneConfig()} policy={null} disabled={false} onFrame={jest.fn()} />);
    pickAnnotations(annotation.name);
    await loadAnnotationPreview(2.5);
    const label = screen.getByRole('combobox', { name: 'Full recording label (optional)' });
    const save = screen.getByRole('button', { name: 'Save training annotation' });
    const timeline = within(screen.getByRole('region', { name: 'Label timeline' }));
    expect(label).toHaveValue('');
    fireEvent.change(label, { target: { value: 'Shift up' } });
    fireEvent.click(save);
    await screen.findByText('Saved drive.json beside the video.');
    expect(update).toHaveBeenLastCalledWith(annotation.id, { ranges: null, fullRecordingLabel: 'Shift up' });
    expect(timeline.getByRole('img', { name: 'Full aligned recording: Shift up, 0:00.00 – 0:00.30' })).toBeVisible();
    expect(timeline.queryByRole('button', { name: /^Range / })).not.toBeInTheDocument();

    fireEvent.change(label, { target: { value: '' } });
    fireEvent.click(save);
    await screen.findByText('Saved drive.json beside the video.');
    expect(update).toHaveBeenLastCalledWith(annotation.id, { ranges: null });
});

it('uses the saved simulator for renamed telemetry and allows restoring automatic detection', async () => {
    const annotation = savedAnnotation('drive.json', 'drive.mp4', 'renamed.jsonl', 0, null);
    mockAnnotationPicker([annotation]);
    const addedGame: TrainingAnnotation = { ...annotation, selection: { ...annotation.selection, game: 'iracing' } };
    const update = jest.mocked(window.recordedTrainingRanges!.update).mockResolvedValueOnce(addedGame).mockResolvedValueOnce(annotation);
    parse.mockImplementation(async (_, game) => ({ ...recording, game }));
    const ref = React.createRef<RecordedFrameViewerHandle>();
    render(<RecordedFrameViewer ref={ref} configuration={cloneConfig()} policy={null} disabled={false} onFrame={jest.fn()} />);
    pickAnnotations(annotation.name);
    await loadAnnotationPreview(2.5);
    const game = screen.getByLabelText('Game for JSONL without game metadata');
    const save = screen.getByRole('button', { name: 'Save training annotation' });
    expect(game).toHaveValue('');
    expect(parse).toHaveBeenLastCalledWith(telemetryStream('renamed.jsonl'), 'acc', expect.any(AbortSignal));

    fireEvent.change(game, { target: { value: 'iracing' } });
    await screen.findByText(/^iracing ·/);
    const consume = jest.fn(async () => undefined);
    await act(async () => ref.current!.measure(consume, new AbortController().signal));
    expect(consume).toHaveBeenLastCalledWith(expect.objectContaining({ session: expect.objectContaining({ game: 'acc' }) }));

    fireEvent.click(save);
    await screen.findByText('Saved drive.json beside the video.');
    expect(update).toHaveBeenLastCalledWith(annotation.id, { ranges: null, game: 'iracing' });
    consume.mockClear();
    await act(async () => ref.current!.measure(consume, new AbortController().signal));
    expect(consume).toHaveBeenLastCalledWith(expect.objectContaining({ session: expect.objectContaining({ game: 'iracing' }) }));

    fireEvent.change(game, { target: { value: '' } });
    await screen.findByText(/^acc ·/);
    fireEvent.click(save);
    await screen.findByText('Saved drive.json beside the video.');
    expect(update).toHaveBeenLastCalledWith(annotation.id, { ranges: null, game: null });
    expect(game).toHaveValue('');
});

it('replays every clip from the checked annotation files, including overlapping clips', async () => {
    const ref = React.createRef<RecordedFrameViewerHandle>();
    const config = cloneConfig();
    config.runtime.controlIntervalMs = 100;
    const first = { ...recording, id: 'first', durationSeconds: 2 };
    const second = { ...recording, id: 'second', durationSeconds: 2 };
    const firstTelemetry = 'iracing_1795554000000_123abc456def.jsonl';
    const secondTelemetry = 'second-controls.jsonl';
    parse.mockImplementation(async (_, game) => ({ ...(game === 'iracing' ? first : second), game }));
    const secondAnnotation = savedAnnotation('second.json', 'second-camera.webm', secondTelemetry, 0.3,
        [{ start: 0.6, end: 0.7, label: 'Overtake' }]);
    const open = mockAnnotationPicker([
        savedAnnotation('first.json', 'first-camera.mp4', firstTelemetry, 0.1,
            [{ start: 0, end: 0.3, label: 'Practice' }, { start: 0.2, end: 0.4, label: 'Hairpin' }, { start: 0.4, end: 0.5 }]),
        { ...secondAnnotation, selection: { ...secondAnnotation.selection, game: 'ac' } },
        savedAnnotation('extra.json', 'extra.mp4', 'extra.jsonl', 0, null),
        savedAnnotation('empty.json', 'empty.mp4', 'empty.jsonl', 0, []),
    ]);
    const onReplayReadyChange = jest.fn();
    function View() {
        const [container, setContainer] = React.useState<HTMLDivElement | null>(null);
        return <>
            <RecordedFrameViewer ref={ref} configuration={config} policy={null} disabled={false}
                trainingPickerContainer={container} onFrame={jest.fn()} onReplayReadyChange={onReplayReadyChange} />
            <section aria-label="AI driver"><div ref={setContainer} /></section>
        </>;
    }
    render(<View />);
    const aiDriver = within(screen.getByRole('region', { name: 'AI driver' }));
    const picker = aiDriver.getByLabelText('Select annotation files for training');
    const selectTrainingFiles = (...names: string[]) => fireEvent.change(picker, { target: {
        files: names.map((name) => new File(['{}'], name)),
    } });
    expect(onReplayReadyChange).toHaveBeenLastCalledWith(false);
    expect(picker).toHaveAttribute('multiple');

    selectTrainingFiles('extra.json');
    const extra = await aiDriver.findByRole('checkbox', { name: /^extra\.json/ });
    expect(extra).toBeChecked();
    expect(onReplayReadyChange).toHaveBeenLastCalledWith(true);

    selectTrainingFiles('first.json', 'second.json', 'empty.json', 'missing.json');
    expect(await aiDriver.findByRole('group', { name: 'Annotations for training (2 selected)' })).toBeInTheDocument();
    expect(extra).not.toBeChecked();
    expect(aiDriver.getByRole('checkbox', { name: /^empty\.json/ })).toBeDisabled();
    expect(aiDriver.getByRole('alert')).toHaveTextContent('missing.json: Put the video in the same folder');
    expect(open).toHaveBeenCalledTimes(5);
    expect(screen.queryByLabelText('Original recorded video')).not.toBeInTheDocument();
    expect(window.fetch).not.toHaveBeenCalled();

    const create = document.createElement.bind(document);
    jest.spyOn(document, 'createElement').mockImplementation(((tag: string) => {
        const element = create(tag);
        if (tag === 'video') {
            const player = element as HTMLVideoElement;
            mockVideo(player, 2.5);
            Object.defineProperty(player, 'src', { set: () => Promise.resolve().then(() => player.dispatchEvent(new Event('loadeddata'))) });
            player.load = jest.fn();
        }
        return element;
    }) as typeof document.createElement);
    const positions: Array<[string | undefined, string | undefined, number, number, boolean | undefined]> = [];
    await act(async () => ref.current!.measure(async (frame) => {
        positions.push([frame.recordingId, frame.clipLabel, Number(frame.videoSeconds.toFixed(3)),
            Number(frame.telemetrySeconds.toFixed(3)), frame.startsRange]);
    }, new AbortController().signal));
    const frames: typeof positions = [];
    const onProgress = jest.fn();
    await act(async () => ref.current!.replay(async (frame) => {
        frames.push([frame.recordingId, frame.clipLabel, Number(frame.videoSeconds.toFixed(3)),
            Number(frame.telemetrySeconds.toFixed(3)), frame.startsRange]);
    }, new AbortController().signal, onProgress));
    expect(frames).toEqual([
        ['first.json', 'Practice', 0.1, 100, true], ['first.json', 'Practice', 0.2, 100.1, false], ['first.json', 'Practice', 0.3, 100.2, false],
        ['first.json', 'Hairpin', 0.2, 100.1, true], ['first.json', 'Hairpin', 0.3, 100.2, false], ['first.json', 'Hairpin', 0.4, 100.3, false],
        ['first.json', undefined, 0.4, 100.3, true], ['first.json', undefined, 0.5, 100.4, false],
        ['second.json', 'Overtake', 0.6, 100.3, true], ['second.json', 'Overtake', 0.7, 100.4, false],
    ]);
    expect(positions).toEqual(frames);
    expect(onProgress.mock.calls.map(([position]) => position)).toEqual([
        ...Array.from({ length: 9 }, (_, processedFrames) => ({ recordingNumber: 1, recordingCount: 2, processedFrames, totalFrames: 8 })),
        ...[0, 1, 2].map((processedFrames) => ({ recordingNumber: 2, recordingCount: 2, processedFrames, totalFrames: 2 })),
    ]);
    expect(parse.mock.calls).toEqual([
        [telemetryStream(firstTelemetry), 'iracing', expect.any(AbortSignal)],
        [telemetryStream(secondTelemetry), 'ac', expect.any(AbortSignal)],
        [telemetryStream(firstTelemetry), 'iracing', expect.any(AbortSignal)],
        [telemetryStream(secondTelemetry), 'ac', expect.any(AbortSignal)],
    ]);

    selectTrainingFiles('empty.json', 'missing.json');
    await waitFor(() => expect(onReplayReadyChange).toHaveBeenLastCalledWith(false));
    expect(aiDriver.getByRole('group', { name: 'Annotations for training (0 selected)' })).toBeInTheDocument();
});

it('ignores an in-flight inspection after the selected model changes', async () => {
    let finish!: (value: ReturnType<typeof result>) => void;
    const first = { steps: 20, inspectCameraAsync: jest.fn(() => new Promise((resolve) => { finish = resolve; })) } as unknown as DrivingPolicy;
    const second = { steps: 40, inspectCameraAsync: jest.fn() } as unknown as DrivingPolicy;
    const props = { configuration: cloneConfig(), disabled: false, onFrame: () => undefined };
    const view = render(<RecordedFrameViewer {...props} policy={first} />);
    await loadVideo();
    fireEvent.click(screen.getByRole('button', { name: 'Inspect current frame' }));
    view.rerender(<RecordedFrameViewer {...props} policy={second} />);
    await act(async () => { finish(result(2)); });
    expect(screen.queryByLabelText('Layer 1, channel 1 activation map')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Inspect current frame' })).toBeEnabled();
});
