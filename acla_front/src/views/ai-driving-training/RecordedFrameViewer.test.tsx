import React from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import RecordedFrameViewer, { RecordedFrameViewerHandle } from './RecordedFrameViewer';
import { cloneConfig } from './driving-model-config';
import type { DrivingPolicy } from './driving-policy';
import { parseRecordedDrivingSession, RecordedDrivingSession } from './recorded-driving-session';
import type { TrainingAnnotation, TrainingSelection } from './RecordedTrainingRanges';

jest.mock('./recorded-driving-session', () => ({ parseRecordedDrivingSession: jest.fn() }));
const parse = parseRecordedDrivingSession as jest.MockedFunction<typeof parseRecordedDrivingSession>;
const recording: RecordedDrivingSession = { id: 'local-1', game: 'acc', rows: [{} as any, {} as any],
    startTimeSeconds: 100, durationSeconds: 0.3, timestampField: 'Graphics_analyst_session_time' };
const result = (peak: number) => ({ layers: [{ index: 0, width: 2, height: 1, channels: 2,
    values: new Float32Array([peak, 0, peak / 2, -0.1]) }] });
const originalFetch = window.fetch;

beforeEach(() => {
    parse.mockReset().mockReturnValue(recording);
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
function openEditor() {
    fireEvent.click(screen.getByRole('button', { name: 'Add new training annotation' }));
}
async function loadVideo(duration = 2.5) {
    openEditor();
    fireEvent.change(screen.getByLabelText('Recorded video'), { target: { files: [new File(['video'], 'drive.mp4', { type: 'video/mp4' })] } });
    const video = await screen.findByLabelText('Original recorded video') as HTMLVideoElement;
    mockVideo(video, duration);
    fireEvent.loadedMetadata(video); fireEvent.loadedData(video);
    return video;
}
async function loadTelemetry() {
    const file = new File(['telemetry'], 'drive.jsonl');
    Object.defineProperty(file, 'text', { value: () => Promise.resolve('telemetry') });
    fireEvent.change(screen.getByLabelText('Telemetry JSONL'), { target: { files: [file] } });
    await waitFor(() => expect(screen.queryByText('Reading telemetry…')).not.toBeInTheDocument());
}
function savedAnnotation(name: string, videoFileName: string, telemetryFileName: string,
    offsetSeconds: number, ranges: TrainingSelection['ranges']): TrainingAnnotation {
    return { id: name, name, videoUrl: `acla-training-media://recording/${videoFileName}`,
        telemetryUrl: `acla-training-media://recording/${telemetryFileName}`,
        selection: { version: 2, videoFileName, telemetryFileName, offsetSeconds, ranges } };
}
function mockAnnotationPicker(annotations: TrainingAnnotation[]) {
    const open = jest.fn(async (file: File) => {
        const annotation = annotations.find((item) => item.name === file.name);
        if (!annotation) throw new Error('Put the video in the same folder as this annotation.');
        return annotation;
    });
    window.recordedTrainingRanges = { read: jest.fn(async () => null), write: jest.fn(), update: jest.fn(), open, scan: jest.fn() };
    window.fetch = jest.fn(async (url: string) => ({ ok: true, text: async () => url.split('/').pop() })) as jest.Mock;
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

it('pairs local frames with parsed telemetry, applies offset changes, and revokes its video URL', async () => {
    const onFrame = jest.fn();
    const view = render(<RecordedFrameViewer configuration={cloneConfig()} policy={null} disabled={false} onFrame={onFrame} />);
    await loadVideo(); await loadTelemetry();
    expect(parse).toHaveBeenCalledWith('telemetry', 'acc');
    expect(onFrame).toHaveBeenLastCalledWith(expect.objectContaining({ session: recording, videoSeconds: 0, offsetSeconds: 0,
        camera: expect.objectContaining({ width: 384, height: 216, rgb: expect.any(Uint8Array) }) }));
    fireEvent.change(screen.getByLabelText('First telemetry at video time (seconds)'), { target: { value: '1.25' } });
    expect(onFrame).toHaveBeenLastCalledWith(expect.objectContaining({ offsetSeconds: 1.25 }));
    expect(screen.getByRole('button', { name: 'Inspect current frame' })).toBeDisabled();
    view.unmount();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:recorded-video');
    expect(onFrame).toHaveBeenLastCalledWith(null);
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
    const readTelemetry = jest.fn(async () => 'telemetry');
    Object.defineProperty(telemetry, 'text', { value: readTelemetry });
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
    Object.defineProperty(telemetry, 'text', { value: async () => 'telemetry' });
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
    fireEvent.change(screen.getByLabelText('First telemetry at video time (seconds)'), { target: { value: '1.25' } });
    await loadTelemetry();
    expect(screen.getByLabelText('First telemetry at video time (seconds)')).toHaveValue(1.25);
    expect(screen.queryByLabelText('Strongest examples for channel 1')).not.toBeInTheDocument();
    expect(screen.queryByAltText('Original recorded frame at 0:00.00')).not.toBeInTheDocument();
});

it('replays only the overlap, awaits each training consumer, and stops on cancellation', async () => {
    mockAnnotationPicker([savedAnnotation('drive.json', 'drive.mp4', 'drive.jsonl', 0.5, null)]);
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
    expect(consume).toHaveBeenCalledWith(expect.objectContaining({ videoSeconds: 0.5, offsetSeconds: 0.5 }));
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

it('marks, edits and removes saved ranges, and restores each video’s selection and offset', async () => {
    let stored: TrainingAnnotation | undefined;
    const write = jest.fn(async (video: File, telemetry: File, { offsetSeconds, ranges }: TrainingSelection) => {
        stored = savedAnnotation('drive.mp4.training-ranges.json', video.name, telemetry.name, offsetSeconds, ranges);
        return stored;
    });
    window.recordedTrainingRanges = { read: jest.fn(async () => stored?.selection ?? null), write, update: jest.fn(), open: jest.fn(), scan: jest.fn() };
    const props = { configuration: cloneConfig(), policy: null, disabled: false, onFrame: jest.fn() };
    const view = render(<RecordedFrameViewer {...props} />);
    const video = await loadVideo();
    await loadTelemetry();
    const start = () => screen.getByLabelText('Range start (seconds)');
    const end = () => screen.getByLabelText('Range end (seconds)');
    fireEvent.click(screen.getByRole('button', { name: 'Mark start here' }));
    await act(async () => { video.currentTime = 0.4; });
    fireEvent.click(screen.getByRole('button', { name: 'Mark end here' }));
    fireEvent.click(screen.getByRole('button', { name: 'Add range' }));
    await act(async () => { video.currentTime = 1; });
    fireEvent.click(screen.getByRole('button', { name: 'Mark start here' }));
    fireEvent.change(end(), { target: { value: '1.5' } });
    fireEvent.click(screen.getByRole('button', { name: 'Add range' }));
    fireEvent.click(screen.getByRole('button', { name: 'Edit range 2' }));
    expect(start()).toHaveValue(1);
    fireEvent.change(start(), { target: { value: '0.3' } });
    fireEvent.click(screen.getByRole('button', { name: 'Update range' }));
    expect(screen.getByRole('alert')).toHaveTextContent('touches another range');
    fireEvent.change(start(), { target: { value: '1.1' } });
    fireEvent.click(screen.getByRole('button', { name: 'Update range' }));
    fireEvent.change(screen.getByLabelText('First telemetry at video time (seconds)'), { target: { value: '0.25' } });
    const save = () => screen.getByRole('button', { name: 'Save training annotation' });
    fireEvent.click(save());
    await waitFor(() => expect(screen.queryByText('Unsaved annotation changes')).not.toBeInTheDocument());
    expect(write).toHaveBeenCalledWith(expect.objectContaining({ name: 'drive.mp4' }), expect.objectContaining({ name: 'drive.jsonl' }), { version: 1, offsetSeconds: 0.25,
        ranges: [{ start: 0, end: 0.4 }, { start: 1.1, end: 1.5 }] });
    expect(screen.getByRole('checkbox', { name: /drive.mp4.training-ranges.json/ })).toBeChecked();
    view.unmount();

    render(<RecordedFrameViewer {...props} />);
    await loadVideo();
    await loadTelemetry();
    expect(screen.getByLabelText('First telemetry at video time (seconds)')).toHaveValue(0.25);
    fireEvent.click(screen.getByRole('button', { name: 'Jump to start of range 2' }));
    fireEvent.click(screen.getByRole('button', { name: 'Remove range 2' }));
    fireEvent.click(screen.getByRole('button', { name: 'Remove range 1' }));
    expect(screen.getByText(/No footage selected/)).toBeInTheDocument();
    fireEvent.click(save());
    await waitFor(() => expect(screen.queryByText('Unsaved annotation changes')).not.toBeInTheDocument());
    expect(write).toHaveBeenLastCalledWith(expect.any(File), expect.any(File), expect.objectContaining({ offsetSeconds: 0.25, ranges: [] }));
});

it('edits a selected saved annotation in place and preserves its training checkbox', async () => {
    const first = savedAnnotation('first.json', 'first.mp4', 'first.jsonl', 0, null);
    const second = savedAnnotation('second.json', 'second.mp4', 'second.jsonl', 0.1, [{ start: 0.2, end: 0.4 }]);
    const edited = { ...second, selection: { ...second.selection, offsetSeconds: 0.15, ranges: [{ start: 0.2, end: 0.45 }] } };
    mockAnnotationPicker([first, second]);
    const update = jest.mocked(window.recordedTrainingRanges!.update).mockResolvedValue(edited);
    render(<RecordedFrameViewer configuration={cloneConfig()} policy={null} disabled={false} onFrame={jest.fn()} />);
    pickAnnotations(first.name, second.name);
    await loadAnnotationPreview(2.5);
    const secondCheckbox = screen.getByRole('checkbox', { name: /^second\.json/ });
    fireEvent.click(secondCheckbox);
    fireEvent.click(screen.getByRole('button', { name: 'Edit second.json' }));
    await loadAnnotationPreview(2.5);
    expect(screen.getByLabelText('Recorded video')).not.toBeVisible();
    const offset = screen.getByLabelText('First telemetry at video time (seconds)');
    expect(offset).toHaveValue(0.1);
    fireEvent.change(offset, { target: { value: '0.15' } });
    fireEvent.click(screen.getByRole('button', { name: 'Edit range 1' }));
    fireEvent.change(screen.getByLabelText('Range end (seconds)'), { target: { value: '0.45' } });
    fireEvent.click(screen.getByRole('button', { name: 'Update range' }));
    fireEvent.click(screen.getByRole('button', { name: 'Save training annotation' }));
    await screen.findByText('Saved second.json beside the video.');
    expect(update).toHaveBeenCalledWith(second.id, edited.selection);
    expect(window.recordedTrainingRanges!.write).not.toHaveBeenCalled();
    expect(secondCheckbox).not.toBeChecked();
    expect(screen.getByRole('button', { name: 'Save training annotation' })).toBeDisabled();
});

it('selects one or more annotation files from the AI driver and replays only the checked files', async () => {
    const ref = React.createRef<RecordedFrameViewerHandle>();
    const config = cloneConfig();
    config.runtime.controlIntervalMs = 100;
    const first = { ...recording, id: 'first', durationSeconds: 2 };
    const second = { ...recording, id: 'second', durationSeconds: 2 };
    parse.mockImplementation((input) => input === 'first-controls.jsonl' ? first : second);
    const open = mockAnnotationPicker([
        savedAnnotation('first.json', 'first-camera.mp4', 'first-controls.jsonl', 0.1,
            [{ start: 0, end: 0.2 }, { start: 0.4, end: 0.5 }]),
        savedAnnotation('second.json', 'second-camera.webm', 'second-controls.jsonl', 0.3, [{ start: 0.6, end: 0.7 }]),
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
    const frames: Array<[string, number, number, boolean | undefined]> = [];
    const onProgress = jest.fn();
    await act(async () => ref.current!.replay(async (frame) => {
        frames.push([frame.session.id, frame.videoSeconds, frame.offsetSeconds, frame.startsRange]);
    }, new AbortController().signal, onProgress));
    expect(frames).toEqual([
        ['first', 0.1, 0.1, true], ['first', 0.2, 0.1, false],
        ['first', 0.4, 0.1, true], ['first', 0.5, 0.1, false],
        ['second', 0.6, 0.3, true], ['second', 0.7, 0.3, false],
    ]);
    expect(onProgress.mock.calls.map(([position]) => position)).toEqual([
        ...[0, 1, 2, 3, 4].map((processedFrames) => ({ recordingNumber: 1, recordingCount: 2, processedFrames, totalFrames: 4 })),
        ...[0, 1, 2].map((processedFrames) => ({ recordingNumber: 2, recordingCount: 2, processedFrames, totalFrames: 2 })),
    ]);
    expect(parse.mock.calls.map(([input]) => input)).toEqual(['first-controls.jsonl', 'second-controls.jsonl']);

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
