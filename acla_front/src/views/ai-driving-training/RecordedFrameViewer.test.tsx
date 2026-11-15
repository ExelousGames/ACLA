import React from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import RecordedFrameViewer, { RecordedFrameViewerHandle } from './RecordedFrameViewer';
import { cloneConfig } from './driving-model-config';
import type { DrivingPolicy } from './driving-policy';
import { parseRecordedDrivingSession, RecordedDrivingSession } from './recorded-driving-session';

jest.mock('./recorded-driving-session', () => ({ parseRecordedDrivingSession: jest.fn() }));
const parse = parseRecordedDrivingSession as jest.MockedFunction<typeof parseRecordedDrivingSession>;
const recording: RecordedDrivingSession = { id: 'local-1', game: 'acc', rows: [{} as any, {} as any],
    startTimeSeconds: 100, durationSeconds: 0.3, timestampField: 'Graphics_analyst_session_time' };
const result = (peak: number) => ({ layers: [{ index: 0, width: 2, height: 1, channels: 2,
    values: new Float32Array([peak, 0, peak / 2, -0.1]) }] });

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
afterEach(() => jest.restoreAllMocks());

async function loadVideo(duration = 2.5) {
    fireEvent.change(screen.getByLabelText('Recorded video'), { target: { files: [new File(['video'], 'drive.mp4', { type: 'video/mp4' })] } });
    const video = await screen.findByLabelText('Original recorded video') as HTMLVideoElement;
    let time = 0;
    Object.defineProperties(video, {
        duration: { configurable: true, value: duration }, videoWidth: { configurable: true, value: 16 },
        videoHeight: { configurable: true, value: 9 }, readyState: { configurable: true, value: 2 },
        currentTime: { configurable: true, get: () => time, set: (value: number) => {
            time = value;
            Promise.resolve().then(() => video.dispatchEvent(new Event('seeked')));
        } },
    });
    fireEvent.loadedMetadata(video); fireEvent.loadedData(video);
    return video;
}
async function loadTelemetry() {
    const file = new File(['telemetry'], 'drive.jsonl');
    Object.defineProperty(file, 'text', { value: () => Promise.resolve('telemetry') });
    fireEvent.change(screen.getByLabelText('Telemetry JSONL'), { target: { files: [file] } });
    await waitFor(() => expect(screen.queryByText('Reading telemetry…')).not.toBeInTheDocument());
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
    fireEvent.change(screen.getByLabelText('Local session folder'), { target: { files: [
        new File(['a'], 'a.mp4'), new File(['b'], 'b.mp4'), new File(['a'], 'a.jsonl'), new File(['b'], 'b.jsonl'),
    ] } });
    expect(screen.getByLabelText('Video in folder')).toHaveValue('');
    expect(screen.getByLabelText('Telemetry in folder')).toHaveValue('');
    expect(screen.queryByLabelText('Original recorded video')).not.toBeInTheDocument();
    expect(parse).not.toHaveBeenCalled();
});

it('keeps missing-folder-file errors visible after loading the video and clears them when the pair is complete', async () => {
    render(<RecordedFrameViewer configuration={cloneConfig()} policy={null} disabled={false} onFrame={() => undefined} />);
    fireEvent.change(screen.getByLabelText('Local session folder'), { target: { files: [new File(['video'], 'drive.mp4')] } });
    await screen.findByLabelText('Original recorded video');
    expect(screen.getByRole('alert')).toHaveTextContent('select the missing file individually');
    await loadTelemetry();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
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
    expect(screen.getByLabelText('First telemetry at video time (seconds)')).toHaveValue(0);
    expect(screen.queryByLabelText('Strongest examples for channel 1')).not.toBeInTheDocument();
    expect(screen.queryByAltText('Original recorded frame at 0:00.00')).not.toBeInTheDocument();
});

it('replays only the overlap, awaits each training consumer, and stops on cancellation', async () => {
    const ref = React.createRef<RecordedFrameViewerHandle>();
    const onBusyChange = jest.fn();
    render(<RecordedFrameViewer ref={ref} configuration={cloneConfig()} policy={null} disabled={false}
        onFrame={() => undefined} onBusyChange={onBusyChange} />);
    await loadVideo(); await loadTelemetry();
    fireEvent.change(screen.getByLabelText('First telemetry at video time (seconds)'), { target: { value: '0.5' } });
    const abort = new AbortController();
    let finish!: () => void;
    const consume = jest.fn(() => new Promise<void>((resolve) => { finish = resolve; }));
    let pending!: Promise<void>;
    await act(async () => { pending = ref.current!.replay(consume, abort.signal); await Promise.resolve(); });
    expect(consume).toHaveBeenCalledTimes(1);
    expect(consume).toHaveBeenCalledWith(expect.objectContaining({ videoSeconds: 0.5, offsetSeconds: 0.5 }));
    expect(onBusyChange).toHaveBeenLastCalledWith(true);
    await act(async () => {
        abort.abort(); finish();
        await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    });
    expect(consume).toHaveBeenCalledTimes(1);
    expect(onBusyChange).toHaveBeenLastCalledWith(false);
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
