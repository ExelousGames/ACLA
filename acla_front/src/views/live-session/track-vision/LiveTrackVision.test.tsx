import React from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import LiveTrackVision, { TrackVisionHandle } from './LiveTrackVision';
import { GpuInferenceError, TrackVisionModel } from './track-vision-model';
import { drawVisionOverlay } from './vision-overlay';

jest.mock('./vision-overlay', () => ({ drawVisionOverlay: jest.fn() }));

jest.mock('./track-vision-model', () => ({
    ...jest.requireActual('./track-vision-model'), TrackVisionModel: { loadBuiltin: jest.fn() },
}));

const flush = async () => { await act(async () => { await Promise.resolve(); }); };
const startCapture = async () => {
    fireEvent.click(screen.getByRole('button', { name: 'Refresh sources' }));
    await flush();
    fireEvent.change(screen.getByRole('combobox', { name: 'Game window or screen' }), { target: { value: 'window:42' } });
    fireEvent.click(screen.getByRole('button', { name: 'Share game screen' }));
    await flush();
};

const deferred = <T,>() => {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>((done) => { resolve = done; });
    return { promise, resolve };
};
let track: { stop: jest.Mock; onended: (() => void) | null };
let stream: MediaStream;
let getDisplayMedia: jest.Mock;
let model: { detect: jest.Mock; dispose: jest.Mock; executionProvider: 'webgpu' | 'wasm'; fallbackReason?: string };
const detection = { task: 'semantic' as const, width: 2, height: 2, classes: new Uint16Array([0, 1, 1, 0]), inferenceMs: 50 };

beforeEach(() => {
    jest.useFakeTimers();
    track = { stop: jest.fn(), onended: null };
    stream = { getTracks: () => [track], getVideoTracks: () => [track] } as unknown as MediaStream;
    getDisplayMedia = jest.fn().mockResolvedValue(stream);
    Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getDisplayMedia } });
    jest.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();
    jest.spyOn(HTMLMediaElement.prototype, 'readyState', 'get').mockReturnValue(4);
    jest.spyOn(HTMLVideoElement.prototype, 'videoWidth', 'get').mockReturnValue(1280);
    jest.spyOn(HTMLVideoElement.prototype, 'videoHeight', 'get').mockReturnValue(720);
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({ drawImage: jest.fn(), clearRect: jest.fn(), beginPath: jest.fn(), moveTo: jest.fn(), lineTo: jest.fn(), stroke: jest.fn() } as any);
    model = { detect: jest.fn().mockResolvedValue(detection), dispose: jest.fn().mockResolvedValue(undefined), executionProvider: 'webgpu' };
    (TrackVisionModel.loadBuiltin as jest.Mock).mockResolvedValue(model);
    window.screenCapture = {
        listSources: jest.fn().mockResolvedValue([{ id: 'window:42', name: 'Simulator' }]),
        selectSource: jest.fn().mockResolvedValue(undefined),
    };
});

afterEach(() => { jest.restoreAllMocks(); jest.clearAllTimers(); jest.useRealTimers(); delete window.screenCapture; });

it('captures silently, publishes detections, clears empty frames, and stops on unmount', async () => {
    const ref = React.createRef<TrackVisionHandle>();
    const { unmount } = render(<LiveTrackVision ref={ref} name="visualization:track-vision" />);
    await flush();
    const listener = jest.fn();
    ref.current!.subscribeDetection(listener);
    await startCapture();
    expect(getDisplayMedia).toHaveBeenCalledWith(expect.objectContaining({ audio: false }));
    expect(ref.current!.getLatestDetection()?.detections.semantic).toEqual(detection);
    expect(screen.getByRole('status')).toHaveTextContent('Semantic · 50 ms');
    expect(listener).toHaveBeenCalled();
    model.detect.mockResolvedValue({ ...detection, classes: new Uint16Array(4) });
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(ref.current!.getLatestDetection()?.detections.semantic).toMatchObject({ classes: new Uint16Array(4) });
    const handle = ref.current!;
    unmount();
    expect(track.stop).toHaveBeenCalledTimes(1);
    expect(model.dispose).toHaveBeenCalledTimes(1);
    expect(handle.getLatestDetection()).toBeNull();
});

it('stops a stream that arrives after the panel is removed', async () => {
    const pending = deferred<MediaStream>();
    getDisplayMedia.mockReturnValue(pending.promise);
    const { unmount } = render(<LiveTrackVision name="vision" />);
    await flush();
    await startCapture();
    unmount();
    await act(async () => { pending.resolve(stream); });
    expect(track.stop).toHaveBeenCalledTimes(1);
});

it('discards inference finishing after Stop and never queues overlapping frames', async () => {
    const pending = deferred<typeof detection>();
    model.detect.mockReturnValue(pending.promise);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    await startCapture();
    await act(async () => { jest.advanceTimersByTime(2000); });
    expect(model.detect).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole('button', { name: 'Stop capture' }));
    await act(async () => { pending.resolve(detection); });
    expect(ref.current!.getLatestDetection()).toBeNull();
    expect(screen.getByRole('status')).toHaveTextContent('Screen capture stopped');
});

it('cleans up when sharing ends and reports permission denials', async () => {
    render(<LiveTrackVision name="vision" />);
    await flush();
    await startCapture();
    act(() => { track.onended!(); });
    expect(track.stop).toHaveBeenCalled();
    getDisplayMedia.mockRejectedValue(new DOMException('Denied', 'NotAllowedError'));
    await startCapture();
    expect(screen.getByRole('alert')).toHaveTextContent('cancelled or denied');
});

it('requires an explicit desktop source and grants it before requesting capture', async () => {
    render(<LiveTrackVision name="vision" />);
    expect(screen.getByRole('button', { name: 'Share game screen' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Refresh sources' }));
    await flush();
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'window:42' } });
    fireEvent.click(screen.getByRole('button', { name: 'Share game screen' }));
    await flush();
    expect(window.screenCapture!.selectSource).toHaveBeenCalledWith('window:42');
    expect(getDisplayMedia).toHaveBeenCalledTimes(1);
});

it('releases a model whose loading finishes after unmount', async () => {
    const pending = deferred<any>();
    (TrackVisionModel.loadBuiltin as jest.Mock).mockReturnValue(pending.promise);
    const { unmount } = render(<LiveTrackVision name="vision" />);
    await flush();
    unmount();
    await act(async () => { pending.resolve(model); });
    expect(model.dispose).toHaveBeenCalledTimes(1);
});

it('shows model errors and keeps the screen preview available without weights', async () => {
    (TrackVisionModel.loadBuiltin as jest.Mock).mockRejectedValue(new Error('Built-in model unavailable.'));
    render(<LiveTrackVision name="vision" />);
    await flush();
    expect(screen.getByRole('alert')).toHaveTextContent('Built-in model unavailable');
    await startCapture();
    expect(screen.getByLabelText('Captured game frame with vision detections')).toBeVisible();
    expect(screen.getByRole('status')).toHaveTextContent('Waiting for enabled detectors');
    expect(model.detect).not.toHaveBeenCalled();
});

it('loads the bundled model automatically and detects without a file upload', async () => {
    const ref = React.createRef<TrackVisionHandle>();
    const { unmount } = render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    expect(TrackVisionModel.loadBuiltin).toHaveBeenCalledTimes(1);
    expect(TrackVisionModel.loadBuiltin).toHaveBeenCalledWith('semantic', false);
    expect(screen.getByRole('checkbox', { name: 'Allow CPU fallback' })).not.toBeChecked();
    expect(screen.getByLabelText('Semantic inference device')).toHaveTextContent('GPU acceleration active');
    expect(screen.queryByLabelText('Track class')).not.toBeInTheDocument();
    await startCapture();
    expect(model.detect).toHaveBeenCalledTimes(1);
    expect(ref.current!.getLatestDetection()?.detections.semantic).toEqual(detection);
    unmount();
    expect(model.dispose).toHaveBeenCalledTimes(1);
});

it('allows retrying a failed bundled load', async () => {
    (TrackVisionModel.loadBuiltin as jest.Mock).mockRejectedValueOnce(new Error('Built-in model unavailable.'));
    render(<LiveTrackVision name="vision" />);
    await flush();
    expect(screen.getByRole('alert')).toHaveTextContent('Built-in model unavailable');
    fireEvent.click(screen.getByRole('button', { name: 'Retry Semantic' }));
    await flush();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Semantic inference device')).toHaveTextContent('GPU acceleration active');
});

it('automatically retries GPU failures until loading succeeds', async () => {
    (TrackVisionModel.loadBuiltin as jest.Mock)
        .mockRejectedValueOnce(new GpuInferenceError('GPU device lost'))
        .mockRejectedValueOnce(new GpuInferenceError('GPU device lost'));
    render(<LiveTrackVision name="vision" />);
    await flush();
    expect(screen.getByText('Retrying GPU…')).toBeInTheDocument();
    await act(async () => { jest.advanceTimersByTime(2999); });
    expect(TrackVisionModel.loadBuiltin).toHaveBeenCalledTimes(1);
    await act(async () => { jest.advanceTimersByTime(1); });
    expect(TrackVisionModel.loadBuiltin).toHaveBeenCalledTimes(2);
    expect(screen.getByText('Retrying GPU…')).toBeInTheDocument();
    await act(async () => { jest.advanceTimersByTime(3000); });
    expect(TrackVisionModel.loadBuiltin).toHaveBeenCalledTimes(3);
    expect(TrackVisionModel.loadBuiltin).toHaveBeenLastCalledWith('semantic', false);
    expect(screen.getByLabelText('Semantic inference device')).toHaveTextContent('GPU acceleration active');
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    await act(async () => { jest.advanceTimersByTime(6000); });
    expect(TrackVisionModel.loadBuiltin).toHaveBeenCalledTimes(3);
});

it('leaves missing weights available for manual retry without automatic loading', async () => {
    (TrackVisionModel.loadBuiltin as jest.Mock).mockRejectedValue(new Error('Weights unavailable'));
    render(<LiveTrackVision name="vision" />);
    await flush();
    await act(async () => { jest.advanceTimersByTime(10000); });
    expect(TrackVisionModel.loadBuiltin).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('button', { name: 'Retry Semantic' })).toBeEnabled();
});

it.each(['disable', 'unmount'])('cancels scheduled GPU retries on %s', async (action) => {
    (TrackVisionModel.loadBuiltin as jest.Mock).mockRejectedValue(new GpuInferenceError('GPU device lost'));
    const { unmount } = render(<LiveTrackVision name="vision" />);
    await flush();
    if (action === 'unmount') unmount();
    else fireEvent.click(screen.getByRole('checkbox', { name: 'Enable Semantic' }));
    await flush();
    await act(async () => { jest.advanceTimersByTime(10000); });
    expect(TrackVisionModel.loadBuiltin).toHaveBeenCalledTimes(1);
});

it('allows CPU fallback on demand and reloads CPU models on GPU when turned off', async () => {
    const cpu = { ...model, executionProvider: 'wasm', fallbackReason: 'GPU could not run this model; using CPU.', dispose: jest.fn().mockResolvedValue(undefined) };
    (TrackVisionModel.loadBuiltin as jest.Mock).mockImplementation(async (_task, allowCpuFallback) => {
        if (allowCpuFallback) return cpu;
        throw new GpuInferenceError('GPU device lost');
    });
    render(<LiveTrackVision name="vision" />);
    await flush();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Allow CPU fallback' }));
    await flush();
    expect(TrackVisionModel.loadBuiltin).toHaveBeenLastCalledWith('semantic', true);
    expect(screen.getByLabelText('Semantic inference device')).toHaveTextContent('CPU inference · GPU could not run this model; using CPU.');
    await act(async () => { jest.advanceTimersByTime(6000); });
    expect(TrackVisionModel.loadBuiltin).toHaveBeenCalledTimes(2);
    (TrackVisionModel.loadBuiltin as jest.Mock).mockResolvedValue(model);
    fireEvent.click(screen.getByRole('checkbox', { name: 'Allow CPU fallback' }));
    await flush();
    expect(cpu.dispose).toHaveBeenCalledTimes(1);
    expect(TrackVisionModel.loadBuiltin).toHaveBeenLastCalledWith('semantic', false);
    expect(screen.getByLabelText('Semantic inference device')).toHaveTextContent('GPU acceleration active');
});

it('keeps healthy GPU sessions when CPU fallback is toggled', async () => {
    render(<LiveTrackVision name="vision" />);
    await flush();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Allow CPU fallback' }));
    await flush();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Allow CPU fallback' }));
    await flush();
    expect(TrackVisionModel.loadBuiltin).toHaveBeenCalledTimes(1);
    expect(model.dispose).not.toHaveBeenCalled();
});

it('discards a pending CPU load when fallback is turned off', async () => {
    const pending = deferred<any>();
    const cpu = { ...model, executionProvider: 'wasm', dispose: jest.fn().mockResolvedValue(undefined) };
    (TrackVisionModel.loadBuiltin as jest.Mock)
        .mockRejectedValueOnce(new GpuInferenceError('GPU device lost'))
        .mockReturnValueOnce(pending.promise);
    render(<LiveTrackVision name="vision" />);
    await flush();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Allow CPU fallback' }));
    await flush();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Allow CPU fallback' }));
    await act(async () => { pending.resolve(cpu); });
    expect(cpu.dispose).toHaveBeenCalledTimes(1);
    expect(TrackVisionModel.loadBuiltin).toHaveBeenLastCalledWith('semantic', false);
    expect(screen.getByLabelText('Semantic inference device')).toHaveTextContent('GPU acceleration active');
});

it('automatically recovers GPU inference during capture', async () => {
    const recovered = { ...model, detect: jest.fn().mockResolvedValue(detection), dispose: jest.fn().mockResolvedValue(undefined) };
    model.detect.mockRejectedValue(new Error('GPU device lost'));
    (TrackVisionModel.loadBuiltin as jest.Mock).mockResolvedValueOnce(model).mockResolvedValue(recovered);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    await startCapture();
    expect(model.dispose).toHaveBeenCalledTimes(1);
    expect(screen.getByText('Retrying GPU…')).toBeInTheDocument();
    await act(async () => { jest.advanceTimersByTime(3000); });
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(TrackVisionModel.loadBuiltin).toHaveBeenLastCalledWith('semantic', false);
    expect(ref.current!.getLatestDetection()?.detections.semantic).toEqual(detection);
    expect(track.stop).not.toHaveBeenCalled();
});

it('waits for queued detectors to load before retrying GPU failures', async () => {
    const pending = deferred<any>();
    (TrackVisionModel.loadBuiltin as jest.Mock).mockImplementation(async (task) => {
        if (task === 'depth') return pending.promise;
        throw new GpuInferenceError('GPU device lost');
    });
    render(<LiveTrackVision name="vision" />);
    await flush();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Enable Depth' }));
    await flush();
    const calls = (TrackVisionModel.loadBuiltin as jest.Mock).mock.calls.length;
    await act(async () => { jest.advanceTimersByTime(10000); });
    expect(TrackVisionModel.loadBuiltin).toHaveBeenCalledTimes(calls);
    await act(async () => { pending.resolve(model); });
    await act(async () => { jest.advanceTimersByTime(3000); });
    expect(TrackVisionModel.loadBuiltin).toHaveBeenCalledTimes(calls + 1);
    expect(screen.getByLabelText('Depth inference device')).toHaveTextContent('GPU acceleration active');
    expect(model.dispose).not.toHaveBeenCalled();
});

it('runs enabled detectors on the same frame and removes a disabled detection immediately', async () => {
    const depth = { ...model, detect: jest.fn().mockResolvedValue({ task: 'depth', width: 2, height: 2, values: new Float32Array([1, 2, 3, 4]), inferenceMs: 20 }), dispose: jest.fn().mockResolvedValue(undefined) };
    const segment = { ...model, detect: jest.fn().mockResolvedValue({ task: 'segment', width: 2, height: 2, instances: [], inferenceMs: 10 }), dispose: jest.fn().mockResolvedValue(undefined) };
    (TrackVisionModel.loadBuiltin as jest.Mock).mockImplementation(async (task) => task === 'depth' ? depth : task === 'segment' ? segment : model);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    expect(screen.getByRole('checkbox', { name: 'Enable Semantic' })).toBeChecked();
    expect(screen.getByRole('checkbox', { name: 'Enable Depth' })).not.toBeChecked();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Enable Depth' }));
    fireEvent.click(screen.getByRole('checkbox', { name: 'Enable Segment' }));
    await flush();
    await startCapture();
    expect(Object.keys(ref.current!.getLatestDetection()!.detections)).toEqual(['semantic', 'depth', 'segment']);
    expect(depth.detect.mock.calls[0][0]).toBe(model.detect.mock.calls[0][0]);
    expect(segment.detect.mock.calls[0][0]).toBe(model.detect.mock.calls[0][0]);
    fireEvent.click(screen.getByRole('checkbox', { name: 'Enable Semantic' }));
    expect(ref.current!.getLatestDetection()).toBeNull();
    await flush();
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(Object.keys(ref.current!.getLatestDetection()!.detections)).toEqual(['depth', 'segment']);
    expect(model.dispose).toHaveBeenCalledTimes(1);
    expect(track.stop).not.toHaveBeenCalled();
});

it('allows all detectors to be disabled while capture continues', async () => {
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Enable Semantic' }));
    await flush();
    await startCapture();
    expect(ref.current!.getLatestDetection()?.detections).toEqual({});
    expect(model.detect).not.toHaveBeenCalled();
    expect(screen.getByRole('status')).toHaveTextContent('Enable a detection');
});

it('keeps working detections running if another detector fails', async () => {
    const failed = { ...model, detect: jest.fn().mockRejectedValue(new Error('Depth inference failed.')), dispose: jest.fn().mockResolvedValue(undefined) };
    (TrackVisionModel.loadBuiltin as jest.Mock).mockImplementation(async (task) => task === 'depth' ? failed : model);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Enable Depth' }));
    await flush();
    await startCapture();
    expect(screen.getByRole('alert')).toHaveTextContent('Depth inference failed');
    expect(ref.current!.getLatestDetection()?.detections.semantic).toEqual(detection);
    expect(ref.current!.getLatestDetection()?.detections.depth).toBeUndefined();
    expect(failed.dispose).toHaveBeenCalled();
    expect(track.stop).not.toHaveBeenCalled();
});

it('discards a detector load that finishes after it is disabled', async () => {
    const pending = deferred<any>();
    (TrackVisionModel.loadBuiltin as jest.Mock).mockReturnValue(pending.promise);
    render(<LiveTrackVision name="vision" />);
    await flush();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Enable Semantic' }));
    await act(async () => { pending.resolve(model); });
    expect(model.dispose).toHaveBeenCalledTimes(1);
    await startCapture();
    expect(model.detect).not.toHaveBeenCalled();
});

it('does not republish a disabled detector when its pending inference finishes', async () => {
    const pending = deferred<typeof detection>();
    model.detect.mockReturnValue(pending.promise);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    await startCapture();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Enable Semantic' }));
    await act(async () => { pending.resolve(detection); });
    expect(ref.current!.getLatestDetection()).toBeNull();
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(ref.current!.getLatestDetection()?.detections).toEqual({});
    expect(model.detect).toHaveBeenCalledTimes(1);
});

it('recolors the displayed frame while inference is pending without restarting the detection stack', async () => {
    const depthResult = { task: 'depth' as const, width: 2, height: 2, values: new Float32Array([1, 10, 30, 60]), inferenceMs: 20 };
    const pending = deferred<typeof depthResult>();
    const depth = { ...model, detect: jest.fn().mockResolvedValueOnce(depthResult).mockReturnValue(pending.promise) };
    (TrackVisionModel.loadBuiltin as jest.Mock).mockImplementation(async (task) => task === 'depth' ? depth : model);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Enable Depth' }));
    await flush();
    await startCapture();

    const displayedResult = ref.current!.getLatestDetection();
    const preview = screen.getByLabelText('Captured game frame with vision detections') as HTMLCanvasElement;
    const context = preview.getContext('2d')!;
    const drawImage = context.drawImage as jest.Mock;
    const displayedFrame = drawImage.mock.calls[drawImage.mock.calls.length - 1][0];
    expect(drawVisionOverlay).toHaveBeenLastCalledWith(context, displayedResult, { near: 5, far: 50 });

    await act(async () => { jest.advanceTimersByTime(200); });
    expect(depth.detect).toHaveBeenCalledTimes(2);
    fireEvent.change(screen.getByRole('slider', { name: 'Close depth' }), { target: { value: '10' } });
    fireEvent.change(screen.getByRole('slider', { name: 'Far depth' }), { target: { value: '80' } });
    expect(drawVisionOverlay).toHaveBeenLastCalledWith(context, displayedResult, { near: 10, far: 80 });
    expect(drawImage).toHaveBeenLastCalledWith(displayedFrame, 0, 0);
    expect(displayedFrame).not.toBe(depth.detect.mock.calls[1][0]);
    expect(TrackVisionModel.loadBuiltin).toHaveBeenCalledTimes(2);
    expect(depth.detect).toHaveBeenCalledTimes(2);
    expect(track.stop).not.toHaveBeenCalled();

    await act(async () => { pending.resolve(depthResult); });
    expect(drawVisionOverlay).toHaveBeenLastCalledWith(context, ref.current!.getLatestDetection(), { near: 10, far: 80 });
    fireEvent.click(screen.getByRole('button', { name: 'Reset depth range' }));
    expect(drawVisionOverlay).toHaveBeenLastCalledWith(context, ref.current!.getLatestDetection(), { near: 5, far: 50 });
});

it('keeps the close and far cutoffs ordered and retains them when depth is toggled', async () => {
    render(<LiveTrackVision name="vision" />);
    await flush();
    expect(screen.queryByRole('slider', { name: 'Close depth' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Enable Depth' }));
    await flush();
    const close = screen.getByRole('slider', { name: 'Close depth' });
    const far = screen.getByRole('slider', { name: 'Far depth' });
    fireEvent.change(close, { target: { value: '50' } });
    expect(close).toHaveValue('49.5');
    fireEvent.change(far, { target: { value: '0' } });
    expect(far).toHaveValue('50');
    fireEvent.click(screen.getByRole('checkbox', { name: 'Enable Depth' }));
    await flush();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Enable Depth' }));
    await flush();
    expect(screen.getByRole('slider', { name: 'Close depth' })).toHaveValue('49.5');
    expect(screen.getByRole('slider', { name: 'Far depth' })).toHaveValue('50');
    fireEvent.click(screen.getByRole('button', { name: 'Reset depth range' }));
    expect(close).not.toBeInTheDocument();
    expect(screen.getByRole('slider', { name: 'Close depth' })).toHaveValue('5');
    expect(screen.getByRole('slider', { name: 'Far depth' })).toHaveValue('50');
});

it('requires the Electron bridge before loading models or offering capture', async () => {
    delete window.screenCapture;
    render(<LiveTrackVision name="vision" />);
    await flush();
    expect(screen.getByRole('alert')).toHaveTextContent('Track Vision is available only in the Electron desktop app.');
    expect(screen.queryByRole('button', { name: 'Share game screen' })).not.toBeInTheDocument();
    expect(TrackVisionModel.loadBuiltin).not.toHaveBeenCalled();
    expect(getDisplayMedia).not.toHaveBeenCalled();
});
