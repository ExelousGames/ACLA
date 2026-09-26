import React from 'react';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import LiveTrackVision, { TrackVisionHandle } from './LiveTrackVision';
import { GpuInferenceError, TrackVisionModel } from './track-vision-model';
import { drawVisionOverlay } from './vision-overlay';
import { MODEL_LABELS, vision } from './test-fixtures';
import { VISION_MAX_AGE_MS } from './track-vision-types';

jest.mock('./vision-overlay', () => ({ drawVisionOverlay: jest.fn() }));

jest.mock('./track-vision-model', () => ({
    ...jest.requireActual('./track-vision-model'), TrackVisionModel: { loadBackend: jest.fn(), loadBuiltin: jest.fn() },
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
let depthModel: { detect: jest.Mock; dispose: jest.Mock; executionProvider: string; name: string; classNames: string[] };
let model: { detect: jest.Mock; dispose: jest.Mock; executionProvider: 'webgpu' | 'wasm'; fallbackReason?: string; name: string; classNames: string[] };
const detection = { task: 'segment' as const, width: 2, height: 2, instances: [], classNames: ['track', 'curb'], inferenceMs: 50 };

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
    const contexts = new WeakMap<HTMLCanvasElement, CanvasRenderingContext2D>();
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (this: HTMLCanvasElement) {
        if (!contexts.has(this)) contexts.set(this, { drawImage: jest.fn(), clearRect: jest.fn(), getImageData: jest.fn((_x, _y, width, height) => ({ data: new Uint8ClampedArray(width * height * 4) })), createImageData: jest.fn((width, height) => ({ data: new Uint8ClampedArray(width * height * 4) })), putImageData: jest.fn() } as any);
        return contexts.get(this)!;
    });
    model = { name: 'track-features-v2', classNames: ['track', 'curb'], detect: jest.fn().mockResolvedValue(detection), dispose: jest.fn().mockResolvedValue(undefined), executionProvider: 'webgpu' };
    (TrackVisionModel.loadBackend as jest.Mock).mockResolvedValue(model);
    depthModel = { name: 'YOLO26n Depth', classNames: [], executionProvider: 'webgpu', dispose: jest.fn().mockResolvedValue(undefined),
        detect: jest.fn().mockResolvedValue(vision(0).detections.depth) };
    (TrackVisionModel.loadBuiltin as jest.Mock).mockResolvedValue(depthModel);
    window.screenCapture = {
        listSources: jest.fn().mockResolvedValue([{ id: 'window:42', name: 'Simulator' }]),
        selectSource: jest.fn().mockResolvedValue(undefined),
    };
});

afterEach(() => { jest.restoreAllMocks(); jest.clearAllTimers(); jest.useRealTimers(); delete window.screenCapture; });

it('updates boundary geometry immediately and keeps the newest cutoff through pending inference and capture restart', async () => {
    const fixture = vision(0, { width: 1280, height: 720 });
    model.detect.mockResolvedValue(fixture.detections.segment);
    depthModel.detect.mockResolvedValue(fixture.detections.depth);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    await startCapture();
    fireEvent.click(screen.getByRole('button', { name: 'Apply camera calibration' }));
    const before = ref.current!.getLatestDetection()!;
    expect(before.boundaryStartDistanceM).toBe(5);
    expect(within(screen.getByRole('dialog', { name: 'Capture preview' })).queryByRole('slider')).not.toBeInTheDocument();
    expect(within(screen.getByRole('region', { name: 'Local 3D reconstruction' })).getByRole('slider', { name: 'Boundary start line' })).toBeInTheDocument();
    const edges = screen.getByLabelText('Reconstructed track edges');
    const originalEdges = edges.innerHTML;
    const listener = jest.fn();
    ref.current!.subscribeDetection(listener);
    fireEvent.change(screen.getByRole('slider', { name: 'Boundary start' }), { target: { value: '12' } });
    const after = ref.current!.getLatestDetection()!;
    expect(after.capturedAt).toBe(before.capturedAt);
    expect(after.calibration).toEqual(before.calibration);
    expect(after.reconstruction!.leftBoundary.length).toBeLessThan(before.reconstruction!.leftBoundary.length);
    expect(after.reconstruction!.cars).toEqual(before.reconstruction!.cars);
    expect(edges.innerHTML).not.toBe(originalEdges);
    expect(listener).toHaveBeenCalledTimes(1);
    expect(model.detect).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('slider', { name: 'Boundary start line' })).toHaveAttribute('aria-valuenow', '12');

    const pending = deferred<typeof detection>();
    model.detect.mockReturnValueOnce(pending.promise);
    await act(async () => { jest.advanceTimersByTime(200); });
    fireEvent.keyDown(screen.getByRole('slider', { name: 'Boundary start line' }), { key: 'ArrowUp' });
    await act(async () => { pending.resolve(fixture.detections.segment as typeof detection); });
    expect(ref.current!.getLatestDetection()!.boundaryStartDistanceM).toBe(12.5);
    expect(screen.getByRole('slider', { name: 'Boundary start' })).toHaveValue('12.5');

    fireEvent.click(screen.getByRole('button', { name: 'Stop capture' }));
    expect(screen.queryByRole('slider', { name: 'Boundary start line' })).not.toBeInTheDocument();
    await startCapture();
    expect(ref.current!.getLatestDetection()!.boundaryStartDistanceM).toBe(12.5);
});

it.each(['restore', 'escape'])('keeps capture and calibration running while expanding and returning with %s', async (action) => {
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    await startCapture();
    fireEvent.click(screen.getByRole('button', { name: 'Enable on capture' }));
    fireEvent.click(screen.getByRole('button', { name: 'Apply camera calibration' }));
    const calibration = ref.current!.getLatestDetection()!.calibration;
    const preview = screen.getByLabelText('Capture preview');
    const canvas = screen.getByLabelText('Captured game frame with vision detections');
    const video = preview.querySelector('video')!;
    const showModal = jest.fn(() => preview.setAttribute('open', ''));
    const show = jest.fn(() => preview.setAttribute('open', ''));
    const close = jest.fn(() => preview.removeAttribute('open'));
    // jsdom does not implement the native dialog API.
    Object.assign(preview, { showModal, show, close });

    fireEvent.click(screen.getByRole('button', { name: 'Expand capture' }));
    expect(showModal).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('button', { name: 'Restore capture' })).toHaveAttribute('aria-expanded', 'true');
    expect(within(preview).getByRole('button', { name: 'Stop capture' })).toBeEnabled();
    const callsBeforeFrame = model.detect.mock.calls.length;
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(model.detect.mock.calls.length).toBeGreaterThan(callsBeforeFrame);
    expect(ref.current!.getLatestDetection()!.calibration).toEqual(calibration);
    expect(within(preview).getByLabelText('Projected ground grid')).toBeVisible();

    if (action === 'restore') fireEvent.click(screen.getByRole('button', { name: 'Restore capture' }));
    else fireEvent(preview, new Event('cancel', { cancelable: true }));
    expect(show).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('button', { name: 'Expand capture' })).toHaveAttribute('aria-expanded', 'false');
    expect(screen.getByLabelText('Captured game frame with vision detections')).toBe(canvas);
    expect(preview.querySelector('video')).toBe(video);
    expect(video.srcObject).toBe(stream);
    expect(getDisplayMedia).toHaveBeenCalledTimes(1);
    expect(TrackVisionModel.loadBackend).toHaveBeenCalledTimes(1);
    expect(track.stop).not.toHaveBeenCalled();
    expect(model.dispose).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: 'Expand capture' }));
    fireEvent.click(within(preview).getByRole('button', { name: 'Stop capture' }));
    expect(track.stop).toHaveBeenCalledTimes(1);
    expect(ref.current!.getLatestDetection()).toBeNull();
    expect(within(preview).getByRole('button', { name: 'Restore capture' })).toBeEnabled();
});

it('keeps camera position and the optional reference grid without any bird-eye view', async () => {
    model.detect.mockResolvedValue(vision(0).detections.segment);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    expect(screen.getByRole('group', { name: 'Camera position' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Enable on capture' })).toBeDisabled();
    expect(screen.queryByText(/bird.?s?.?eye/i)).not.toBeInTheDocument();
    await startCapture();
    const capture = within(screen.getByLabelText('Capture preview'));
    fireEvent.click(screen.getByRole('button', { name: 'Enable on capture' }));
    expect(capture.getByLabelText('Projected ground grid')).toBeInTheDocument();
    expect(screen.getByLabelText('Perspective 3D track edges and cars')).toBeInTheDocument();
    expect(screen.getByLabelText('Reconstructed cars').querySelectorAll('circle').length).toBeGreaterThan(0);
    expect(ref.current!.getLatestDetection()?.reconstruction).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Apply camera calibration' }));
    const applied = ref.current!.getLatestDetection();
    expect(applied?.reconstruction?.cars).toHaveLength(1);
    expect(applied?.geometry).not.toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Disable on capture' }));
    expect(capture.queryByLabelText('Projected ground grid')).not.toBeInTheDocument();
    expect(ref.current!.getLatestDetection()).toBe(applied);
    expect(model.detect).toHaveBeenCalledTimes(1);
    expect(track.stop).not.toHaveBeenCalled();
});

it('clears local 3D and positions when depth is disabled and recovers when enabled', async () => {
    model.detect.mockResolvedValue(vision(0).detections.segment);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    await startCapture();
    fireEvent.click(screen.getByRole('button', { name: 'Apply camera calibration' }));
    expect(ref.current!.getLatestDetection()?.reconstruction?.cars).toHaveLength(1);
    fireEvent.click(screen.getByRole('checkbox', { name: 'Enable Depth' }));
    await flush();
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(ref.current!.getLatestDetection()?.reconstruction).toBeNull();
    expect(ref.current!.getLatestDetection()?.geometry).toBeNull();
    expect(screen.getByLabelText('Reconstructed cars').children).toHaveLength(0);
    expect(screen.getByLabelText('Reconstruction status')).toHaveTextContent('Enable segmentation and depth');
    expect(screen.getByLabelText('Driver position')).toHaveTextContent('Unknown');
    fireEvent.click(screen.getByRole('checkbox', { name: 'Enable Depth' }));
    await flush();
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(ref.current!.getLatestDetection()?.reconstruction?.cars).toHaveLength(1);
});

it('publishes metric geometry and positions only after camera calibration is applied', async () => {
    model.detect.mockResolvedValue(vision(0, { classNames: MODEL_LABELS }).detections.segment);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    await startCapture();
    expect(screen.getByLabelText('Driver position')).toHaveTextContent('Unknown');
    expect(ref.current!.getLatestDetection()!.geometry).toBeNull();
    expect(screen.getByLabelText('Road fit status')).toHaveTextContent('Road observed');
    const capturedAt = ref.current!.getLatestDetection()!.capturedAt;
    const listener = jest.fn();
    ref.current!.subscribeDetection(listener);
    fireEvent.click(screen.getByRole('button', { name: 'Apply camera calibration' }));
    expect(ref.current!.getLatestDetection()).toMatchObject({ capturedAt, analysis: {
        cornerDirection: 'left', playerPosition: 'inside', carAhead: 1, opponentPosition: 'outside',
    } });
    expect(ref.current!.getLatestDetection()!.geometry!.trackWidthM).toBeCloseTo(10, 0);
    expect(listener).toHaveBeenCalled();
    expect(screen.getByLabelText('Driver position')).toHaveTextContent('Inside');
    expect(screen.getByLabelText('Opponent position')).toHaveTextContent('Outside');
    fireEvent.change(screen.getByLabelText('Camera right of car center (m)'), { target: { value: '-2.5' } });
    expect(ref.current!.getLatestDetection()!.analysis).toEqual({});
    expect(ref.current!.getLatestDetection()!.geometry).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Apply camera calibration' }));
    expect(ref.current!.getLatestDetection()).toMatchObject({ capturedAt, analysis: { playerPosition: 'middle' } });
    fireEvent.click(screen.getByRole('button', { name: 'Stop capture' }));
    expect(ref.current!.getLatestDetection()).toBeNull();
    expect(screen.getByLabelText('Driver position')).toHaveTextContent('Unknown');
    expect(screen.queryByLabelText('Perspective 3D track edges and cars')).not.toBeInTheDocument();
});

it('expires positions and boundary coordinates during pending inference without renewing the timestamp', async () => {
    model.detect.mockResolvedValue(vision(0).detections.segment);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    await startCapture();
    fireEvent.click(screen.getByRole('button', { name: 'Enable on capture' }));
    fireEvent.click(screen.getByRole('button', { name: 'Apply camera calibration' }));
    const edges = screen.getByLabelText('Reconstructed track edges');
    expect(edges.querySelectorAll('path').length).toBeGreaterThan(0);
    const capturedAt = ref.current!.getLatestDetection()!.capturedAt;
    const pending = deferred<typeof detection>();
    model.detect.mockReturnValueOnce(pending.promise);
    await act(async () => { jest.advanceTimersByTime(VISION_MAX_AGE_MS + 1); });
    expect(screen.getByLabelText('Driver position')).toHaveTextContent('Unknown');
    expect(screen.getByLabelText('Road fit status')).toHaveTextContent('Waiting for a fresh frame');
    expect(screen.getByLabelText('Reconstructed cars').children).toHaveLength(0);
    expect(edges.querySelectorAll('path')).toHaveLength(0);
    fireEvent.change(screen.getByRole('slider', { name: 'Boundary start' }), { target: { value: '12' } });
    expect(edges.querySelectorAll('path')).toHaveLength(0);
    fireEvent.click(screen.getByRole('button', { name: 'Apply camera calibration' }));
    expect(screen.getByLabelText('Driver position')).toHaveTextContent('Unknown');
    expect(ref.current!.getLatestDetection()!.capturedAt).toBe(capturedAt);
    fireEvent.click(screen.getByRole('checkbox', { name: 'Enable Segmentation' }));
    await act(async () => { pending.resolve(detection); });
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(ref.current!.getLatestDetection()?.analysis).toBeNull();
    expect(ref.current!.getLatestDetection()?.geometry).toBeNull();
});

it('validates camera settings and previews height and angle changes without rerunning inference', async () => {
    const fixture = vision(0, { width: 1280, height: 720 });
    model.detect.mockResolvedValue(fixture.detections.segment);
    depthModel.detect.mockResolvedValue(fixture.detections.depth);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    expect(screen.getByRole('button', { name: 'Apply camera calibration' })).toBeDisabled();
    await startCapture();
    fireEvent.click(screen.getByRole('button', { name: 'Enable on capture' }));
    const capturedAt = ref.current!.getLatestDetection()!.capturedAt;
    const grid = screen.getByLabelText('Projected ground grid');
    const originalGrid = grid.innerHTML;
    const localView = screen.getByLabelText('Perspective 3D track edges and cars');
    const localGrid = () => Array.from(screen.getByLabelText('Depth distance grid').querySelectorAll('path'))
        .map((path) => path.getAttribute('d')).join(' ');
    const originalLocalGrid = localGrid();
    expect(originalLocalGrid).not.toBe('');
    expect(localView).toHaveAttribute('viewBox', '0 0 800 450');
    fireEvent.change(screen.getByLabelText('Camera height (m)'), { target: { value: '1.8' } });
    // Height moves the calibration plane, but must not displace measured road contours.
    expect(localGrid()).toBe(originalLocalGrid);
    fireEvent.change(screen.getByLabelText('Pitch down (°)'), { target: { value: '8' } });
    expect(grid.innerHTML).not.toBe(originalGrid);
    expect(localGrid()).not.toBe(originalLocalGrid);
    const draftLocalGrid = localGrid();
    expect(model.detect).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole('button', { name: 'Apply camera calibration' }));
    expect(ref.current!.getLatestDetection()).toMatchObject({ capturedAt, calibration: { heightM: 1.8, pitchDeg: 8 } });
    expect(localGrid()).toBe(draftLocalGrid);
    fireEvent.change(screen.getByLabelText('Camera height (m)'), { target: { value: '' } });
    expect(ref.current!.getLatestDetection()?.calibration).toBeUndefined();
    expect(screen.getByRole('button', { name: 'Apply camera calibration' })).toBeDisabled();
    expect(screen.queryByLabelText('Projected ground grid')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Perspective 3D track edges and cars')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Disable on capture' }));
    expect(screen.getByRole('button', { name: 'Enable on capture' })).toBeDisabled();
    fireEvent.change(screen.getByLabelText('Camera height (m)'), { target: { value: '0' } });
    expect(screen.getByRole('button', { name: 'Apply camera calibration' })).toBeDisabled();
    fireEvent.change(screen.getByLabelText('Camera height (m)'), { target: { value: '1.8' } });
    fireEvent.click(screen.getByRole('button', { name: 'Apply camera calibration' }));
    fireEvent.click(screen.getByRole('button', { name: 'Clear calibration' }));
    expect(ref.current!.getLatestDetection()?.calibration).toBeUndefined();
});

it('uses the newest calibration after pending inference and clears it on resized or restarted capture', async () => {
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    await startCapture();
    const pending = deferred<typeof detection>();
    model.detect.mockReturnValueOnce(pending.promise);
    await act(async () => { jest.advanceTimersByTime(200); });
    fireEvent.change(screen.getByLabelText('Camera height (m)'), { target: { value: '1.8' } });
    fireEvent.click(screen.getByRole('button', { name: 'Apply camera calibration' }));
    await act(async () => { pending.resolve(detection); });
    expect(ref.current!.getLatestDetection()?.calibration?.heightM).toBe(1.8);
    jest.spyOn(HTMLVideoElement.prototype, 'videoWidth', 'get').mockReturnValue(1920);
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(ref.current!.getLatestDetection()?.calibration).toBeUndefined();
    fireEvent.click(screen.getByRole('button', { name: 'Apply camera calibration' }));
    expect(ref.current!.getLatestDetection()?.calibration?.imageWidth).toBe(1920);
    fireEvent.click(screen.getByRole('button', { name: 'Stop capture' }));
    await startCapture();
    expect(ref.current!.getLatestDetection()?.calibration).toBeUndefined();
});

it('keeps calibration available during detector retry without reviving cleared results', async () => {
    const pending = deferred<typeof model>();
    (TrackVisionModel.loadBackend as jest.Mock).mockRejectedValueOnce(new Error('Backend unavailable')).mockReturnValueOnce(pending.promise);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    await startCapture();
    fireEvent.click(screen.getByRole('button', { name: 'Retry Segmentation' }));
    await flush();
    fireEvent.change(screen.getByLabelText('Camera height (m)'), { target: { value: '1.8' } });
    fireEvent.click(screen.getByRole('button', { name: 'Apply camera calibration' }));
    expect(screen.getByRole('button', { name: 'Clear calibration' })).toBeEnabled();
    expect(ref.current!.getLatestDetection()).toBeNull();
    await act(async () => { pending.resolve(model); });
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(ref.current!.getLatestDetection()?.calibration?.heightM).toBe(1.8);
});

it('captures silently, publishes detections, clears empty frames, and stops on unmount', async () => {
    const ref = React.createRef<TrackVisionHandle>();
    const { unmount } = render(<LiveTrackVision ref={ref} name="visualization:track-vision" />);
    await flush();
    const listener = jest.fn();
    ref.current!.subscribeDetection(listener);
    await startCapture();
    expect(getDisplayMedia).toHaveBeenCalledWith(expect.objectContaining({ audio: false }));
    expect(ref.current!.getLatestDetection()?.detections.segment).toEqual(detection);
    expect(screen.getByRole('status')).toHaveTextContent('Segmentation · 50 ms');
    expect(listener).toHaveBeenCalled();
    model.detect.mockResolvedValue({ ...detection, instances: [] });
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(ref.current!.getLatestDetection()?.detections.segment).toMatchObject({ instances: [] });
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
    (TrackVisionModel.loadBackend as jest.Mock).mockReturnValue(pending.promise);
    const { unmount } = render(<LiveTrackVision name="vision" />);
    await flush();
    unmount();
    await act(async () => { pending.resolve(model); });
    expect(model.dispose).toHaveBeenCalledTimes(1);
});

it('shows model errors and keeps the screen preview available without weights', async () => {
    (TrackVisionModel.loadBackend as jest.Mock).mockRejectedValue(new Error('Backend model unavailable.'));
    render(<LiveTrackVision name="vision" />);
    await flush();
    expect(screen.getByRole('alert')).toHaveTextContent('Backend model unavailable');
    await startCapture();
    expect(screen.getByLabelText('Captured game frame with vision detections')).toBeVisible();
    expect(screen.getByRole('status')).toHaveTextContent('Depth ·');
    expect(model.detect).not.toHaveBeenCalled();
});

it('loads the backend model automatically and detects without a file upload', async () => {
    const ref = React.createRef<TrackVisionHandle>();
    const { unmount } = render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    expect(TrackVisionModel.loadBackend).toHaveBeenCalledTimes(1);
    expect(TrackVisionModel.loadBackend).toHaveBeenCalledWith(false);
    expect(screen.getByRole('checkbox', { name: 'Allow CPU fallback' })).not.toBeChecked();
    expect(screen.getByLabelText('Segmentation inference device')).toHaveTextContent('GPU acceleration active');
    expect(screen.getByLabelText('Model labels')).toHaveTextContent('track, curb');
    expect(screen.getByRole('checkbox', { name: 'Enable Depth' })).toBeChecked();
    expect(TrackVisionModel.loadBuiltin).toHaveBeenCalledWith('depth', false);
    expect(screen.queryByRole('checkbox', { name: 'Enable Semantic' })).not.toBeInTheDocument();
    await startCapture();
    expect(model.detect).toHaveBeenCalledTimes(1);
    expect(ref.current!.getLatestDetection()?.detections.segment).toEqual(detection);
    unmount();
    expect(model.dispose).toHaveBeenCalledTimes(1);
});

it('allows retrying a failed backend load', async () => {
    (TrackVisionModel.loadBackend as jest.Mock).mockRejectedValueOnce(new Error('Backend model unavailable.'));
    render(<LiveTrackVision name="vision" />);
    await flush();
    expect(screen.getByRole('alert')).toHaveTextContent('Backend model unavailable');
    fireEvent.click(screen.getByRole('button', { name: 'Retry Segmentation' }));
    await flush();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Segmentation inference device')).toHaveTextContent('GPU acceleration active');
});

it('runs depth alongside backend segmentation and releases only depth when disabled', async () => {
    const depthResult = { task: 'depth', width: 2, height: 2, values: new Float32Array([1, 2, 3, 4]), inferenceMs: 20, classNames: [] };
    const depth = { ...model, name: 'YOLO26n Depth', classNames: [], detect: jest.fn().mockResolvedValue(depthResult), dispose: jest.fn().mockResolvedValue(undefined) };
    (TrackVisionModel.loadBuiltin as jest.Mock).mockResolvedValue(depth);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    expect(TrackVisionModel.loadBuiltin).toHaveBeenCalledWith('depth', false);
    expect(TrackVisionModel.loadBackend).toHaveBeenCalledTimes(1);
    expect(screen.getByLabelText('Depth inference device')).toHaveTextContent('GPU acceleration active');
    expect(screen.queryByRole('group', { name: /Depth range/ })).not.toBeInTheDocument();
    expect(screen.queryByRole('slider', { name: 'Close depth' })).not.toBeInTheDocument();
    expect(screen.queryByRole('slider', { name: 'Far depth' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Reset depth range' })).not.toBeInTheDocument();
    await startCapture();
    expect(ref.current!.getLatestDetection()?.detections).toEqual({ segment: detection, depth: depthResult });
    const preview = screen.getByLabelText('Captured game frame with vision detections') as HTMLCanvasElement;
    expect(drawVisionOverlay).toHaveBeenLastCalledWith(preview.getContext('2d'), expect.objectContaining({
        detections: { segment: detection, depth: depthResult },
    }));
    expect(depth.detect.mock.calls[0][0]).toBe(model.detect.mock.calls[0][0]);
    fireEvent.click(screen.getByRole('checkbox', { name: 'Enable Depth' }));
    expect(ref.current!.getLatestDetection()).toBeNull();
    await flush();
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(ref.current!.getLatestDetection()?.detections).toEqual({ segment: detection });
    expect(depth.dispose).toHaveBeenCalledTimes(1);
    expect(model.dispose).not.toHaveBeenCalled();
    expect(track.stop).not.toHaveBeenCalled();
});

it('keeps depth available when backend segmentation cannot load', async () => {
    const depthResult = { task: 'depth', width: 2, height: 2, values: new Float32Array([1, 2, 3, 4]), inferenceMs: 20, classNames: [] };
    const depth = { ...model, detect: jest.fn().mockResolvedValue(depthResult), dispose: jest.fn().mockResolvedValue(undefined) };
    (TrackVisionModel.loadBackend as jest.Mock).mockRejectedValue(new Error('Backend unavailable'));
    (TrackVisionModel.loadBuiltin as jest.Mock).mockResolvedValue(depth);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    await startCapture();
    expect(ref.current!.getLatestDetection()?.detections).toEqual({ depth: depthResult });
    expect(screen.getByRole('alert')).toHaveTextContent('Backend unavailable');
    expect(track.stop).not.toHaveBeenCalled();
});

it('keeps segmentation running when depth inference fails', async () => {
    const depth = { ...model, detect: jest.fn().mockRejectedValue(new Error('Depth inference failed')), dispose: jest.fn().mockResolvedValue(undefined) };
    (TrackVisionModel.loadBuiltin as jest.Mock).mockResolvedValue(depth);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    await startCapture();
    expect(ref.current!.getLatestDetection()?.detections).toEqual({ segment: detection });
    expect(screen.getByRole('alert')).toHaveTextContent('Depth inference failed');
    expect(depth.dispose).toHaveBeenCalledTimes(1);
    expect(model.dispose).not.toHaveBeenCalled();
    expect(track.stop).not.toHaveBeenCalled();
});

it('automatically retries GPU failures until loading succeeds', async () => {
    (TrackVisionModel.loadBackend as jest.Mock)
        .mockRejectedValueOnce(new GpuInferenceError('GPU device lost'))
        .mockRejectedValueOnce(new GpuInferenceError('GPU device lost'));
    render(<LiveTrackVision name="vision" />);
    await flush();
    expect(screen.getByText('Retrying GPU…')).toBeInTheDocument();
    await act(async () => { jest.advanceTimersByTime(2999); });
    expect(TrackVisionModel.loadBackend).toHaveBeenCalledTimes(1);
    await act(async () => { jest.advanceTimersByTime(1); });
    expect(TrackVisionModel.loadBackend).toHaveBeenCalledTimes(2);
    expect(screen.getByText('Retrying GPU…')).toBeInTheDocument();
    await act(async () => { jest.advanceTimersByTime(3000); });
    expect(TrackVisionModel.loadBackend).toHaveBeenCalledTimes(3);
    expect(TrackVisionModel.loadBackend).toHaveBeenLastCalledWith(false);
    expect(screen.getByLabelText('Segmentation inference device')).toHaveTextContent('GPU acceleration active');
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    await act(async () => { jest.advanceTimersByTime(6000); });
    expect(TrackVisionModel.loadBackend).toHaveBeenCalledTimes(3);
});

it('leaves missing weights available for manual retry without automatic loading', async () => {
    (TrackVisionModel.loadBackend as jest.Mock).mockRejectedValue(new Error('Weights unavailable'));
    render(<LiveTrackVision name="vision" />);
    await flush();
    await act(async () => { jest.advanceTimersByTime(10000); });
    expect(TrackVisionModel.loadBackend).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('button', { name: 'Retry Segmentation' })).toBeEnabled();
});

it.each(['disable', 'unmount'])('cancels scheduled GPU retries on %s', async (action) => {
    (TrackVisionModel.loadBackend as jest.Mock).mockRejectedValue(new GpuInferenceError('GPU device lost'));
    const { unmount } = render(<LiveTrackVision name="vision" />);
    await flush();
    if (action === 'unmount') unmount();
    else fireEvent.click(screen.getByRole('checkbox', { name: 'Enable Segmentation' }));
    await flush();
    await act(async () => { jest.advanceTimersByTime(10000); });
    expect(TrackVisionModel.loadBackend).toHaveBeenCalledTimes(1);
});

it('allows CPU fallback on demand and reloads CPU models on GPU when turned off', async () => {
    const cpu = { ...model, executionProvider: 'wasm', fallbackReason: 'GPU could not run this model; using CPU.', dispose: jest.fn().mockResolvedValue(undefined) };
    (TrackVisionModel.loadBackend as jest.Mock).mockImplementation(async (allowCpuFallback) => {
        if (allowCpuFallback) return cpu;
        throw new GpuInferenceError('GPU device lost');
    });
    render(<LiveTrackVision name="vision" />);
    await flush();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Allow CPU fallback' }));
    await flush();
    expect(TrackVisionModel.loadBackend).toHaveBeenLastCalledWith(true);
    expect(screen.getByLabelText('Segmentation inference device')).toHaveTextContent('CPU inference · GPU could not run this model; using CPU.');
    await act(async () => { jest.advanceTimersByTime(6000); });
    expect(TrackVisionModel.loadBackend).toHaveBeenCalledTimes(2);
    (TrackVisionModel.loadBackend as jest.Mock).mockResolvedValue(model);
    fireEvent.click(screen.getByRole('checkbox', { name: 'Allow CPU fallback' }));
    await flush();
    expect(cpu.dispose).toHaveBeenCalledTimes(1);
    expect(TrackVisionModel.loadBackend).toHaveBeenLastCalledWith(false);
    expect(screen.getByLabelText('Segmentation inference device')).toHaveTextContent('GPU acceleration active');
});

it('keeps healthy GPU sessions when CPU fallback is toggled', async () => {
    render(<LiveTrackVision name="vision" />);
    await flush();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Allow CPU fallback' }));
    await flush();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Allow CPU fallback' }));
    await flush();
    expect(TrackVisionModel.loadBackend).toHaveBeenCalledTimes(1);
    expect(model.dispose).not.toHaveBeenCalled();
});

it('discards a pending CPU load when fallback is turned off', async () => {
    const pending = deferred<any>();
    const cpu = { ...model, executionProvider: 'wasm', dispose: jest.fn().mockResolvedValue(undefined) };
    (TrackVisionModel.loadBackend as jest.Mock)
        .mockRejectedValueOnce(new GpuInferenceError('GPU device lost'))
        .mockReturnValueOnce(pending.promise);
    render(<LiveTrackVision name="vision" />);
    await flush();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Allow CPU fallback' }));
    await flush();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Allow CPU fallback' }));
    await act(async () => { pending.resolve(cpu); });
    expect(cpu.dispose).toHaveBeenCalledTimes(1);
    expect(TrackVisionModel.loadBackend).toHaveBeenLastCalledWith(false);
    expect(screen.getByLabelText('Segmentation inference device')).toHaveTextContent('GPU acceleration active');
});

it('automatically recovers GPU inference during capture', async () => {
    const recovered = { ...model, detect: jest.fn().mockResolvedValue(detection), dispose: jest.fn().mockResolvedValue(undefined) };
    model.detect.mockRejectedValue(new Error('GPU device lost'));
    (TrackVisionModel.loadBackend as jest.Mock).mockResolvedValueOnce(model).mockResolvedValue(recovered);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    await startCapture();
    expect(model.dispose).toHaveBeenCalledTimes(1);
    expect(screen.getByText('Retrying GPU…')).toBeInTheDocument();
    await act(async () => { jest.advanceTimersByTime(3000); });
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(TrackVisionModel.loadBackend).toHaveBeenLastCalledWith(false);
    expect(ref.current!.getLatestDetection()?.detections.segment).toEqual(detection);
    expect(track.stop).not.toHaveBeenCalled();
});

it('allows all detectors to be disabled while capture continues', async () => {
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Enable Segmentation' }));
    fireEvent.click(screen.getByRole('checkbox', { name: 'Enable Depth' }));
    await flush();
    await startCapture();
    expect(ref.current!.getLatestDetection()?.detections).toEqual({});
    expect(model.detect).not.toHaveBeenCalled();
    expect(screen.getByRole('status')).toHaveTextContent('Enable a detection');
});

it('discards a detector load that finishes after it is disabled', async () => {
    const pending = deferred<any>();
    (TrackVisionModel.loadBackend as jest.Mock).mockReturnValue(pending.promise);
    render(<LiveTrackVision name="vision" />);
    await flush();
    fireEvent.click(screen.getByRole('checkbox', { name: 'Enable Segmentation' }));
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
    fireEvent.click(screen.getByRole('checkbox', { name: 'Enable Segmentation' }));
    fireEvent.click(screen.getByRole('checkbox', { name: 'Enable Depth' }));
    await act(async () => { pending.resolve(detection); });
    expect(ref.current!.getLatestDetection()).toBeNull();
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(ref.current!.getLatestDetection()?.detections).toEqual({});
    expect(model.detect).toHaveBeenCalledTimes(1);
});

it('requires the Electron bridge before loading models or offering capture', async () => {
    delete window.screenCapture;
    render(<LiveTrackVision name="vision" />);
    await flush();
    expect(screen.getByRole('alert')).toHaveTextContent('Track Vision is available only in the Electron desktop app.');
    expect(screen.queryByRole('button', { name: 'Share game screen' })).not.toBeInTheDocument();
    expect(TrackVisionModel.loadBackend).not.toHaveBeenCalled();
    expect(getDisplayMedia).not.toHaveBeenCalled();
});
