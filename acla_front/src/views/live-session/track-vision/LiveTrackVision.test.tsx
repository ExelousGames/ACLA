import React from 'react';
import { act, fireEvent, render as renderComponent, screen, within } from '@testing-library/react';
import LiveTrackVision, { TrackVisionHandle } from './LiveTrackVision';
import { GpuInferenceError, TrackVisionModel } from './track-vision-model';
import { drawVisionOverlay } from './vision-overlay';
import { MODEL_LABELS, multipleCarVision, vision } from './test-fixtures';
import { VISION_MAX_AGE_MS } from './track-vision-types';
import { reconstructTrack } from './track-position-analysis';

jest.mock('./vision-overlay', () => ({ drawVisionOverlay: jest.fn() }));

jest.mock('./track-vision-model', () => ({
    ...jest.requireActual('./track-vision-model'), TrackVisionModel: { loadBackend: jest.fn(), loadBuiltin: jest.fn() },
}));

const selectStep = (name: string) => fireEvent.click(screen.getByRole('tab', { name }));
const cameraButton = (name: string) => {
    selectStep('Camera position');
    return screen.getByRole('button', { name });
};
const render = (element: React.ReactElement) => {
    const view = renderComponent(element);
    const settings = screen.queryByText('Setting');
    if (settings) fireEvent.click(settings);
    return view;
};

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
let depthModel: { detect: jest.Mock; dispose: jest.Mock; executionProvider: string; name: string; classNames: string[]; inputSize: number };
let model: { detect: jest.Mock; dispose: jest.Mock; executionProvider: 'webgpu'; name: string; classNames: string[]; inputSize: number };
const detection = { task: 'segment' as const, width: 2, height: 2, instances: [{
    classId: 0, confidence: 0.9, box: [0, 0, 1, 1] as [number, number, number, number], mask: new Uint8Array(4).fill(1),
}], classNames: ['track', 'curb'], inferenceMs: 50 };

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
        if (!contexts.has(this)) contexts.set(this, { canvas: this, drawImage: jest.fn(), clearRect: jest.fn(), save: jest.fn(), restore: jest.fn(), fillRect: jest.fn(), fillText: jest.fn(), measureText: jest.fn(() => ({ width: 80 })), getImageData: jest.fn((_x, _y, width, height) => ({ data: new Uint8ClampedArray(width * height * 4) })), createImageData: jest.fn((width, height) => ({ data: new Uint8ClampedArray(width * height * 4) })), putImageData: jest.fn() } as any);
        return contexts.get(this)!;
    });
    model = { name: 'track-features-v2', classNames: ['track', 'curb'], detect: jest.fn().mockResolvedValue(detection), dispose: jest.fn().mockResolvedValue(undefined), executionProvider: 'webgpu', inputSize: 768 };
    (TrackVisionModel.loadBackend as jest.Mock).mockResolvedValue(model);
    depthModel = { name: 'Depth-Anything-V2-Small', classNames: [], executionProvider: 'webgpu', inputSize: 518, dispose: jest.fn().mockResolvedValue(undefined),
        detect: jest.fn().mockResolvedValue(vision(0).detections.depth) };
    (TrackVisionModel.loadBuiltin as jest.Mock).mockResolvedValue(depthModel);
    window.screenCapture = {
        listSources: jest.fn().mockResolvedValue([{ id: 'window:42', name: 'Simulator' }]),
        selectSource: jest.fn().mockResolvedValue(undefined),
    };
});

afterEach(() => { jest.restoreAllMocks(); jest.clearAllTimers(); jest.useRealTimers(); delete window.screenCapture; });

it.each([
    ['Segmentation', ['384', '640', '768'], '768'],
    ['Depth', ['252', '392', '518'], '518'],
])('offers three input resolutions for %s with the original size selected', async (label, sizes, defaultSize) => {
    render(<LiveTrackVision name="vision" />);
    await flush();
    const select = screen.getByRole('combobox', { name: `${label} input resolution` });
    expect(select).toHaveValue(defaultSize);
    const options = within(select).getAllByRole('option') as HTMLOptionElement[];
    expect(options.map((option) => option.value)).toEqual(sizes);
    expect(options.map((option) => option.textContent?.split(' · ')[0])).toEqual(['Low', 'Medium', 'High']);
});

it.each(['Segmentation', 'Depth'])('reloads only %s at its selected resolution while capture continues', async (label) => {
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    await startCapture();
    const original = label === 'Segmentation' ? model : depthModel;
    const unchanged = label === 'Segmentation' ? depthModel : model;
    const loader = label === 'Segmentation' ? TrackVisionModel.loadBackend : TrackVisionModel.loadBuiltin;
    const inputSize = label === 'Segmentation' ? 384 : 252;
    const replacement = { ...original, inputSize, dispose: jest.fn().mockResolvedValue(undefined) };
    const pending = deferred<typeof replacement>();
    (loader as jest.Mock).mockReturnValueOnce(pending.promise);
    fireEvent.change(screen.getByRole('combobox', { name: `${label} input resolution` }), { target: { value: inputSize } });
    await flush();
    expect(ref.current!.getLatestDetection()).toBeNull();
    expect(original.dispose).toHaveBeenCalledTimes(1);
    expect(unchanged.dispose).not.toHaveBeenCalled();
    if (label === 'Segmentation') expect(loader).toHaveBeenLastCalledWith(inputSize);
    else expect(loader).toHaveBeenLastCalledWith('depth', inputSize);
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(ref.current!.getLatestDetection()).toBeNull();
    await act(async () => { pending.resolve(replacement); });
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(ref.current!.getLatestDetection()?.detections).toHaveProperty('depth');
    expect(ref.current!.getLatestDetection()?.detections).toHaveProperty('segment');
    expect(TrackVisionModel.loadBackend).toHaveBeenCalledTimes(label === 'Segmentation' ? 2 : 1);
    expect(TrackVisionModel.loadBuiltin).toHaveBeenCalledTimes(label === 'Depth' ? 2 : 1);
    expect(track.stop).not.toHaveBeenCalled();
    expect(getDisplayMedia).toHaveBeenCalledTimes(1);
});

it('discards pending detections and superseded model loads after rapid resolution changes', async () => {
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    await startCapture();
    const inference = deferred<typeof detection>();
    model.detect.mockReturnValueOnce(inference.promise);
    await act(async () => { jest.advanceTimersByTime(200); });
    const low = { ...model, inputSize: 384, dispose: jest.fn().mockResolvedValue(undefined) };
    const medium = { ...model, inputSize: 640, dispose: jest.fn().mockResolvedValue(undefined) };
    const pending = deferred<typeof model>();
    (TrackVisionModel.loadBackend as jest.Mock).mockReturnValueOnce(pending.promise).mockResolvedValueOnce(medium);
    const select = screen.getByRole('combobox', { name: 'Segmentation input resolution' });
    fireEvent.change(select, { target: { value: 384 } });
    await flush();
    fireEvent.change(select, { target: { value: 640 } });
    await act(async () => { inference.resolve(detection); });
    expect(ref.current!.getLatestDetection()).toBeNull();
    expect(depthModel.detect).toHaveBeenCalledTimes(1);
    await act(async () => { pending.resolve(low); });
    expect(low.dispose).toHaveBeenCalledTimes(1);
    expect(TrackVisionModel.loadBackend).toHaveBeenLastCalledWith(640);
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(ref.current!.getLatestDetection()?.detections.segment).toEqual(detection);
    expect(select).toHaveValue('640');
    expect(depthModel.dispose).not.toHaveBeenCalled();
    expect(track.stop).not.toHaveBeenCalled();
});

it('shows all masks in one depth preview with matching per-mask table names and refreshes them with capture', async () => {
    const instances = [
        { ...detection.instances[0], mask: new Uint8Array([1, 0, 1, 0]) },
        { ...detection.instances[0], mask: new Uint8Array([0, 1, 0, 1]) },
    ];
    model.detect.mockResolvedValue({ ...detection, instances });
    depthModel.detect.mockResolvedValue({ task: 'depth', width: 2, height: 2,
        values: new Float32Array([10, 30, 20, 40]), classNames: [], inferenceMs: 1 });
    render(<LiveTrackVision name="vision" />);
    await flush();
    await startCapture();
    selectStep('Label depths');
    const preview = screen.getByLabelText('Captured game frame with vision detections') as HTMLCanvasElement;
    expect(preview).toBeVisible();
    expect(screen.getByRole('button', { name: 'Expand capture' })).toBeVisible();
    expect(screen.queryByRole('list', { name: 'Individual mask depths' })).not.toBeInTheDocument();
    const captions = preview.getContext('2d')!.fillText as jest.Mock;
    expect(captions).toHaveBeenCalledWith('track #1 · 15.0 m', expect.any(Number), expect.any(Number));
    expect(captions).toHaveBeenCalledWith('track #2 · 35.0 m', expect.any(Number), expect.any(Number));
    const rows = within(screen.getByRole('table')).getAllByRole('row');
    expect(rows).toHaveLength(3);
    expect(rows[1]).toHaveTextContent('track #1');
    expect(rows[1]).toHaveTextContent('15.0 m');
    expect(rows[2]).toHaveTextContent('track #2');
    expect(rows[2]).toHaveTextContent('35.0 m');
    selectStep('Filtering');
    expect(within(screen.getByRole('tabpanel', { name: 'Filtering' })).getAllByRole('definition')[2]).toHaveTextContent('1');
    selectStep('Label depths');
    captions.mockClear();
    model.detect.mockResolvedValue({ ...detection, instances: [instances[1]] });
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(within(screen.getByRole('table')).getAllByRole('row')).toHaveLength(2);
    expect(captions).toHaveBeenCalledTimes(1);
    expect(captions).toHaveBeenCalledWith('track #1 · 35.0 m', expect.any(Number), expect.any(Number));
    fireEvent.click(screen.getByRole('button', { name: 'Stop capture' }));
    expect(preview).not.toBeVisible();
    expect(within(screen.getByRole('table')).getAllByRole('row')).toHaveLength(1);
});

it('walks the visual pipeline without restarting capture, reloading models or changing published detections', async () => {
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    expect(screen.getAllByRole('tab').map((tab) => tab.textContent?.slice(2))).toEqual([
        'Capture', 'Camera position', 'Segmentation', 'Filtering', 'Depth map', 'Label depths', 'Reconstructed scene',
    ]);
    expect(screen.getByRole('tabpanel', { name: 'Capture' })).toBeVisible();
    expect(screen.queryByRole('button', { name: 'Apply camera calibration' })).not.toBeInTheDocument();
    await startCapture();
    const canvas = screen.getByLabelText('Captured game frame with vision detections');
    const result = ref.current!.getLatestDetection();
    (drawVisionOverlay as jest.Mock).mockClear();
    selectStep('Segmentation');
    expect(drawVisionOverlay).toHaveBeenLastCalledWith((canvas as HTMLCanvasElement).getContext('2d'), result, '');
    expect(screen.getByLabelText('Segmentation label legend')).toHaveTextContent('track');
    selectStep('Filtering');
    expect(screen.getByLabelText('Applied filters')).toHaveTextContent('Car interior retained');
    selectStep('Depth map');
    expect(screen.getByLabelText('Depth map color scale')).toBeVisible();
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
    selectStep('Label depths');
    expect(screen.getByRole('table')).toHaveTextContent('track');
    expect(screen.getByLabelText('Label depth color scale')).toBeVisible();
    selectStep('Reconstructed scene');
    expect(screen.getByLabelText('2D reconstructed scene')).toBeVisible();
    expect(canvas).not.toBeVisible();
    expect(screen.queryByRole('button', { name: 'Reset view' })).not.toBeInTheDocument();
    expect(screen.queryByRole('tab', { name: '3D overview' })).not.toBeInTheDocument();
    expect(screen.queryByRole('tab', { name: 'Camera view' })).not.toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'Screen analysis' })).toBeVisible();
    selectStep('Capture');
    expect(screen.getByLabelText('Captured game frame with vision detections')).toBe(canvas);
    expect(canvas).toBeVisible();
    expect(ref.current!.getLatestDetection()).toBe(result);
    expect(model.detect).toHaveBeenCalledTimes(1);
    expect(depthModel.detect).toHaveBeenCalledTimes(1);
    expect(getDisplayMedia).toHaveBeenCalledTimes(1);
    expect(track.stop).not.toHaveBeenCalled();
    expect(TrackVisionModel.loadBackend).toHaveBeenCalledTimes(1);
});

it('inspects full-map depths at the mouse, refreshes stationary hover and clears unavailable readings', async () => {
    model.detect.mockResolvedValue({ ...detection, instances: [] });
    const depth = { task: 'depth', width: 4, height: 4, values: new Float32Array(16).fill(24.5), classNames: [], inferenceMs: 1 };
    depthModel.detect.mockResolvedValue(depth);
    render(<LiveTrackVision name="vision" />);
    await flush();
    selectStep('Depth map');
    expect(screen.getByText('Waiting for depth.')).toBeVisible();
    await startCapture();
    const canvas = screen.getByLabelText('Captured game frame with vision detections') as HTMLCanvasElement;
    jest.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({ left: 10, top: 20, width: 640, height: 480 } as DOMRect);
    fireEvent.mouseMove(canvas, { clientX: 330, clientY: 260 });
    expect(screen.getByLabelText('Depth at mouse')).toHaveTextContent('24.50 m');
    expect(screen.getByLabelText('Depth map color scale')).toHaveTextContent('Near 24.5 m');
    depthModel.detect.mockResolvedValue({ ...depth, values: new Float32Array(16).fill(37.25) });
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(screen.getByLabelText('Depth at mouse')).toHaveTextContent('37.25 m');
    fireEvent.mouseMove(canvas, { clientX: 330, clientY: 25 });
    expect(screen.queryByLabelText('Depth at mouse')).not.toBeInTheDocument();
    fireEvent.mouseMove(canvas, { clientX: 330, clientY: 260 });
    depthModel.detect.mockResolvedValue({ ...depth, values: new Float32Array(16) });
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(screen.getByLabelText('Depth at mouse')).toHaveTextContent('No valid depth');
    fireEvent.mouseLeave(canvas);
    expect(screen.queryByLabelText('Depth at mouse')).not.toBeInTheDocument();
    fireEvent.mouseMove(canvas, { clientX: 330, clientY: 260 });
    selectStep('Label depths');
    expect(screen.queryByLabelText('Depth at mouse')).not.toBeInTheDocument();
    selectStep('Depth map');
    expect(screen.queryByLabelText('Depth at mouse')).not.toBeInTheDocument();
    fireEvent.mouseMove(canvas, { clientX: 330, clientY: 260 });
    depthModel.detect.mockRejectedValueOnce(new Error('GPU device lost'));
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(screen.queryByLabelText('Depth at mouse')).not.toBeInTheDocument();
    expect(screen.getByText('Waiting for depth.')).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Stop capture' }));
    expect(canvas).not.toBeVisible();
});

it('shows relative depth in the hover, legend, table and mask captions without meter units', async () => {
    depthModel.detect.mockResolvedValue({ task: 'depth', scale: 'relative', width: 2, height: 2,
        values: new Float32Array(4).fill(0.02), classNames: [], inferenceMs: 1 });
    render(<LiveTrackVision name="vision" />);
    await flush();
    expect(screen.queryByText(/Depth-Anything-V2-Small|track-features-v2|Ultralytics/)).not.toBeInTheDocument();
    await startCapture();
    selectStep('Depth map');
    const canvas = screen.getByLabelText('Captured game frame with vision detections') as HTMLCanvasElement;
    jest.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({ left: 0, top: 0, width: 640, height: 360 } as DOMRect);
    fireEvent.mouseMove(canvas, { clientX: 320, clientY: 180 });
    expect(screen.getByLabelText('Depth at mouse')).toHaveTextContent('0.0200 rel');
    expect(screen.getByLabelText('Depth map color scale')).toHaveTextContent('Near 0.0200 rel');
    selectStep('Label depths');
    expect(screen.getByRole('table')).toHaveTextContent('relative depth (unitless)');
    expect(screen.getByRole('table')).toHaveTextContent('0.0200 rel');
    expect(canvas.getContext('2d')!.fillText).toHaveBeenCalledWith('track #1 · 0.0200 rel', expect.any(Number), expect.any(Number));
});

it('keeps the captured window background synchronized with completed scene frames', async () => {
    model.detect.mockResolvedValue(vision(0).detections.segment);
    render(<LiveTrackVision name="vision" />);
    await flush();
    selectStep('Reconstructed scene');
    const background = screen.getByLabelText('Captured window scene') as HTMLCanvasElement;
    expect(background).not.toBeVisible();
    await startCapture();
    expect(background).toBeVisible();
    const context = background.getContext('2d')!;
    const draw = context.drawImage as jest.Mock;
    const preview = screen.getByLabelText('Captured game frame with vision detections') as HTMLCanvasElement;
    const source = (preview.getContext('2d')!.drawImage as jest.Mock).mock.calls[0][0];
    expect(draw).toHaveBeenLastCalledWith(source, 0, 0);
    expect(background).toHaveAttribute('width', '1280');
    expect(background).toHaveAttribute('height', '720');
    const boundaries = screen.getByLabelText('2D reconstructed scene');
    expect(boundaries).toHaveAttribute('viewBox', '0 0 1280 720');
    const displayed = boundaries.innerHTML;
    expect(within(screen.getByLabelText('Reconstructed cars')).getAllByLabelText(/^Car · \d+% confidence$/)).toHaveLength(1);
    const drawsBefore = draw.mock.calls.length;

    const pending = deferred<typeof detection>();
    model.detect.mockReturnValueOnce(pending.promise);
    await act(async () => { jest.advanceTimersByTime(VISION_MAX_AGE_MS + 1); });
    expect(screen.getByLabelText('Reconstructed scene status')).toHaveTextContent('Showing last frame (stale)');
    expect(background).toBeVisible();
    expect(draw).toHaveBeenCalledTimes(drawsBefore);
    expect(boundaries.innerHTML).toBe(displayed);
    await act(async () => { pending.resolve({ ...detection, instances: [] }); });
    expect(draw).toHaveBeenCalledTimes(drawsBefore + 1);
    for (const label of ['Left track boundary', 'Right track boundary', 'Track middle line']) {
        expect(within(boundaries).getByLabelText(label)).toBeEmptyDOMElement();
    }
    expect(screen.queryByLabelText('Reconstructed cars')).not.toBeInTheDocument();

    jest.spyOn(HTMLVideoElement.prototype, 'videoWidth', 'get').mockReturnValue(1920);
    jest.spyOn(HTMLVideoElement.prototype, 'videoHeight', 'get').mockReturnValue(1080);
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(background).toHaveAttribute('width', '1920');
    expect(background).toHaveAttribute('height', '1080');
    expect(boundaries).toHaveAttribute('viewBox', '0 0 1920 1080');
    expect(draw).toHaveBeenCalledTimes(drawsBefore + 2);

    fireEvent.click(screen.getByRole('button', { name: 'Stop capture' }));
    expect(background).not.toBeVisible();
    expect(context.clearRect).toHaveBeenLastCalledWith(0, 0, background.width, background.height);
    expect(within(boundaries).queryByLabelText('Left track boundary')).not.toBeInTheDocument();
    expect(within(boundaries).queryByLabelText('Right track boundary')).not.toBeInTheDocument();
    expect(within(boundaries).queryByLabelText('Track middle line')).not.toBeInTheDocument();
});

it('supports keyboard navigation and names the active pipeline panel', async () => {
    render(<LiveTrackVision name="vision" />);
    await flush();
    const captureTab = screen.getByRole('tab', { name: 'Capture' });
    captureTab.focus();
    fireEvent.keyDown(captureTab, { key: 'ArrowRight' });
    expect(screen.getByRole('tab', { name: 'Camera position' })).toHaveFocus();
    expect(screen.getByRole('tabpanel', { name: 'Camera position' })).toBeVisible();
    fireEvent.keyDown(screen.getByRole('tab', { name: 'Camera position' }), { key: 'End' });
    expect(screen.getByRole('tab', { name: 'Reconstructed scene' })).toHaveFocus();
    fireEvent.keyDown(screen.getByRole('tab', { name: 'Reconstructed scene' }), { key: 'ArrowRight' });
    expect(captureTab).toHaveFocus();
    expect(captureTab).toHaveAttribute('aria-selected', 'true');
});

it('reconstructs the full capture without boundary start controls', async () => {
    const fixture = vision(0, { width: 1280, height: 720 });
    model.detect.mockResolvedValue(fixture.detections.segment);
    depthModel.detect.mockResolvedValue(fixture.detections.depth);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    await startCapture();
    fireEvent.click(cameraButton('Apply camera calibration'));
    const result = ref.current!.getLatestDetection()!;
    expect(result.reconstruction).toEqual(reconstructTrack({ ...fixture, calibration: result.calibration }));
    expect(result.reconstruction!.leftBoundary.length).toBeGreaterThan(0);
    expect(result).not.toHaveProperty('boundaryDetectionStartV');
    const capture = within(screen.getByRole('dialog', { name: 'Capture preview' }));
    expect(capture.queryByRole('slider', { name: 'Boundary start line' })).not.toBeInTheDocument();
    expect(capture.queryByRole('slider', { name: 'Boundary start' })).not.toBeInTheDocument();
    expect(capture.queryByLabelText('Projected ground grid')).not.toBeInTheDocument();
    selectStep('Reconstructed scene');
    expect(within(screen.getByRole('region', { name: 'Reconstructed scene' })).queryByRole('slider')).not.toBeInTheDocument();
    expect(model.detect).toHaveBeenCalledTimes(1);
});

it('publishes 2D boundaries without constructing display polygons or scene memory', async () => {
    const fixture = vision(0, { width: 1280, height: 720 });
    model.detect.mockResolvedValue(fixture.detections.segment);
    depthModel.detect.mockResolvedValue(fixture.detections.depth);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    await startCapture();
    const scene = ref.current!.getLatestDetection()!.reconstructedScene;
    expect(scene?.leftBoundary.length).toBeGreaterThan(0);
    expect(scene?.rightBoundary.length).toBeGreaterThan(0);
    fireEvent.click(cameraButton('Apply camera calibration'));
    expect(ref.current!.getLatestDetection()).not.toHaveProperty('sceneMemory');
    expect(ref.current!.getLatestDetection()!.reconstruction).not.toHaveProperty('masks');
    expect(ref.current!.getLatestDetection()!.reconstruction).not.toHaveProperty('pointCloud');
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(ref.current!.getLatestDetection()!.reconstructedScene).toEqual(scene);
    expect(screen.queryByLabelText('Perspective 3D masks')).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Camera height (m)'), { target: { value: '1.8' } });
    expect(ref.current!.getLatestDetection()!.reconstructedScene).toEqual(scene);
    fireEvent.click(screen.getByRole('button', { name: 'Stop capture' }));
    expect(ref.current!.getLatestDetection()).toBeNull();
    expect(screen.queryByLabelText('Rolling scene memory')).not.toBeInTheDocument();
// Exercises several captured frames and calibrated coaching geometry in Jest's VM.
}, 15000);

it.each(['restore', 'escape'])('keeps capture and calibration running while expanding and returning with %s', async (action) => {
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    await startCapture();
    fireEvent.click(cameraButton('Enable on capture'));
    fireEvent.click(cameraButton('Apply camera calibration'));
    const calibration = ref.current!.getLatestDetection()!.calibration;
    const preview = screen.getByLabelText('Capture preview');
    const canvas = screen.getByLabelText('Captured game frame with vision detections');
    const video = (HTMLMediaElement.prototype.play as jest.Mock).mock.instances[0] as HTMLVideoElement;
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
    expect(within(preview).queryByRole('slider', { name: 'Boundary start line' })).not.toBeInTheDocument();

    if (action === 'restore') fireEvent.click(screen.getByRole('button', { name: 'Restore capture' }));
    else fireEvent(preview, new Event('cancel', { cancelable: true }));
    expect(show).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('button', { name: 'Expand capture' })).toHaveAttribute('aria-expanded', 'false');
    expect(screen.getByLabelText('Captured game frame with vision detections')).toBe(canvas);
    expect(preview).toContainElement(video);
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

it('shows a 2D scene independent of the optional capture reference grid', async () => {
    model.detect.mockResolvedValue(vision(0).detections.segment);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    selectStep('Camera position');
    expect(screen.getByRole('group', { name: 'Camera position' })).toBeInTheDocument();
    expect(cameraButton('Enable on capture')).toBeDisabled();
    await startCapture();
    const capture = within(screen.getByLabelText('Capture preview'));
    fireEvent.click(cameraButton('Enable on capture'));
    expect(capture.getByLabelText('Projected ground grid')).toBeInTheDocument();
    expect(screen.getByLabelText('2D reconstructed scene')).toBeInTheDocument();
    expect(screen.queryByLabelText('Capture camera')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Camera field of view')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Left track boundary')).not.toBeEmptyDOMElement();
    const scene = ref.current!.getLatestDetection()!.reconstructedScene!;
    expect(scene.centerline.length).toBeGreaterThan(0);
    scene.centerline.forEach((line) => expect(screen.getByLabelText('Track middle line')).toContainHTML(
        `<polyline vector-effect="non-scaling-stroke" points="${line.map(({ x, y }) => `${x.toFixed(2)},${y.toFixed(2)}`).join(' ')}"></polyline>`));
    expect(ref.current!.getLatestDetection()?.reconstruction).toBeNull();
    fireEvent.click(cameraButton('Apply camera calibration'));
    const applied = ref.current!.getLatestDetection();
    expect(applied?.reconstruction?.cars).toHaveLength(1);
    expect(applied?.geometry).not.toBeNull();
    selectStep('Reconstructed scene');
    expect(screen.getByLabelText('2D reconstructed scene')).toBeVisible();
    expect(screen.queryByRole('button', { name: 'Reset view' })).not.toBeInTheDocument();
    expect(ref.current!.getLatestDetection()).toBe(applied);
    fireEvent.click(cameraButton('Disable on capture'));
    expect(capture.queryByLabelText('Projected ground grid')).not.toBeInTheDocument();
    expect(ref.current!.getLatestDetection()).toBe(applied);
    expect(model.detect).toHaveBeenCalledTimes(1);
    expect(track.stop).not.toHaveBeenCalled();
});

it('publishes metric geometry and positions only after camera calibration is applied', async () => {
    model.detect.mockResolvedValue(vision(0, { classNames: MODEL_LABELS }).detections.segment);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    await startCapture();
    expect(screen.getByLabelText('Driver position')).toHaveTextContent('Unknown');
    expect(ref.current!.getLatestDetection()!.geometry).toBeNull();
    expect(screen.getByLabelText('Reconstructed scene status')).toHaveTextContent('Visible track boundaries in 2D');
    const capturedAt = ref.current!.getLatestDetection()!.capturedAt;
    const listener = jest.fn();
    ref.current!.subscribeDetection(listener);
    fireEvent.click(cameraButton('Apply camera calibration'));
    expect(ref.current!.getLatestDetection()).toMatchObject({ capturedAt, analysis: {
        driverPosition: { leftBoundaryDistanceM: expect.any(Number), rightBoundaryDistanceM: expect.any(Number) },
        carAhead: 1, opponents: [expect.objectContaining({ longitudinalOffsetM: expect.any(Number), lateralOffsetM: expect.any(Number) })],
    } });
    expect(ref.current!.getLatestDetection()!.geometry!.trackWidthM).toBeCloseTo(10, 0);
    expect(listener).toHaveBeenCalled();
    expect(screen.getByLabelText('Driver position')).toHaveTextContent(/Left boundary: [\d.]+ mRight boundary: [\d.]+ m/);
    expect(screen.getByLabelText('Opponent positions')).toHaveTextContent(/m ahead · [\d.]+ m right/);
    expect(screen.queryByLabelText('Visible corner')).not.toBeInTheDocument();
    expect(ref.current!.getLatestDetection()!.analysis).not.toHaveProperty('cornerDirection');
    fireEvent.change(screen.getByLabelText('Camera right of car center (m)'), { target: { value: '-2.5' } });
    expect(ref.current!.getLatestDetection()!.analysis).toEqual({});
    expect(ref.current!.getLatestDetection()!.geometry).toBeNull();
    fireEvent.click(cameraButton('Apply camera calibration'));
    expect(ref.current!.getLatestDetection()!.capturedAt).toBe(capturedAt);
    expect(ref.current!.getLatestDetection()!.analysis!.driverPosition!.leftBoundaryDistanceM).toBeCloseTo(5, 0);
    fireEvent.click(screen.getByRole('button', { name: 'Stop capture' }));
    expect(ref.current!.getLatestDetection()).toBeNull();
    expect(screen.getByLabelText('Driver position')).toHaveTextContent('Unknown');
    expect(screen.queryByLabelText('Perspective 3D masks')).not.toBeInTheDocument();
});

it('shows a list of opponents relative to the driver and clears it when capture stops', async () => {
    const frame = multipleCarVision();
    model.detect.mockResolvedValue(frame.detections.segment);
    depthModel.detect.mockResolvedValue(frame.detections.depth);
    render(<LiveTrackVision name="vision" />);
    await flush();
    await startCapture();
    fireEvent.click(cameraButton('Apply camera calibration'));
    selectStep('Reconstructed scene');
    const opponents = screen.getByLabelText('Opponent positions');
    const items = within(opponents).getAllByRole('listitem');
    expect(items).toHaveLength(3);
    expect(items[0]).toHaveTextContent(/m ahead · [\d.]+ m left/);
    expect(items[1]).toHaveTextContent(/m ahead · [\d.]+ m right/);
    expect(items[2]).toHaveTextContent('Aligned with driver');
    expect(screen.queryByText('Visible corner')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Stop capture' }));
    expect(within(opponents).queryByRole('list')).not.toBeInTheDocument();
    expect(opponents).toHaveTextContent('Unknown');
});

it('keeps the reconstructed scene visible during pending inference while positions expire', async () => {
    model.detect.mockResolvedValue(vision(0).detections.segment);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    await startCapture();
    fireEvent.click(cameraButton('Enable on capture'));
    fireEvent.click(cameraButton('Apply camera calibration'));
    selectStep('Reconstructed scene');
    const edges = screen.getByLabelText('Left track boundary');
    expect(edges).not.toBeEmptyDOMElement();
    const displayed = edges.innerHTML;
    const displayedCars = screen.getByLabelText('Reconstructed cars').innerHTML;
    expect(screen.getByLabelText('Driver position')).not.toHaveTextContent('Unknown');
    const capturedAt = ref.current!.getLatestDetection()!.capturedAt;
    const pending = deferred<typeof detection>();
    model.detect.mockReturnValueOnce(pending.promise);
    await act(async () => { jest.advanceTimersByTime(VISION_MAX_AGE_MS + 1); });
    expect(screen.getByLabelText('Driver position')).toHaveTextContent('Unknown');
    expect(screen.getByLabelText('Reconstructed scene status')).toHaveTextContent('Showing last frame (stale)');
    expect(screen.getByLabelText('Opponent positions')).toHaveTextContent('Unknown');
    expect(screen.getByLabelText('Reconstructed cars').innerHTML).toBe(displayedCars);
    expect(edges.innerHTML).toBe(displayed);
    fireEvent.click(cameraButton('Apply camera calibration'));
    expect(screen.getByLabelText('Driver position')).toHaveTextContent('Unknown');
    expect(ref.current!.getLatestDetection()!.capturedAt).toBe(capturedAt);
    fireEvent.click(screen.getByRole('button', { name: 'Stop capture' }));
    await act(async () => { pending.resolve(detection); });
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(ref.current!.getLatestDetection()).toBeNull();
    expect(screen.queryByLabelText('Left track boundary')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Reconstructed cars')).not.toBeInTheDocument();
});

it('shows all cars and car packs independently of track visibility, calibration and the display label', async () => {
    const frame = vision(0, { width: 1280, height: 720, classNames: MODEL_LABELS, road: () => false,
        cars: [[-0.04, 0.7, 0.12, 0.95], [0.3, 0.4, 0.5, 0.6], [0.6, 0.3, 0.95, 0.55]] });
    const segment = frame.detections.segment!;
    if (segment.task !== 'segment') throw new Error('segment');
    segment.instances[3].classId = MODEL_LABELS.indexOf('car pack');
    segment.instances[3].confidence = 0.85;
    segment.instances.push({ ...segment.instances[1], confidence: 0.6 });
    model.classNames = MODEL_LABELS;
    model.detect.mockResolvedValue(segment);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    await startCapture();
    selectStep('Segmentation');
    fireEvent.change(screen.getByLabelText('Display label'), { target: { value: 'track' } });
    selectStep('Reconstructed scene');
    const cars = screen.getByLabelText('Reconstructed cars');
    expect(cars).toBeVisible();
    const individualCars = within(cars).getAllByLabelText('Car · 90% confidence');
    expect(individualCars).toHaveLength(2);
    expect(within(cars).getAllByLabelText(/^Car(?: pack)? · \d+% confidence$/)).toHaveLength(3);
    expect(within(cars).getByLabelText('Car pack · 85% confidence').innerHTML).toContain('stroke-dasharray="6 4"');
    const [left, top, right] = ref.current!.getLatestDetection()!.reconstructedScene!.cars[0].box;
    expect(left).toBe(0);
    expect(top).toBeCloseTo(504);
    expect(right - left).toBeCloseTo(153.6);
    expect(individualCars[0].innerHTML).toContain(`<rect x="${left}" y="${top}" width="${right - left}"`);
    expect(screen.getByLabelText('Reconstructed scene status')).toHaveTextContent('Detected cars and car packs in 2D');
    expect(ref.current!.getLatestDetection()?.reconstruction).toBeNull();
    expect(ref.current!.getLatestDetection()?.reconstructedScene?.cars).toHaveLength(3);
    expect(depthModel.detect).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole('button', { name: 'Stop capture' }));
    expect(screen.queryByLabelText('Reconstructed cars')).not.toBeInTheDocument();
});

it('validates camera settings and previews height and angle changes without rerunning inference', async () => {
    const fixture = vision(0, { width: 1280, height: 720 });
    model.detect.mockResolvedValue(fixture.detections.segment);
    depthModel.detect.mockResolvedValue(fixture.detections.depth);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    expect(cameraButton('Apply camera calibration')).toBeDisabled();
    await startCapture();
    fireEvent.click(cameraButton('Enable on capture'));
    const capturedAt = ref.current!.getLatestDetection()!.capturedAt;
    const originalGrid = screen.getByLabelText('Projected ground grid').innerHTML;
    const localView = screen.getByLabelText('2D reconstructed scene');
    const originalScene = localView.innerHTML;
    expect(within(localView).getByLabelText('Left track boundary')).not.toBeEmptyDOMElement();
    expect(localView).toHaveAttribute('viewBox', '0 0 1280 720');
    selectStep('Camera position');
    fireEvent.change(screen.getByLabelText('Camera height (m)'), { target: { value: '1.8' } });
    // Camera settings affect coaching geometry, while the 2D scene stays in image space.
    expect(localView.innerHTML).toBe(originalScene);
    selectStep('Camera position');
    fireEvent.change(screen.getByLabelText('Pitch down (°)'), { target: { value: '8' } });
    expect(screen.getByLabelText('Projected ground grid').innerHTML).not.toBe(originalGrid);
    expect(localView.innerHTML).toBe(originalScene);
    expect(model.detect).toHaveBeenCalledTimes(1);
    fireEvent.click(cameraButton('Apply camera calibration'));
    expect(ref.current!.getLatestDetection()).toMatchObject({ capturedAt, calibration: { heightM: 1.8, pitchDeg: 8 } });
    expect(localView.innerHTML).toBe(originalScene);
    fireEvent.change(screen.getByLabelText('Camera height (m)'), { target: { value: '' } });
    expect(ref.current!.getLatestDetection()?.calibration).toBeUndefined();
    expect(cameraButton('Apply camera calibration')).toBeDisabled();
    expect(screen.queryByLabelText('Projected ground grid')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Perspective 3D masks')).not.toBeInTheDocument();
    fireEvent.click(cameraButton('Disable on capture'));
    expect(cameraButton('Enable on capture')).toBeDisabled();
    fireEvent.change(screen.getByLabelText('Camera height (m)'), { target: { value: '0' } });
    expect(cameraButton('Apply camera calibration')).toBeDisabled();
    fireEvent.change(screen.getByLabelText('Camera height (m)'), { target: { value: '1.8' } });
    fireEvent.click(cameraButton('Apply camera calibration'));
    fireEvent.click(cameraButton('Clear calibration'));
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
    fireEvent.click(cameraButton('Apply camera calibration'));
    await act(async () => { pending.resolve(detection); });
    expect(ref.current!.getLatestDetection()?.calibration?.heightM).toBe(1.8);
    jest.spyOn(HTMLVideoElement.prototype, 'videoWidth', 'get').mockReturnValue(1920);
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(ref.current!.getLatestDetection()?.calibration).toBeUndefined();
    fireEvent.click(cameraButton('Apply camera calibration'));
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
    fireEvent.click(cameraButton('Apply camera calibration'));
    expect(cameraButton('Clear calibration')).toBeEnabled();
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
    fireEvent.change(screen.getByRole('combobox', { name: 'Game window or screen' }), { target: { value: 'window:42' } });
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
    expect(screen.getByRole('status')).toHaveTextContent('Waiting for segmentation and depth');
    expect(depthModel.detect).not.toHaveBeenCalled();
    expect(model.detect).not.toHaveBeenCalled();
    selectStep('Reconstructed scene');
    expect(screen.getByLabelText('Captured window scene')).toBeVisible();
    expect(screen.queryByLabelText('Left track boundary')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Right track boundary')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Track middle line')).not.toBeInTheDocument();
});

it('loads the backend model automatically and detects without a file upload', async () => {
    const ref = React.createRef<TrackVisionHandle>();
    const { unmount } = render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    expect(TrackVisionModel.loadBackend).toHaveBeenCalledTimes(1);
    expect(TrackVisionModel.loadBackend).toHaveBeenCalledWith(768);
    expect(screen.queryByRole('checkbox', { name: 'Allow CPU fallback' })).not.toBeInTheDocument();
    expect(screen.getByLabelText('Segmentation inference device')).toHaveTextContent('GPU acceleration active');
    expect(screen.getByLabelText('Model labels')).toHaveTextContent('track, curb');
    expect(screen.queryByRole('checkbox', { name: 'Enable Depth' })).not.toBeInTheDocument();
    expect(screen.queryByRole('checkbox', { name: 'Enable Segmentation' })).not.toBeInTheDocument();
    expect(screen.queryByText('Model settings')).not.toBeInTheDocument();
    expect(TrackVisionModel.loadBuiltin).toHaveBeenCalledWith('depth', 518);
    expect(screen.queryByRole('checkbox', { name: 'Enable Semantic' })).not.toBeInTheDocument();
    await startCapture();
    expect(model.detect).toHaveBeenCalledTimes(1);
    expect(ref.current!.getLatestDetection()?.detections.segment).toEqual(detection);
    unmount();
    expect(model.dispose).toHaveBeenCalledTimes(1);
});

it('filters the current and future preview frames without changing detection or analysis', async () => {
    const fixture = vision(0, { classNames: MODEL_LABELS });
    model.classNames = MODEL_LABELS;
    model.detect.mockResolvedValue(fixture.detections.segment);
    depthModel.detect.mockResolvedValue(fixture.detections.depth);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    selectStep('Segmentation');
    const selector = screen.getByRole('combobox', { name: 'Display label' });
    expect(selector).toHaveValue('');
    expect(within(selector).getAllByRole('option').map((option) => option.textContent)).toEqual(['All labels', ...MODEL_LABELS]);
    await startCapture();
    fireEvent.click(cameraButton('Apply camera calibration'));
    const result = ref.current!.getLatestDetection()!;
    expect(result.geometry).not.toBeNull();
    const listener = jest.fn();
    ref.current!.subscribeDetection(listener);
    const preview = screen.getByLabelText('Captured game frame with vision detections') as HTMLCanvasElement;
    const context = preview.getContext('2d')!;
    selectStep('Segmentation');
    const drawsBefore = (context.drawImage as jest.Mock).mock.calls.length;

    fireEvent.change(selector, { target: { value: 'car' } });
    expect(context.drawImage).toHaveBeenCalledTimes(drawsBefore + 1);
    expect(drawVisionOverlay).toHaveBeenLastCalledWith(context, result, 'car');
    expect(ref.current!.getLatestDetection()).toBe(result);
    expect(listener).not.toHaveBeenCalled();
    expect(model.detect).toHaveBeenCalledTimes(1);
    expect(depthModel.detect).toHaveBeenCalledTimes(1);

    await act(async () => { jest.advanceTimersByTime(200); });
    expect(drawVisionOverlay).toHaveBeenLastCalledWith(context, expect.objectContaining({ detections: result.detections }), 'car');
    expect(ref.current!.getLatestDetection()!.analysis).toEqual(result.analysis);
    expect(selector).toHaveValue('car');
    fireEvent.change(selector, { target: { value: '' } });
    expect(drawVisionOverlay).toHaveBeenLastCalledWith(context, ref.current!.getLatestDetection(), '');
    expect(TrackVisionModel.loadBackend).toHaveBeenCalledTimes(1);
    expect(model.dispose).not.toHaveBeenCalled();
    expect(track.stop).not.toHaveBeenCalled();
});

it('disables label selection without segmentation and resets labels missing from a reloaded model', async () => {
    render(<LiveTrackVision name="vision" />);
    selectStep('Segmentation');
    const selector = screen.getByRole('combobox', { name: 'Display label' });
    expect(selector).toBeDisabled();
    await flush();
    expect(selector).toBeEnabled();
    fireEvent.change(selector, { target: { value: 'curb' } });
    expect(selector).toHaveValue('curb');
    model.detect.mockRejectedValueOnce(new Error('GPU device lost'));
    await startCapture();
    expect(selector).toBeDisabled();
    (TrackVisionModel.loadBackend as jest.Mock).mockResolvedValue({ ...model, classNames: ['track', 'grass'] });
    fireEvent.click(screen.getByRole('button', { name: 'Retry Segmentation' }));
    await flush();
    expect(selector).toBeEnabled();
    expect(selector).toHaveValue('');
    expect(within(selector).queryByRole('option', { name: 'curb' })).not.toBeInTheDocument();
    expect(within(selector).getByRole('option', { name: 'grass' })).toBeInTheDocument();
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

it('runs depth alongside backend segmentation and releases both models on unmount', async () => {
    const depthResult = { task: 'depth', width: 2, height: 2, values: new Float32Array([1, 2, 3, 4]), inferenceMs: 20, classNames: [] };
    const depth = { ...model, name: 'Depth-Anything-V2-Small', classNames: [], detect: jest.fn().mockResolvedValue(depthResult), dispose: jest.fn().mockResolvedValue(undefined) };
    (TrackVisionModel.loadBuiltin as jest.Mock).mockResolvedValue(depth);
    const ref = React.createRef<TrackVisionHandle>();
    const { unmount } = render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    expect(TrackVisionModel.loadBuiltin).toHaveBeenCalledWith('depth', 518);
    expect(TrackVisionModel.loadBackend).toHaveBeenCalledTimes(1);
    expect(screen.getByLabelText('Depth inference device')).toHaveTextContent('GPU acceleration active');
    expect(screen.queryByRole('group', { name: /Depth range/ })).not.toBeInTheDocument();
    expect(screen.queryByRole('slider', { name: 'Close depth' })).not.toBeInTheDocument();
    expect(screen.queryByRole('slider', { name: 'Far depth' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Reset depth range' })).not.toBeInTheDocument();
    await startCapture();
    expect(ref.current!.getLatestDetection()?.detections).toEqual({ segment: detection, depth: depthResult });
    selectStep('Segmentation');
    const preview = screen.getByLabelText('Captured game frame with vision detections') as HTMLCanvasElement;
    expect(drawVisionOverlay).toHaveBeenLastCalledWith(preview.getContext('2d'), expect.objectContaining({
        detections: { segment: detection, depth: depthResult },
    }), '');
    expect(depth.detect.mock.calls[0][0]).toBe(model.detect.mock.calls[0][0]);
    unmount();
    expect(ref.current).toBeNull();
    expect(depth.dispose).toHaveBeenCalledTimes(1);
    expect(model.dispose).toHaveBeenCalledTimes(1);
    expect(track.stop).toHaveBeenCalledTimes(1);
});

it('loads depth but waits for masks when backend segmentation cannot load', async () => {
    const depthResult = { task: 'depth', width: 2, height: 2, values: new Float32Array([1, 2, 3, 4]), inferenceMs: 20, classNames: [] };
    const depth = { ...model, detect: jest.fn().mockResolvedValue(depthResult), dispose: jest.fn().mockResolvedValue(undefined) };
    (TrackVisionModel.loadBackend as jest.Mock).mockRejectedValue(new Error('Backend unavailable'));
    (TrackVisionModel.loadBuiltin as jest.Mock).mockResolvedValue(depth);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    await startCapture();
    expect(ref.current!.getLatestDetection()).toBeNull();
    expect(depth.detect).not.toHaveBeenCalled();
    expect(screen.getByRole('alert')).toHaveTextContent('Backend unavailable');
    expect(track.stop).not.toHaveBeenCalled();
});

it('waits for same-frame masks and preserves the full image including the interior for depth', async () => {
    const pending = deferred<typeof detection>();
    const segmented = { ...detection, classNames: ['track', 'car interior', 'fence'], instances: [
        { ...detection.instances[0], mask: new Uint8Array([1, 1, 0, 0]) },
        { ...detection.instances[0], classId: 1, confidence: 0.6, mask: new Uint8Array([0, 1, 0, 0]) },
        { ...detection.instances[0], classId: 2, mask: new Uint8Array([0, 0, 0, 1]) },
    ] };
    model.classNames = segmented.classNames;
    model.detect.mockReturnValue(pending.promise);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    selectStep('Segmentation');
    fireEvent.change(screen.getByRole('combobox', { name: 'Display label' }), { target: { value: 'track' } });
    await startCapture();
    expect(depthModel.detect).not.toHaveBeenCalled();
    fireEvent.change(screen.getByRole('slider', { name: 'Segmentation confidence' }), { target: { value: '0.95' } });
    await act(async () => { pending.resolve(segmented); });
    expect(depthModel.detect).toHaveBeenCalledWith(model.detect.mock.calls[0][0], 0.5, expect.objectContaining({
        width: 2, height: 2, mask: new Uint8Array([1, 1, 1, 1]),
    }));
    expect(ref.current!.getLatestDetection()?.detections.segment).toBe(segmented);
});

it.each(['empty', 'interior only', 'entire image is interior'])('runs depth on the full image when segmentation is %s', async (reason) => {
    model.detect.mockResolvedValue({ ...detection, classNames: ['car interior'], instances: [
        { ...detection.instances[0], mask: new Uint8Array([0, 1, 0, 0]) },
    ] });
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    await startCapture();
    expect(depthModel.detect).toHaveBeenLastCalledWith(expect.any(HTMLCanvasElement), 0.5, expect.objectContaining({
        mask: new Uint8Array([1, 1, 1, 1]),
    }));
    model.detect.mockResolvedValue({ ...detection, classNames: ['car interior'], instances: reason === 'empty' ? [] : [
        { ...detection.instances[0], mask: new Uint8Array(reason === 'entire image is interior' ? [1, 1, 1, 1] : [0, 0, 1, 0]) },
    ] });
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(depthModel.detect).toHaveBeenCalledTimes(2);
    expect(depthModel.detect).toHaveBeenLastCalledWith(expect.any(HTMLCanvasElement), 0.5, expect.objectContaining({
        width: 2, height: 2, mask: new Uint8Array([1, 1, 1, 1]),
    }));
    expect(ref.current!.getLatestDetection()?.detections.depth).toBeDefined();
    expect(screen.getByRole('status')).not.toHaveTextContent('waiting');
});

it('does not reuse old masks for depth after segmentation fails', async () => {
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    await startCapture();
    expect(depthModel.detect).toHaveBeenCalledTimes(1);
    model.detect.mockRejectedValue(new Error('Segmentation failed'));
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(depthModel.detect).toHaveBeenCalledTimes(1);
    expect(ref.current!.getLatestDetection()?.detections.depth).toBeUndefined();
    expect(ref.current!.getLatestDetection()).toBeNull();
    expect(screen.getByRole('status')).toHaveTextContent('Waiting for segmentation and depth');
    expect(track.stop).not.toHaveBeenCalled();
});

it('pauses analysis and further inference when depth inference fails', async () => {
    const depth = { ...model, detect: jest.fn().mockRejectedValue(new Error('Depth inference failed')), dispose: jest.fn().mockResolvedValue(undefined) };
    (TrackVisionModel.loadBuiltin as jest.Mock).mockResolvedValue(depth);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    await startCapture();
    expect(ref.current!.getLatestDetection()).toBeNull();
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(model.detect).toHaveBeenCalledTimes(1);
    expect(depth.detect).toHaveBeenCalledTimes(1);
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
    expect(TrackVisionModel.loadBackend).toHaveBeenLastCalledWith(768);
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

it('cancels scheduled GPU retries on unmount', async () => {
    (TrackVisionModel.loadBackend as jest.Mock).mockRejectedValue(new GpuInferenceError('GPU device lost'));
    const { unmount } = render(<LiveTrackVision name="vision" />);
    await flush();
    unmount();
    await flush();
    await act(async () => { jest.advanceTimersByTime(10000); });
    expect(TrackVisionModel.loadBackend).toHaveBeenCalledTimes(1);
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
    expect(TrackVisionModel.loadBackend).toHaveBeenLastCalledWith(768);
    expect(ref.current!.getLatestDetection()?.detections.segment).toEqual(detection);
    expect(track.stop).not.toHaveBeenCalled();
});

it('applies filtering confidence to current and pending frames, reconstruction and coaching without reloading models', async () => {
    const fixture = vision(0, { width: 1280, height: 720 });
    model.detect.mockResolvedValue(fixture.detections.segment);
    depthModel.detect.mockResolvedValue(fixture.detections.depth);
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    const slider = screen.getByRole('slider', { name: 'Filtering confidence' });
    expect(slider).toHaveValue('0.65');
    await startCapture();
    fireEvent.click(cameraButton('Apply camera calibration'));
    const initial = ref.current!.getLatestDetection()!;
    expect(initial.geometry).not.toBeNull();
    expect(initial.reconstructedScene!.cars).toHaveLength(1);
    const listener = jest.fn();
    ref.current!.subscribeDetection(listener);
    const pending = deferred<typeof fixture.detections.segment>();
    model.detect.mockReturnValueOnce(pending.promise);
    await act(async () => { jest.advanceTimersByTime(200); });

    fireEvent.change(slider, { target: { value: '0.95' } });
    const filtered = ref.current!.getLatestDetection()!;
    expect(filtered.filterConfidence).toBe(0.95);
    expect(filtered.detections).toBe(initial.detections);
    expect(filtered.capturedAt).toBe(initial.capturedAt);
    expect(filtered.geometry).toBeNull();
    expect(filtered.analysis).toEqual({});
    expect(filtered.reconstructedScene).toMatchObject({ cars: [], leftBoundary: [], rightBoundary: [] });
    expect(listener).toHaveBeenCalledTimes(1);
    selectStep('Filtering');
    expect(screen.getByLabelText('Applied filters')).toHaveTextContent('Confidence ≥ 95%');
    expect(within(screen.getByRole('tabpanel')).getAllByRole('definition')[1]).toHaveTextContent('0');
    await act(async () => { pending.resolve(fixture.detections.segment); });
    expect(ref.current!.getLatestDetection()!.filterConfidence).toBe(0.95);
    expect(ref.current!.getLatestDetection()!.reconstructedScene!.cars).toHaveLength(0);

    fireEvent.change(slider, { target: { value: '0.65' } });
    expect(ref.current!.getLatestDetection()!.geometry).toEqual(initial.geometry);
    expect(ref.current!.getLatestDetection()!.analysis).toEqual(initial.analysis);
    expect(ref.current!.getLatestDetection()!.reconstructedScene).toEqual(initial.reconstructedScene);
    expect(TrackVisionModel.loadBackend).toHaveBeenCalledTimes(1);
    expect(TrackVisionModel.loadBuiltin).toHaveBeenCalledTimes(1);
    expect(model.detect).toHaveBeenCalledTimes(2);
    expect(depthModel.detect).toHaveBeenCalledTimes(2);
    expect(track.stop).not.toHaveBeenCalled();
}, 15000);

it.each(['Segmentation', 'Depth'])('waits for %s to load before running either model and resumes after retry', async (label) => {
    const loader = label === 'Segmentation' ? TrackVisionModel.loadBackend : TrackVisionModel.loadBuiltin;
    (loader as jest.Mock).mockRejectedValueOnce(new Error('Weights unavailable'));
    const ref = React.createRef<TrackVisionHandle>();
    render(<LiveTrackVision ref={ref} name="vision" />);
    await flush();
    await startCapture();
    expect(model.detect).not.toHaveBeenCalled();
    expect(depthModel.detect).not.toHaveBeenCalled();
    expect(ref.current!.getLatestDetection()).toBeNull();
    expect(screen.getByLabelText('Captured game frame with vision detections')).toBeVisible();
    selectStep('Reconstructed scene');
    const background = screen.getByLabelText('Captured window scene') as HTMLCanvasElement;
    const preview = screen.getByLabelText('Captured game frame with vision detections') as HTMLCanvasElement;
    const backgroundDraws = (background.getContext('2d')!.drawImage as jest.Mock).mock.calls.length;
    const previewDraws = (preview.getContext('2d')!.drawImage as jest.Mock).mock.calls.length;
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(background.getContext('2d')!.drawImage).toHaveBeenCalledTimes(backgroundDraws + 1);
    expect(preview.getContext('2d')!.drawImage).toHaveBeenCalledTimes(previewDraws + 1);
    expect(model.detect).not.toHaveBeenCalled();
    expect(depthModel.detect).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: `Retry ${label}` }));
    await flush();
    await act(async () => { jest.advanceTimersByTime(200); });
    expect(ref.current!.getLatestDetection()?.detections).toEqual({ segment: detection, depth: vision(0).detections.depth });
    expect(model.detect).toHaveBeenCalledTimes(1);
    expect(depthModel.detect).toHaveBeenCalledTimes(1);
    expect(model.dispose).not.toHaveBeenCalled();
    expect(depthModel.dispose).not.toHaveBeenCalled();
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
