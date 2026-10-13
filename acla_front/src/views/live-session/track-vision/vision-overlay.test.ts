import { drawVisionOverlay } from './vision-overlay';
import { TrackVisionFrame } from './track-vision-types';

afterEach(() => { jest.restoreAllMocks(); });

it.each(['track', 'car', 'grass'])('shows only observed pixels for %s even when depth supports hidden predictions', (label) => {
    const width = 10, pixels = new Uint8ClampedArray(400);
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
        createImageData: () => ({ data: pixels }), putImageData: jest.fn(),
    } as any);
    const context = { save: jest.fn(), restore: jest.fn(), drawImage: jest.fn(), strokeRect: jest.fn(), fillText: jest.fn(),
    } as unknown as CanvasRenderingContext2D;
    const target = Uint8Array.from({ length: 100 }, (_, i) => Number(i % width >= 2 && i % width <= 7
        && Math.floor(i / width) >= 2 && Math.floor(i / width) <= 7 && i % width !== 5));
    const foreground = Uint8Array.from(target, (_, i) => Number(i % width === 5));
    drawVisionOverlay(context, { capturedAt: 0, width: 640, height: 640, detections: {
        segment: { task: 'segment', width, height: width, classNames: [label, 'occluder'], inferenceMs: 0, instances: [
            { classId: 0, confidence: 0.9, box: [0.2, 0.2, 0.8, 0.8], mask: target },
            { classId: 1, confidence: 0.9, box: [0.5, 0, 0.6, 1], mask: foreground },
        ] },
        depth: { task: 'depth', width, height: width, classNames: [], inferenceMs: 0,
            values: Float32Array.from(target, (_, i) => foreground[i] ? 5 : 10) },
    } }, label);
    expect(Array.from(pixels.slice(45 * 4, 46 * 4))).toEqual([0, 0, 0, 0]);
    expect(Array.from(pixels.slice(44 * 4, 45 * 4))).toEqual([55, 239, 172, 110]);
    expect(target[45]).toBe(0);
    expect(context.fillText).toHaveBeenCalledTimes(1);
});

it('does not paint depth data over the captured frame', () => {
    const getContext = jest.spyOn(HTMLCanvasElement.prototype, 'getContext');
    const context = { save: jest.fn(), restore: jest.fn(), drawImage: jest.fn() } as unknown as CanvasRenderingContext2D;
    const values = new Float32Array([1, 10, 30, 60]);
    const result: TrackVisionFrame = {
        capturedAt: 1, width: 1280, height: 720,
        detections: { depth: { task: 'depth', width: 2, height: 2, values, inferenceMs: 1, classNames: [] } },
    };
    drawVisionOverlay(context, result);
    expect(getContext).not.toHaveBeenCalled();
    expect(context.drawImage).not.toHaveBeenCalled();
    expect(context.save).not.toHaveBeenCalled();
    expect(Array.from(values)).toEqual([1, 10, 30, 60]);
});

it.each([false, true])('draws segmentation labels and masks with letterbox padding removed (depth enabled: %s)', (withDepth) => {
    const pixels = new Uint8ClampedArray(16);
    const layerContext = { createImageData: () => ({ data: pixels }), putImageData: jest.fn() };
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(layerContext as any);
    const context = {
        save: jest.fn(), restore: jest.fn(), drawImage: jest.fn(), strokeRect: jest.fn(), fillText: jest.fn(),
    } as unknown as CanvasRenderingContext2D;
    const result: TrackVisionFrame = {
        capturedAt: 1, width: 1280, height: 720, detections: { segment: {
            task: 'segment', width: 2, height: 2, inferenceMs: 1, classNames: ['curb', 'track surface'],
            instances: [{ classId: 1, confidence: 0.9, box: [0, 0, 1, 1], mask: new Uint8Array([1, 0, 1, 0]) }],
        } },
    };
    if (withDepth) result.detections.depth = {
        task: 'depth', width: 2, height: 2, values: new Float32Array([1, 10, 30, 60]), inferenceMs: 1, classNames: [],
    };
    drawVisionOverlay(context, result);
    expect(context.drawImage).toHaveBeenCalledTimes(1);
    expect(context.drawImage).toHaveBeenCalledWith(expect.any(HTMLCanvasElement), 0, 0.4375, 2, 1.125, 0, 0, 1280, 720);
    expect(context.fillText).toHaveBeenCalledWith('track surface · 90%', 4, 16);
    expect(context.strokeRect).toHaveBeenCalledWith(0, 0, 1280, 720);
    expect(Array.from(pixels.slice(0, 4))).toEqual([87, 185, 255, 110]);
    expect(Array.from(pixels.slice(4, 8))).toEqual([0, 0, 0, 0]);
    jest.restoreAllMocks();
});

it.each(['track', 'car', 'grass'])('shows only the selected %s label without subtracting other detections', (label) => {
    const pixels = new Uint8ClampedArray(16);
    const layerContext = { createImageData: () => ({ data: pixels }), putImageData: jest.fn() };
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(layerContext as any);
    const context = {
        save: jest.fn(), restore: jest.fn(), drawImage: jest.fn(), strokeRect: jest.fn(), fillText: jest.fn(),
    } as unknown as CanvasRenderingContext2D;
    const instances = [
        { classId: 0, confidence: 0.9, box: [0, 0, 1, 1] as [number, number, number, number], mask: new Uint8Array([1, 1, 1, 0]) },
        { classId: 1, confidence: 0.9, box: [0, 0, 1, 1] as [number, number, number, number], mask: new Uint8Array([1, 0, 0, 0]) },
        { classId: 2, confidence: 0.9, box: [0, 0, 1, 1] as [number, number, number, number], mask: new Uint8Array([0, 1, 0, 0]) },
    ];
    const original = instances.map((instance) => ({ ...instance, mask: instance.mask.slice() }));
    drawVisionOverlay(context, {
        capturedAt: 1, width: 640, height: 640, detections: { segment: {
            task: 'segment', width: 2, height: 2, inferenceMs: 1, classNames: ['track', 'curb', 'car', 'grass'], instances,
        } },
    }, label);
    const expected = label === 'track'
        ? [55, 239, 172, 110, 55, 239, 172, 110, 55, 239, 172, 110, 0, 0, 0, 0]
        : label === 'car' ? [0, 0, 0, 0, 255, 190, 87, 110, 0, 0, 0, 0, 0, 0, 0, 0] : Array(16).fill(0);
    expect(Array.from(pixels)).toEqual(expected);
    expect(context.strokeRect).toHaveBeenCalledTimes(label === 'grass' ? 0 : 1);
    expect(context.fillText).toHaveBeenCalledTimes(label === 'grass' ? 0 : 1);
    expect((context.fillText as jest.Mock).mock.calls).toEqual(label === 'grass' ? [] : [[`${label} · 90%`, 4, 16]]);
    expect(instances).toEqual(original);
});

it.each(['car', 'car pack', 'curb', 'grass', 'other', 'fence', 'sand', 'Outfield asphalt road', 'car interior'])
('composites both raw masks where %s overlaps track, preserving detection order', (label) => {
    const pixels = new Uint8ClampedArray(16);
    const layerContext = { createImageData: () => ({ data: pixels }), putImageData: jest.fn() };
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(layerContext as any);
    const context = {
        save: jest.fn(), restore: jest.fn(), drawImage: jest.fn(), strokeRect: jest.fn(), fillText: jest.fn(),
    } as unknown as CanvasRenderingContext2D;
    for (const trackLabel of ['track', ' ROAD ', '\tAsPhAlT ', 'tarmac']) {
        for (const trackFirst of [true, false]) {
            pixels.fill(0);
            const foreground = { classId: 0, confidence: 0.95, box: [0, 0, 1, 1] as [number, number, number, number],
                mask: new Uint8Array([1, 1, 0, 0]) };
            const track = { ...foreground, classId: 1, confidence: 0.8, mask: new Uint8Array([1, 0, 1, 0]) };
            const instances = trackFirst ? [track, foreground] : [foreground, track];
            const originalOrder = instances.slice();
            drawVisionOverlay(context, {
                capturedAt: 1, width: 640, height: 640, detections: { segment: {
                    task: 'segment', width: 2, height: 2, inferenceMs: 1, classNames: [label, trackLabel], instances,
                } },
            });
            const overlapping = trackFirst ? [67, 219, 202, 173] : [75, 205, 225, 173];
            expect(Array.from(pixels)).toEqual([
                ...overlapping, 55, 239, 172, 110, 87, 185, 255, 110, 0, 0, 0, 0,
            ]);
            expect(instances).toEqual(originalOrder);
            expect(track.mask).toEqual(new Uint8Array([1, 0, 1, 0]));
            expect(foreground.mask).toEqual(new Uint8Array([1, 1, 0, 0]));
        }
    }
});

it.each(['', 'car interior', 'track'])('shows raw interior and track detections with display label "%s"', (displayLabel) => {
    const width = 20, height = 7, pixels = new Uint8ClampedArray(width * height * 4);
    const layerContext = { createImageData: () => ({ data: pixels }), putImageData: jest.fn() };
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(layerContext as any);
    const context = { save: jest.fn(), restore: jest.fn(), drawImage: jest.fn(), strokeRect: jest.fn(), fillText: jest.fn(),
    } as unknown as CanvasRenderingContext2D;
    const track = { classId: 0, confidence: 0.9, box: [0, 0, 1, 1] as [number, number, number, number],
        mask: Uint8Array.from({ length: width * height }, (_, i) => Number(Math.floor(i / width) >= 3
            || (i % width >= 6 && i % width < 14))) };
    const cockpit = { ...track, classId: 1, confidence: 0.6, mask: Uint8Array.from({ length: width * height }, (_, i) =>
        Number(Math.floor(i / width) === 1 && i % width >= 8 && i % width < 12)) };
    const original = track.mask.slice();
    const originalInterior = cockpit.mask.slice();
    for (const instances of [[track, cockpit], [cockpit, track]]) {
        pixels.fill(0);
        drawVisionOverlay(context, { capturedAt: 0, width: 640, height: 640, detections: { segment: {
            task: 'segment', width, height, inferenceMs: 1, classNames: ['track', 'car interior'], instances,
        } } }, displayLabel);
        for (let i = 0; i < width * height; i++) {
            const showTrack = track.mask[i] && displayLabel !== 'car interior';
            const showInterior = cockpit.mask[i] && displayLabel !== 'track';
            const expected = showTrack && showInterior ? instances[0] === track ? [75, 205, 225, 173] : [67, 219, 202, 173]
                : showInterior ? [87, 185, 255, 110] : showTrack ? [55, 239, 172, 110] : [0, 0, 0, 0];
            expect(Array.from(pixels.slice(i * 4, i * 4 + 4))).toEqual(expected);
        }
        expect(track.mask).toEqual(original);
        expect(cockpit.mask).toEqual(originalInterior);
    }
    expect(context.fillText).toHaveBeenCalledTimes(displayLabel ? 2 : 4);
    if (displayLabel !== 'track') expect(context.fillText).toHaveBeenCalledWith('car interior · 60%', 4, 16);
});
