export interface FloatTensor { dims: readonly number[]; data: Float32Array }
export interface BoundaryPoint { x: number; y: number }
export interface TrackBoundaryDetection {
    capturedAt: number;
    width: number;
    height: number;
    inferenceMs: number;
    confidence: number;
    /** Normalized image coordinates. Gaps are kept as separate segments. */
    segments: Array<{ left: BoundaryPoint[]; right: BoundaryPoint[] }>;
}

export function letterbox(width: number, height: number, size: number) {
    const scale = Math.min(size / width, size / height);
    const resizedWidth = Math.round(width * scale);
    const resizedHeight = Math.round(height * scale);
    return {
        resizedWidth, resizedHeight,
        padX: Math.floor((size - resizedWidth) / 2),
        padY: Math.floor((size - resizedHeight) / 2),
    };
}

export function rgbaToChw(rgba: Uint8ClampedArray): Float32Array {
    const pixels = rgba.length / 4;
    const data = new Float32Array(pixels * 3);
    for (let i = 0; i < pixels; i++) {
        data[i] = rgba[i * 4] / 255;
        data[pixels + i] = rgba[i * 4 + 1] / 255;
        data[pixels * 2 + i] = rgba[i * 4 + 2] / 255;
    }
    return data;
}

interface Candidate { index: number; score: number; x1: number; y1: number; x2: number; y2: number }
function intersectionOverUnion(a: Candidate, b: Candidate) {
    const intersection = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1))
        * Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
    return intersection / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - intersection || 1);
}

/** Decode raw YOLOv8/YOLO11 segmentation exports (no embedded NMS). */
export function decodeTrackMask(
    predictions: FloatTensor, prototypes: FloatTensor, size: number,
    classId: number, threshold: number,
): { mask: Uint8Array; width: number; height: number; confidence: number } {
    const [batch, channels, count] = predictions.dims;
    const [maskBatch, maskChannels, height, width] = prototypes.dims;
    const classes = channels - 4 - maskChannels;
    const expectedCount = (size / 8) ** 2 + (size / 16) ** 2 + (size / 32) ** 2;
    if (predictions.dims.length !== 3 || prototypes.dims.length !== 4 || batch !== 1 || maskBatch !== 1
        || classes < 1 || count !== expectedCount || width < 1 || height < 1 || maskChannels < 1
        || predictions.data.length !== channels * count || prototypes.data.length !== maskChannels * width * height) {
        throw new Error('Unsupported model outputs. Use a raw YOLOv8 or YOLO11 segmentation ONNX export without NMS.');
    }
    if (!Number.isInteger(classId) || classId < 0 || classId >= classes) {
        throw new Error(`Track class must be between 0 and ${classes - 1}.`);
    }
    if (!Number.isFinite(threshold) || threshold <= 0 || threshold > 1) throw new Error('Invalid confidence threshold.');
    const at = (channel: number, index: number) => predictions.data[channel * count + index];
    const candidates: Candidate[] = [];
    for (let i = 0; i < count; i++) {
        const score = at(4 + classId, i);
        if (!Number.isFinite(score) || score < threshold || score > 1) continue;
        // A box belongs to the most likely class, not every class above threshold.
        let winningClass = 0;
        for (let c = 1; c < classes; c++) if (at(4 + c, i) > at(4 + winningClass, i)) winningClass = c;
        if (winningClass !== classId) continue;
        const [x, y, w, h] = [at(0, i), at(1, i), at(2, i), at(3, i)];
        if (![x, y, w, h].every(Number.isFinite) || w <= 0 || h <= 0) continue;
        candidates.push({ index: i, score, x1: x - w / 2, y1: y - h / 2, x2: x + w / 2, y2: y + h / 2 });
    }
    candidates.sort((a, b) => b.score - a.score);
    const kept: Candidate[] = [];
    for (const candidate of candidates.slice(0, 300)) {
        if (kept.every((other) => intersectionOverUnion(candidate, other) < 0.45)) kept.push(candidate);
        if (kept.length === 10) break;
    }
    const mask = new Uint8Array(width * height);
    let confidence = 0;
    for (const candidate of kept) {
        const coefficients = Array.from({ length: maskChannels }, (_, c) => at(4 + classes + c, candidate.index));
        if (!coefficients.every(Number.isFinite)) continue;
        let hasPixels = false;
        for (let y = 0; y < height; y++) {
            if ((y + 0.5) * size / height < candidate.y1 || (y + 0.5) * size / height >= candidate.y2) continue;
            for (let x = 0; x < width; x++) {
                if ((x + 0.5) * size / width < candidate.x1 || (x + 0.5) * size / width >= candidate.x2) continue;
                const pixel = y * width + x;
                let logit = 0;
                for (let c = 0; c < maskChannels; c++) logit += coefficients[c] * prototypes.data[c * width * height + pixel];
                // sigmoid(logit) > 0.5 is equivalent to logit > 0.
                if (logit > 0) { mask[pixel] = 1; hasPixels = true; }
            }
        }
        if (hasPixels) confidence = Math.max(confidence, candidate.score);
    }
    return { mask, width, height, confidence };
}

export function traceTrackBoundaries(
    mask: Uint8Array, maskWidth: number, maskHeight: number,
    frameWidth: number, frameHeight: number, size: number,
): TrackBoundaryDetection['segments'] {
    const { padX, padY, resizedWidth, resizedHeight } = letterbox(frameWidth, frameHeight, size);
    const segments: TrackBoundaryDetection['segments'] = [];
    let segment: TrackBoundaryDetection['segments'][number] | undefined;
    for (let y = 0; y < maskHeight; y++) {
        const imageY = ((y + 0.5) * size / maskHeight - padY) / resizedHeight;
        if (imageY < 0 || imageY > 1) continue;
        let bestStart = -1;
        let bestEnd = -1;
        let start = -1;
        for (let x = 0; x <= maskWidth; x++) {
            const imageX = ((x + 0.5) * size / maskWidth - padX) / resizedWidth;
            const active = x < maskWidth && imageX >= 0 && imageX <= 1 && mask[y * maskWidth + x] !== 0;
            if (active && start < 0) start = x;
            if (!active && start >= 0) {
                if (x - start > bestEnd - bestStart) { bestStart = start; bestEnd = x; }
                start = -1;
            }
        }
        if (bestEnd - bestStart < 2) { segment = undefined; continue; }
        const left = Math.max(0, (bestStart * size / maskWidth - padX) / resizedWidth);
        const right = Math.min(1, (bestEnd * size / maskWidth - padX) / resizedWidth);
        const previous = segment?.left[segment.left.length - 1];
        const previousRight = segment?.right[segment.right.length - 1];
        if (previous && previousRight && (Math.abs(previous.x - left) > 0.15 || Math.abs(previousRight.x - right) > 0.15)) segment = undefined;
        if (!segment) { segment = { left: [], right: [] }; segments.push(segment); }
        segment.left.push({ x: left, y: imageY });
        segment.right.push({ x: right, y: imageY });
    }
    return segments.filter(({ left }) => left.length >= 2);
}
