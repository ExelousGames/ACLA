import { FloatTensor } from './yolo-segmentation';
import { DepthResult, SegmentResult } from './track-vision-types';

export function decodeDepth(output: FloatTensor): DepthResult {
    const dims = output.dims;
    const height = dims[dims.length - 2];
    const width = dims[dims.length - 1];
    if ((dims.length !== 3 && dims.length !== 4) || dims[0] !== 1 || (dims.length === 4 && dims[1] !== 1)
        || height < 1 || width < 1 || output.data.length !== width * height) {
        throw new Error('Depth requires float32 [1, 1, height, width] distances.');
    }
    // Copy before ONNX releases the output tensor.
    return { task: 'depth', width, height, values: output.data.slice() };
}

interface Candidate { index: number; classId: number; confidence: number; box: [number, number, number, number] }
const iou = (a: Candidate, b: Candidate) => {
    const [ax1, ay1, ax2, ay2] = a.box;
    const [bx1, by1, bx2, by2] = b.box;
    const intersection = Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1))
        * Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1));
    return intersection / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection || 1);
};

/** Raw Ultralytics segmentation outputs, retaining separate masks for every detected class. */
export function decodeSegments(predictions: FloatTensor, prototypes: FloatTensor, size: number, threshold: number, classCount: number): SegmentResult {
    const [batch, channels, count] = predictions.dims;
    const [maskBatch, maskChannels, height, width] = prototypes.dims;
    const classes = channels - 4 - maskChannels;
    if (predictions.dims.length !== 3 || prototypes.dims.length !== 4 || batch !== 1 || maskBatch !== 1
        || classes < 1 || classes !== classCount || maskChannels < 1 || width < 1 || height < 1
        || count < 1
        || predictions.data.length !== channels * count || prototypes.data.length !== maskChannels * height * width) {
        throw new Error('Segmentation output does not match the backend labels or raw mask format.');
    }
    if (!Number.isFinite(threshold) || threshold <= 0 || threshold > 1) throw new Error('Invalid confidence threshold.');
    const at = (channel: number, index: number) => predictions.data[channel * count + index];
    const candidates: Candidate[] = [];
    for (let index = 0; index < count; index++) {
        let classId = 0;
        for (let c = 1; c < classes; c++) if (at(4 + c, index) > at(4 + classId, index)) classId = c;
        const confidence = at(4 + classId, index);
        const [x, y, w, h] = [at(0, index), at(1, index), at(2, index), at(3, index)];
        if (![confidence, x, y, w, h].every(Number.isFinite) || confidence < threshold || confidence > 1 || w <= 0 || h <= 0) continue;
        candidates.push({ index, classId, confidence, box: [(x - w / 2) / size, (y - h / 2) / size, (x + w / 2) / size, (y + h / 2) / size] });
    }
    candidates.sort((a, b) => b.confidence - a.confidence);
    const kept: Candidate[] = [];
    for (const candidate of candidates.slice(0, 300)) {
        if (kept.every((other) => other.classId !== candidate.classId || iou(other, candidate) < 0.45)) kept.push(candidate);
        if (kept.length === 30) break;
    }
    const instances: SegmentResult['instances'] = [];
    for (const candidate of kept) {
        const coefficients = Array.from({ length: maskChannels }, (_, c) => at(4 + classes + c, candidate.index));
        if (!coefficients.every(Number.isFinite)) continue;
        const mask = new Uint8Array(width * height);
        for (let y = 0; y < height; y++) {
            if ((y + 0.5) / height < candidate.box[1] || (y + 0.5) / height >= candidate.box[3]) continue;
            for (let x = 0; x < width; x++) {
                if ((x + 0.5) / width < candidate.box[0] || (x + 0.5) / width >= candidate.box[2]) continue;
                const pixel = y * width + x;
                let logit = 0;
                for (let c = 0; c < maskChannels; c++) logit += coefficients[c] * prototypes.data[c * width * height + pixel];
                mask[pixel] = Number(logit > 0);
            }
        }
        instances.push({ classId: candidate.classId, confidence: candidate.confidence, box: candidate.box, mask });
    }
    return { task: 'segment', width, height, instances };
}
