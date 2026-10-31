import type { Tensor } from 'onnxruntime-web';
import type { SegmentResult } from './track-vision-types';
import type { SegmentReadbackTiming } from './vision-timing';

const MAX_INSTANCES = 30;
const HEADER_BYTES = 16 + MAX_INSTANCES * 32;
const COMMON = `
struct Params { count: u32, channels: u32, classes: u32, maskChannels: u32,
    width: u32, height: u32, endToEnd: u32, size: f32, threshold: f32 }
struct Candidate { box: vec4f, confidence: f32, classId: u32, index: u32, validMask: u32 }
struct Selected { count: u32, error: atomic<u32>, padding: vec2u, items: array<Candidate, 30> }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> predictions: array<f32>;
fn at(channel: u32, index: u32) -> f32 {
    if (params.endToEnd != 0u) { return predictions[index * params.channels + channel]; }
    return predictions[channel * params.count + index];
}
fn finite(value: f32) -> bool { return abs(value) <= 3.402823e38; }
`;
const CANDIDATES = `${COMMON}
@group(0) @binding(2) var<storage, read_write> candidates: array<Candidate>;
@group(0) @binding(3) var<storage, read_write> selected: Selected;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    if (index >= params.count) { return; }
    var classId = 0u;
    if (params.endToEnd == 0u) {
        for (var c = 1u; c < params.classes; c++) {
            if (at(4u + c, index) > at(4u + classId, index)) { classId = c; }
        }
    }
    var confidence = at(4u + classId, index);
    let xy = vec2f(at(0u, index), at(1u, index));
    let wh = vec2f(at(2u, index), at(3u, index));
    var box = vec4f(xy - wh / 2.0, xy + wh / 2.0) / params.size;
    if (params.endToEnd != 0u) { box = vec4f(xy, wh) / params.size; }
    if (!finite(confidence) || confidence < params.threshold || confidence > 1.0
        || !all(abs(box) <= vec4f(3.402823e38)) || any(box.zw <= box.xy)) { return; }
    if (params.endToEnd != 0u) {
        let label = at(5u, index);
        if (!finite(label) || label < 0.0 || label >= f32(params.classes) || label != floor(label)) {
            atomicStore(&selected.error, 1u); return;
        }
        classId = u32(label);
    }
    var validMask = 1u;
    let offset = select(4u + params.classes, 6u, params.endToEnd != 0u);
    for (var c = 0u; c < params.maskChannels; c++) {
        if (!finite(at(offset + c, index))) { validMask = 0u; }
    }
    candidates[index] = Candidate(box, confidence, classId, index, validMask);
}`;
const SELECT = `
struct Candidate { box: vec4f, confidence: f32, classId: u32, index: u32, validMask: u32 }
struct Selected { count: u32, error: u32, padding: vec2u, items: array<Candidate, 30> }
@group(0) @binding(0) var<storage, read> candidates: array<Candidate>;
@group(0) @binding(1) var<storage, read_write> selected: Selected;
@group(0) @binding(2) var<uniform> endToEnd: u32;
var<workgroup> scores: array<f32, 256>;
var<workgroup> indices: array<u32, 256>;
var<workgroup> previousScore: f32;
var<workgroup> previousIndex: u32;
var<workgroup> count: u32;
fn overlaps(a: vec4f, b: vec4f) -> bool {
    let intersectionSize = max(vec2f(0), min(a.zw, b.zw) - max(a.xy, b.xy));
    let intersection = intersectionSize.x * intersectionSize.y;
    let areaA = (a.z - a.x) * (a.w - a.y);
    let areaB = (b.z - b.x) * (b.w - b.y);
    let unionArea = areaA + areaB - intersection;
    return intersection / select(1.0, unionArea, unionArea != 0.0) >= 0.45;
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) local: u32) {
    if (local == 0u) { previousScore = 2.0; previousIndex = 0u; count = 0u; }
    workgroupBarrier();
    for (var rank = 0u; rank < 300u; rank++) {
        var score = 0.0; var index = 0xffffffffu;
        for (var i = local; i < arrayLength(&candidates); i += 256u) {
            let candidate = candidates[i];
            let after = candidate.confidence < previousScore || (candidate.confidence == previousScore && i > previousIndex);
            if (after && (candidate.confidence > score || (candidate.confidence == score && i < index))) {
                score = candidate.confidence; index = i;
            }
        }
        scores[local] = score; indices[local] = index;
        workgroupBarrier();
        for (var stride = 128u; stride > 0u; stride /= 2u) {
            if (local < stride && (scores[local + stride] > scores[local]
                || (scores[local + stride] == scores[local] && indices[local + stride] < indices[local]))) {
                scores[local] = scores[local + stride]; indices[local] = indices[local + stride];
            }
            workgroupBarrier();
        }
        if (workgroupUniformLoad(&scores[0]) <= 0.0) { break; }
        if (local == 0u) {
            let candidate = candidates[indices[0]];
            var keep = true;
            if (endToEnd == 0u) {
                for (var k = 0u; k < count; k++) {
                    let other = selected.items[k];
                    if (other.classId == candidate.classId && overlaps(other.box, candidate.box)) { keep = false; }
                }
            }
            if (keep) { selected.items[count] = candidate; count++; }
            previousScore = scores[0]; previousIndex = indices[0];
        }
        workgroupBarrier();
        if (workgroupUniformLoad(&count) == 30u) { break; }
    }
    if (local == 0u) { selected.count = count; }
}`;
const MASKS = `${COMMON}
@group(0) @binding(2) var<storage, read> prototypes: array<f32>;
@group(0) @binding(3) var<storage, read_write> selected: Selected;
@group(0) @binding(4) var<storage, read_write> masks: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let pixels = params.width * params.height;
    let words = (pixels + 3u) / 4u;
    let instance = id.y;
    if (id.x >= words || instance >= selected.count) { return; }
    let candidate = selected.items[instance];
    if (candidate.validMask == 0u) { return; }
    let offset = select(4u + params.classes, 6u, params.endToEnd != 0u);
    var packed = 0u;
    for (var byte = 0u; byte < 4u; byte++) {
        let pixel = id.x * 4u + byte;
        let uv = (vec2f(f32(pixel % params.width), f32(pixel / params.width)) + 0.5) / vec2f(f32(params.width), f32(params.height));
        if (pixel >= pixels || any(uv < candidate.box.xy) || any(uv >= candidate.box.zw)) { continue; }
        var logit = 0.0;
        for (var c = 0u; c < params.maskChannels; c++) {
            logit += at(offset + c, candidate.index) * prototypes[c * pixels + pixel];
        }
        if (logit > 0.0) { packed |= 1u << (byte * 8u); }
    }
    masks[instance * words + id.x] = packed;
}`;

const pipelines = new WeakMap<GPUDevice, GPUComputePipeline[]>();

export function gpuSegmentShape(predictions: Tensor, prototypes: Tensor, classCount: number) {
    const [maskBatch, maskChannels, height, width] = prototypes.dims;
    const [batch, rows, columns] = predictions.dims;
    const endToEnd = columns === 6 + maskChannels && rows !== 4 + classCount + maskChannels;
    const channels = endToEnd ? columns : rows, count = endToEnd ? rows : columns;
    if (predictions.type !== 'float32' || prototypes.type !== 'float32'
        || predictions.location !== 'gpu-buffer' || prototypes.location !== 'gpu-buffer'
        || predictions.dims.length !== 3 || prototypes.dims.length !== 4 || batch !== 1 || maskBatch !== 1
        || classCount < 1 || (!endToEnd && channels !== 4 + classCount + maskChannels)
        || maskChannels < 1 || width < 1 || height < 1 || count < 1) {
        throw new Error('Segmentation output does not match the backend labels or GPU float32 mask format.');
    }
    return { width, height, count, channels, maskChannels, endToEnd };
}

/** GPU selection, class-aware NMS and prototype masks. Only final masks/metadata are downloaded. */
export class GpuSegments {
    readonly width: number;
    readonly height: number;
    readbackTiming?: SegmentReadbackTiming;
    private selected: GPUBuffer;
    private masks: GPUBuffer;
    private reading?: Promise<SegmentResult['instances']>;
    private released = false;
    private releasePromise?: Promise<void>;

    constructor(private device: GPUDevice, predictions: Tensor, prototypes: Tensor, size: number, threshold: number,
        classCount: number, private onRelease = () => {}) {
        const shape = gpuSegmentShape(predictions, prototypes, classCount);
        if (!Number.isFinite(threshold) || threshold <= 0 || threshold > 1) throw new Error('Invalid confidence threshold.');
        const { width, height, count, channels, maskChannels, endToEnd } = shape;
        this.width = width; this.height = height;
        let cached = pipelines.get(device);
        if (!cached) {
            cached = [CANDIDATES, SELECT, MASKS].map((code) => device.createComputePipeline({ layout: 'auto',
                compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } }));
            pipelines.set(device, cached);
        }
        const storage = (bytes: number) => device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        this.selected = storage(HEADER_BYTES);
        this.masks = storage(Math.ceil(width * height / 4) * 4 * MAX_INSTANCES);
        const candidates = storage(count * 32);
        const params = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const data = new ArrayBuffer(48);
        new Uint32Array(data).set([count, channels, classCount, maskChannels, width, height, Number(endToEnd)]);
        new Float32Array(data).set([size, threshold], 7);
        // Preserve JS threshold comparisons against float32 predictions when rounding down.
        if (new Float32Array(data)[8] < threshold) new Uint32Array(data)[8]++;
        device.queue.writeBuffer(params, 0, data);
        try {
            const encoder = device.createCommandEncoder();
            const dispatch = (pipeline: GPUComputePipeline, buffers: Array<{ buffer: GPUBuffer; offset?: number; size?: number }>, x: number, y = 1) => {
                const pass = encoder.beginComputePass();
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
                    entries: buffers.map((resource, binding) => ({ binding, resource })) }));
                pass.dispatchWorkgroups(x, y); pass.end();
            };
            // A separate uniform avoids storage-buffer offset alignment requirements.
            const mode = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            try {
                device.queue.writeBuffer(mode, 0, new Uint32Array([Number(endToEnd)]));
                dispatch(cached[0], [params, predictions.gpuBuffer, candidates, this.selected].map((buffer) => ({ buffer })), Math.ceil(count / 256));
                dispatch(cached[1], [candidates, this.selected, mode].map((buffer) => ({ buffer })), 1);
                dispatch(cached[2], [params, predictions.gpuBuffer, prototypes.gpuBuffer, this.selected, this.masks].map((buffer) => ({ buffer })),
                    Math.ceil(Math.ceil(width * height / 4) / 64), MAX_INSTANCES);
                device.queue.submit([encoder.finish()]);
            } finally { mode.destroy(); }
        } catch (error) {
            this.selected.destroy(); this.masks.destroy(); throw error;
        } finally { candidates.destroy(); params.destroy(); }
    }

    read(): Promise<SegmentResult['instances']> {
        if (this.reading) return this.reading;
        if (this.released) return Promise.reject(new Error('The segmentation frame has been released.'));
        const started = performance.now();
        // Native mapping stays outside ORT's queue and needs no explicit device-wide completion wait.
        this.reading = (async () => {
            const maskBytes = Math.ceil(this.width * this.height / 4) * 4;
            const staging = this.device.createBuffer({ size: HEADER_BYTES + maskBytes * MAX_INSTANCES,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
            try {
                const encoder = this.device.createCommandEncoder();
                encoder.copyBufferToBuffer(this.selected, 0, staging, 0, HEADER_BYTES);
                encoder.copyBufferToBuffer(this.masks, 0, staging, HEADER_BYTES, maskBytes * MAX_INSTANCES);
                this.device.queue.submit([encoder.finish()]);
                const submittedAt = performance.now();
                await staging.mapAsync(GPUMapMode.READ);
                const mappedAt = performance.now();
                const data = staging.getMappedRange(), ints = new Uint32Array(data), floats = new Float32Array(data);
                if (ints[1]) throw new Error('Segmentation class ID does not match the backend labels.');
                const instances: SegmentResult['instances'] = [];
                for (let i = 0; i < ints[0]; i++) {
                    const offset = 4 + i * 8;
                    if (!ints[offset + 7]) continue;
                    instances.push({ classId: ints[offset + 5], confidence: floats[offset + 4],
                        box: [floats[offset], floats[offset + 1], floats[offset + 2], floats[offset + 3]],
                        mask: new Uint8Array(data, HEADER_BYTES + i * maskBytes, this.width * this.height).slice() });
                }
                const finished = performance.now();
                this.readbackTiming = { submissionMs: submittedAt - started, mapMs: mappedAt - submittedAt,
                    assemblyMs: finished - mappedAt, totalMs: finished - started };
                return instances;
            } finally { staging.destroy(); }
        })();
        return this.reading;
    }

    dispose(): Promise<void> {
        if (this.releasePromise) return this.releasePromise;
        this.released = true;
        this.onRelease();
        this.releasePromise = (async () => {
            await this.reading?.catch(() => undefined);
            this.selected.destroy(); this.masks.destroy();
        })();
        return this.releasePromise;
    }
}

export async function readSegmentInstances(segment: SegmentResult) {
    if (segment.gpu && segment.readbackMs === undefined) {
        const started = performance.now();
        segment.instances = await segment.gpu.read();
        segment.readbackTiming = segment.gpu.readbackTiming;
        segment.readbackMs ??= segment.readbackTiming?.totalMs ?? performance.now() - started;
    }
    return segment.instances;
}
