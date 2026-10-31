import type { Tensor } from 'onnxruntime-web';
import type { DepthResult } from './track-vision-types';
import { decodeDepth } from './vision-decoding';
import { VISION_DEPTH_COLORS } from './vision-colors';
import { runWithVisionGpuQueue } from './vision-gpu-queue';
import { geometryBuffers, GPU_IMAGE_GEOMETRY, GpuImageGeometry } from './vision-gpu-input';
import type { DepthReadbackTiming } from './vision-timing';

const DEPTH = `${GPU_IMAGE_GEOMETRY}
@group(0) @binding(2) var<storage, read> disparity: array<f32>;
fn depthAt(pixel: u32) -> f32 {
    let size = vec2u(geometry.size);
    let uv = (vec2f(f32(pixel % size.x), f32(pixel / size.x)) + 0.5) / geometry.size;
    let raw = disparity[pixel];
    if (!inCapture(uv) || !retained(uv) || !(raw >= 0.0 && raw <= 3.402823e38)) { return 0.0; }
    return 1.0 / (1.0 + raw);
}`;
const REDUCTION = `
var<workgroup> nearValues: array<f32, 256>;
var<workgroup> farValues: array<f32, 256>;
fn reduce(local: u32) {
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (local < stride) {
            nearValues[local] = min(nearValues[local], nearValues[local + stride]);
            farValues[local] = max(farValues[local], farValues[local + stride]);
        }
        workgroupBarrier();
    }
}`;
const RANGE = `${DEPTH}
@group(0) @binding(3) var<storage, read_write> chunks: array<vec2f>;
${REDUCTION}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) local: u32,
    @builtin(workgroup_id) group: vec3u) {
    var value = 0.0;
    if (id.x < u32(geometry.size.x * geometry.size.y)) { value = depthAt(id.x); }
    nearValues[local] = select(3.402823e38, value, value > 0.0);
    farValues[local] = value;
    reduce(local);
    if (local == 0u) { chunks[group.x] = vec2f(nearValues[0], farValues[0]); }
}`;
const RANGE_FINAL = `
@group(0) @binding(0) var<storage, read> chunks: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> range: array<vec2f>;
${REDUCTION}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) local: u32) {
    var near = 3.402823e38; var far = 0.0;
    for (var i = local; i < arrayLength(&chunks); i += 256u) {
        near = min(near, chunks[i].x); far = max(far, chunks[i].y);
    }
    nearValues[local] = near; farValues[local] = far;
    reduce(local);
    if (local == 0u) { range[0] = vec2f(nearValues[0], farValues[0]); }
}`;
const colors = VISION_DEPTH_COLORS.map((color) => `vec3f(${[1, 3, 5].map((i) => (parseInt(color.slice(i, i + 2), 16) / 255).toFixed(8)).join(', ')})`);
const COLOR = `${DEPTH}
@group(0) @binding(3) var<storage, read> range: array<vec2f>;
@vertex fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    var vertices = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
    return vec4f(vertices[index], 0, 1);
}
@fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f {
    let value = depthAt(u32(position.y) * u32(geometry.size.x) + u32(position.x));
    if (value <= 0.0 || range[0].y <= 0.0) { return vec4f(0); }
    var t = 0.0;
    if (range[0].y > range[0].x) { t = clamp((value - range[0].x) / (range[0].y - range[0].x), 0.0, 1.0); }
    let scaled = t * ${colors.length - 1}.0;
    let lower = u32(scaled);
    var colors = array<vec3f, ${colors.length}>(${colors.join(', ')});
    return vec4f(mix(colors[lower], colors[min(lower + 1u, ${colors.length - 1}u)], fract(scaled)), 1);
}`;

const pipelines = new WeakMap<GPUDevice, { range: GPUComputePipeline; finish: GPUComputePipeline; color: GPURenderPipeline }>();
function depthPipelines(device: GPUDevice) {
    let cached = pipelines.get(device);
    if (!cached) {
        const compute = (code: string) => device.createComputePipeline({ layout: 'auto',
            compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
        const color = device.createShaderModule({ code: COLOR });
        cached = { range: compute(RANGE), finish: compute(RANGE_FINAL),
            color: device.createRenderPipeline({ layout: 'auto', vertex: { module: color, entryPoint: 'vertex' },
                fragment: { module: color, entryPoint: 'fragment', targets: [{ format: 'rgba8unorm' as const }] },
                primitive: { topology: 'triangle-list' } }) };
        pipelines.set(device, cached);
    }
    return cached;
}

export function gpuDepthShape(output: Tensor) {
    const dims = output.dims, width = dims[dims.length - 1], height = dims[dims.length - 2];
    if (output.type !== 'float32' || output.location !== 'gpu-buffer'
        || (dims.length !== 3 && dims.length !== 4) || dims[0] !== 1 || (dims.length === 4 && dims[1] !== 1)
        || !(width > 0 && height > 0)) throw new Error('Depth requires GPU float32 [1, height, width] or [1, 1, height, width] output.');
    return { width, height };
}

/** One frame owns its ORT output. Rendering never requests the CPU tensor data. */
export class GpuDepth {
    private buffers: ReturnType<typeof geometryBuffers>;
    private range: GPUBuffer;
    private colorGroup: GPUBindGroup;
    private layer?: HTMLCanvasElement;
    private context?: GPUCanvasContext;
    private reading?: Promise<Float32Array>;
    private readingRange?: Promise<{ near: number | null; far: number | null }>;
    private released = false;
    private releasePromise?: Promise<void>;
    private pipeline: ReturnType<typeof depthPipelines>;
    readonly width: number;
    readonly height: number;
    rangeValues?: { near: number | null; far: number | null };
    readbackTiming?: DepthReadbackTiming;

    constructor(private device: GPUDevice, private output: Tensor, private geometry: GpuImageGeometry,
        private onRelease: () => void = () => {}) {
        const shape = gpuDepthShape(output);
        this.width = shape.width; this.height = shape.height;
        this.pipeline = depthPipelines(device);
        this.buffers = geometryBuffers(device, this.width, this.height, geometry);
        this.range = device.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const groups = Math.ceil(this.width * this.height / 256);
        const chunks = device.createBuffer({ size: groups * 8, usage: GPUBufferUsage.STORAGE });
        try {
            const entries = [
                { binding: 0, resource: { buffer: this.buffers.params } }, { binding: 1, resource: { buffer: this.buffers.mask } },
                { binding: 2, resource: { buffer: output.gpuBuffer } },
            ];
            this.colorGroup = device.createBindGroup({ layout: this.pipeline.color.getBindGroupLayout(0),
                entries: [...entries, { binding: 3, resource: { buffer: this.range } }] });
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.pipeline.range);
            pass.setBindGroup(0, device.createBindGroup({ layout: this.pipeline.range.getBindGroupLayout(0),
                entries: [...entries, { binding: 3, resource: { buffer: chunks } }] }));
            pass.dispatchWorkgroups(groups);
            pass.setPipeline(this.pipeline.finish);
            pass.setBindGroup(0, device.createBindGroup({ layout: this.pipeline.finish.getBindGroupLayout(0), entries: [
                { binding: 0, resource: { buffer: chunks } }, { binding: 1, resource: { buffer: this.range } },
            ] }));
            pass.dispatchWorkgroups(1);
            pass.end();
            device.queue.submit([encoder.finish()]);
        } catch (error) {
            this.buffers.params.destroy(); this.buffers.mask.destroy(); this.range.destroy();
            throw error;
        } finally { chunks.destroy(); }
    }

    draw(context: CanvasRenderingContext2D, crop: { x: number; y: number; width: number; height: number }, width: number, height: number) {
        if (this.released) return;
        if (!this.layer) {
            this.layer = document.createElement('canvas');
            this.layer.width = this.width; this.layer.height = this.height;
            this.context = this.layer.getContext('webgpu') ?? undefined;
            if (!this.context) throw new Error('WebGPU depth display is unavailable.');
            this.context.configure({ device: this.device, format: 'rgba8unorm', alphaMode: 'premultiplied' });
        }
        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginRenderPass({ colorAttachments: [{ view: this.context!.getCurrentTexture().createView(),
            clearValue: [0, 0, 0, 0], loadOp: 'clear' as const, storeOp: 'store' as const }] });
        pass.setPipeline(this.pipeline.color); pass.setBindGroup(0, this.colorGroup); pass.draw(3); pass.end();
        this.device.queue.submit([encoder.finish()]);
        context.save();
        context.fillStyle = '#090d13'; context.fillRect(0, 0, width, height); context.imageSmoothingEnabled = false;
        context.drawImage(this.layer, crop.x, crop.y, crop.width, crop.height, 0, 0, width, height);
        context.restore();
    }

    read(): Promise<Float32Array> {
        if (this.reading) return this.reading;
        if (this.released) return Promise.reject(new Error('The depth frame has been released.'));
        const started = performance.now();
        this.reading = runWithVisionGpuQueue(async () => {
            const dequeuedAt = performance.now();
            // Only synchronize when a CPU consumer already needs this frame. Separating
            // pending GPU work prevents it being misreported as tensor download time.
            // This shared queue wait includes unfinished segmentation and depth work.
            await this.device.queue.onSubmittedWorkDone();
            const downloadStarted = performance.now();
            const data = await this.output.getData();
            const conversionStarted = performance.now();
            if (!(data instanceof Float32Array)) throw new Error('Depth requires float32 values.');
            const { values } = decodeDepth({ dims: this.output.dims, data });
            const [left, top, width, height] = this.geometry.rect;
            const region = this.geometry.region;
            for (let i = 0; i < values.length; i++) {
                const u = (i % this.width + 0.5) / this.width, v = (Math.floor(i / this.width) + 0.5) / this.height;
                if (u < left || u >= left + width || v < top || v >= top + height
                    || (region && !region.mask[Math.floor(v * region.height) * region.width + Math.floor(u * region.width)])) values[i] = 0;
            }
            const finished = performance.now();
            this.readbackTiming = { queueMs: dequeuedAt - started, gpuWaitMs: downloadStarted - dequeuedAt,
                downloadMs: conversionStarted - downloadStarted, conversionMs: finished - conversionStarted,
                totalMs: finished - started };
            return values;
        });
        return this.reading;
    }

    /** Only the numeric legend needs these eight bytes; coloring uses the GPU range. */
    readRange() {
        if (this.readingRange) return this.readingRange;
        if (this.released) return Promise.reject(new Error('The depth frame has been released.'));
        this.readingRange = runWithVisionGpuQueue(async () => {
            const staging = this.device.createBuffer({ size: 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
            try {
                const encoder = this.device.createCommandEncoder();
                encoder.copyBufferToBuffer(this.range, 0, staging, 0, 8); this.device.queue.submit([encoder.finish()]);
                await staging.mapAsync(GPUMapMode.READ);
                const values = new Float32Array(staging.getMappedRange());
                const near = values[0], far = values[1];
                this.rangeValues = { near: far > 0 ? near : null, far: far > 0 ? far : null };
                return this.rangeValues;
            } finally { staging.destroy(); }
        });
        return this.readingRange;
    }

    dispose(): Promise<void> {
        if (this.releasePromise) return this.releasePromise;
        this.released = true;
        this.onRelease();
        // Queue behind any pending downloads; ORT cannot dispose a tensor during getData().
        this.releasePromise = runWithVisionGpuQueue(async () => {
            this.context?.unconfigure();
            this.buffers.params.destroy(); this.buffers.mask.destroy(); this.range.destroy(); this.output.dispose();
        });
        return this.releasePromise;
    }
}

export async function readDepthValues(depth: DepthResult): Promise<Float32Array> {
    if (depth.values) return depth.values;
    if (!depth.gpu) throw new Error('Depth values are unavailable.');
    const started = performance.now();
    const values = await depth.gpu.read();
    depth.readbackTiming = depth.gpu.readbackTiming;
    depth.readbackMs ??= depth.readbackTiming?.totalMs ?? performance.now() - started;
    depth.values = values;
    return values;
}
