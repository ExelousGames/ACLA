/// <reference types="@webgpu/types" />
import type { Tensor } from 'onnxruntime-web';
import { VISION_INPUT_SIZE } from './vision-config';
import type { MaskRegion } from './world-mask';
import { letterbox } from './yolo-segmentation';

export interface GpuImageGeometry {
    rect: [number, number, number, number];
    region?: MaskRegion;
}

export function imageGeometry(frame: HTMLCanvasElement, size: number, depth: boolean, region?: MaskRegion): GpuImageGeometry {
    const grid = depth ? VISION_INPUT_SIZE : size;
    const box = letterbox(frame.width, frame.height, grid);
    return { rect: [box.padX / grid, box.padY / grid, box.resizedWidth / grid, box.resizedHeight / grid],
        region: depth && region ? { ...region, mask: region.mask.slice() } : undefined };
}

// Shared by preprocessing and depth rendering: masks are sampled at pixel centers
// in the same normalized letterbox coordinates, without CPU mask resizing.
export const GPU_IMAGE_GEOMETRY = `
struct Geometry { size: vec2f, maskSize: vec2f, rect: vec4f, normalize: vec4f }
@group(0) @binding(0) var<uniform> geometry: Geometry;
@group(0) @binding(1) var<storage, read> mask: array<u32>;
fn retained(uv: vec2f) -> bool {
    if (geometry.maskSize.x == 0.0) { return true; }
    let pixel = min(vec2u(uv * geometry.maskSize), vec2u(geometry.maskSize) - vec2u(1));
    return mask[pixel.y * u32(geometry.maskSize.x) + pixel.x] != 0u;
}
fn inCapture(uv: vec2f) -> bool {
    return all(uv >= geometry.rect.xy) && all(uv < geometry.rect.xy + geometry.rect.zw);
}`;

export function geometryBuffers(device: GPUDevice, width: number, height: number, geometry: GpuImageGeometry, normalize = false) {
    const params = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const mask = device.createBuffer({ size: Math.max(4, (geometry.region?.mask.length ?? 0) * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(params, 0, new Float32Array([width, height, geometry.region?.width ?? 0,
        geometry.region?.height ?? 0, ...geometry.rect, Number(normalize), 0, 0, 0]));
    if (geometry.region) device.queue.writeBuffer(mask, 0, Uint32Array.from(geometry.region.mask));
    return { params, mask };
}

const PREPROCESS = `${GPU_IMAGE_GEOMETRY}
@group(0) @binding(2) var source: texture_2d<f32>;
@group(0) @binding(3) var linearSampler: sampler;
@group(0) @binding(4) var<storage, read_write> input: array<f32>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = vec2u(geometry.size);
    if (any(id.xy >= size)) { return; }
    let uv = (vec2f(id.xy) + 0.5) / geometry.size;
    var rgb = vec3f(114.0 / 255.0);
    if (inCapture(uv) && retained(uv)) {
        rgb = textureSampleLevel(source, linearSampler, (uv - geometry.rect.xy) / geometry.rect.zw, 0.0).rgb;
    }
    if (geometry.normalize.x != 0.0) {
        rgb = (rgb - vec3f(0.485, 0.456, 0.406)) / vec3f(0.229, 0.224, 0.225);
    }
    let pixel = id.y * size.x + id.x;
    let pixels = size.x * size.y;
    input[pixel] = rgb.r; input[pixels + pixel] = rgb.g; input[2u * pixels + pixel] = rgb.b;
}`;

/** Reusable GPU input storage. The ONNX tensor borrows this buffer. */
export class VisionGpuInput {
    readonly tensor: Tensor;
    private buffer: GPUBuffer;
    private pipeline: GPUComputePipeline;
    private sampler: GPUSampler;
    private texture?: GPUTexture;

    constructor(private device: GPUDevice, runtime: typeof import('onnxruntime-web'), private size: number) {
        this.pipeline = device.createComputePipeline({ layout: 'auto',
            compute: { module: device.createShaderModule({ code: PREPROCESS }), entryPoint: 'main' } });
        this.sampler = device.createSampler({ minFilter: 'linear', magFilter: 'linear' });
        this.buffer = device.createBuffer({ size: Math.ceil(3 * size * size * 4 / 16) * 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        this.tensor = runtime.Tensor.fromGpuBuffer(this.buffer, { dataType: 'float32', dims: [1, 3, size, size] });
    }

    prepare(frame: HTMLCanvasElement, geometry: GpuImageGeometry, depth: boolean) {
        if (!this.texture || this.texture.width !== frame.width || this.texture.height !== frame.height) {
            this.texture?.destroy();
            this.texture = this.device.createTexture({ size: [frame.width, frame.height], format: 'rgba8unorm',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
        }
        this.device.queue.copyExternalImageToTexture({ source: frame }, { texture: this.texture }, [frame.width, frame.height]);
        const { params, mask } = geometryBuffers(this.device, this.size, this.size, geometry, depth);
        try {
            const encoder = this.device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.pipeline);
            pass.setBindGroup(0, this.device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [
                { binding: 0, resource: { buffer: params } }, { binding: 1, resource: { buffer: mask } },
                { binding: 2, resource: this.texture.createView() }, { binding: 3, resource: this.sampler },
                { binding: 4, resource: { buffer: this.buffer } },
            ] }));
            pass.dispatchWorkgroups(Math.ceil(this.size / 8), Math.ceil(this.size / 8));
            pass.end();
            this.device.queue.submit([encoder.finish()]);
        } finally { params.destroy(); mask.destroy(); }
        return this.tensor;
    }

    dispose() {
        this.texture?.destroy();
        this.buffer.destroy();
    }
}
