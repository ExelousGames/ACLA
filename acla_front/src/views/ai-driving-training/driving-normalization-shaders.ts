import { DrivingModelConfig, drivingCameraInputs } from './driving-model-config';

/** Per-image LayerNorm. Cache xhat and inverse standard deviation for its exact derivative. */
export function drivingNormalizationShaders(config: DrivingModelConfig): Record<string, string> {
    const constants = `const N = ${drivingCameraInputs(config)}u;
const EPSILON = ${config.network.layerNormEpsilon.toExponential()}f;
var<workgroup> sums: array<f32, 64>;
fn total(value: f32, local: u32) -> f32 {
    sums[local] = value; workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if (local < stride) { sums[local] += sums[local + stride]; }
        workgroupBarrier();
    }
    let result = sums[0]; workgroupBarrier(); return result;
}
`;
    return {
        cameraNormalize: constants + `
@group(0) @binding(0) var<storage, read> inputs: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputs: array<f32>;
@group(0) @binding(3) var<storage, read_write> cache: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) local: u32) {
    var sum = 0.0;
    for (var i = local; i < N; i += 64u) { sum += inputs[i]; }
    let mean = total(sum, local) / f32(N);
    var squares = 0.0;
    for (var i = local; i < N; i += 64u) { squares += (inputs[i] - mean) * (inputs[i] - mean); }
    let inverse = inverseSqrt(total(squares, local) / f32(N) + EPSILON);
    for (var i = local; i < N; i += 64u) {
        let normalized = (inputs[i] - mean) * inverse;
        cache[i] = normalized;
        outputs[i] = weights[i] * normalized + weights[N + i];
    }
    if (local == 0u) { cache[N] = inverse; }
}`,
        cameraNormalizeBackward: constants + `
@group(0) @binding(0) var<storage, read> cache: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> deltas: array<f32>;
@group(0) @binding(3) var<storage, read_write> gradient: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) local: u32) {
    var sum = 0.0; var projected = 0.0;
    for (var i = local; i < N; i += 64u) {
        let delta = deltas[i];
        gradient[i] += delta * cache[i];
        gradient[N + i] += delta;
        sum += delta * weights[i];
        projected += delta * weights[i] * cache[i];
    }
    let mean = total(sum, local) / f32(N);
    let projection = total(projected, local) / f32(N);
    // All upstream components have been consumed before any are overwritten.
    storageBarrier();
    for (var i = local; i < N; i += 64u) {
        deltas[i] = cache[N] * (deltas[i] * weights[i] - mean - cache[i] * projection);
    }
}`,
    };
}
