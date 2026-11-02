import { DrivingModelConfig, drivingCameraInputs, drivingCameraLayers, drivingPolicyInputs } from './driving-model-config';
import { drivingCameraProjectionShaders } from './driving-camera-projection-shaders';

/** Shared filters, channel-last activations and explicit convolution backpropagation. */
export function drivingCameraShaders(config: DrivingModelConfig): Record<string, string> {
    const shaders: Record<string, string> = {
        ...drivingCameraProjectionShaders(config),
        cameraGradient: `
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> deltas: array<f32>;
@group(0) @binding(2) var<storage, read> offset: array<u32>;
@group(0) @binding(3) var<storage, read_write> gradient: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= ${drivingCameraInputs(config)}u) { return; }
    let t = offset[0] / ${drivingPolicyInputs(config)}u;
    let column = offset[0] % ${drivingPolicyInputs(config)}u + id.x;
    var sum = 0.0;
    for (var row = 0u; row < ${4 * config.network.hiddenUnits}u; row++) {
        sum += weights[row * ${drivingPolicyInputs(config) + config.network.hiddenUnits + 1}u + column]
            * deltas[t * ${4 * config.network.hiddenUnits}u + row];
    }
    gradient[id.x] = sum;
}`,
    };
    drivingCameraLayers(config).forEach((layer, l) => {
        const { width, height, channels, inputWidth, inputHeight, inputChannels, kernelSize, stride, padding, columns } = layer;
        const count = width * height * channels;
        const constants = `
const W = ${width}u; const H = ${height}u; const C = ${channels}u;
const IW = ${inputWidth}u; const IH = ${inputHeight}u; const IC = ${inputChannels}u;
const K = ${kernelSize}u; const S = ${stride}u; const P = ${padding}i; const COLS = ${columns}u;
`;
        const input = `
@group(0) @binding(0) var<storage, read> inputs: array<${l ? 'f32' : 'u32'}>;
fn readInput(x: i32, y: i32, c: u32) -> f32 {
    if (x < 0 || y < 0 || x >= i32(IW) || y >= i32(IH)) { return 0.0; }
    let index = (u32(y) * IW + u32(x)) * IC + c;
    return ${l ? 'inputs[index]' : 'f32((inputs[index / 4u] >> ((index % 4u) * 8u)) & 255u) / 255.0'};
}
`;
        shaders[`cameraForward${l}`] = constants + input + `
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputs: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= ${count}u) { return; }
    let c = id.x % C; let x = (id.x / C) % W; let y = id.x / (C * W);
    var sum = weights[c * COLS + COLS - 1u];
    for (var ky = 0u; ky < K; ky++) {
        for (var k = 0u; k < K * IC; k++) {
            sum += readInput(i32(x * S + k / IC) - P, i32(y * S + ky) - P, k % IC)
                * weights[c * COLS + ky * K * IC + k];
        }
    }
    outputs[id.x] = select(sum * 0.01, sum, sum >= 0.0);
}`;
        shaders[`cameraDelta${l}`] = `
@group(0) @binding(0) var<storage, read> outputs: array<f32>;
@group(0) @binding(1) var<storage, read_write> gradient: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= ${count}u) { return; }
    gradient[id.x] *= select(0.01, 1.0, outputs[id.x] >= 0.0);
}`;
        if (l) shaders[`cameraBackward${l}`] = constants + `
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> deltas: array<f32>;
@group(0) @binding(2) var<storage, read_write> gradient: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= ${inputWidth * inputHeight * inputChannels}u) { return; }
    let c = id.x % IC; let x = (id.x / IC) % IW; let y = id.x / (IC * IW);
    var sum = 0.0;
    // Each input receives contributions from every overlapping output window.
    for (var ky = 0u; ky < K; ky++) {
        let oy = i32(y) + P - i32(ky);
        if (oy < 0 || oy % i32(S) != 0 || oy / i32(S) >= i32(H)) { continue; }
        for (var kx = 0u; kx < K; kx++) {
            let ox = i32(x) + P - i32(kx);
            if (ox < 0 || ox % i32(S) != 0 || ox / i32(S) >= i32(W)) { continue; }
            let column = (ky * K + kx) * IC + c;
            let pixel = (u32(oy) / S * W + u32(ox) / S) * C;
            for (var out = 0u; out < C; out++) { sum += weights[out * COLS + column] * deltas[pixel + out]; }
        }
    }
    gradient[id.x] = sum;
}`;
        shaders[`cameraWeightGradient${l}`] = constants + input + `
@group(0) @binding(1) var<storage, read> deltas: array<f32>;
@group(0) @binding(2) var<storage, read_write> gradient: array<f32>;
var<workgroup> sums: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) local: u32) {
    let channel = group.x / COLS; let column = group.x % COLS;
    let ky = column / (K * IC); let k = column % (K * IC);
    var sum = 0.0;
    for (var p = local; p < ${width * height}u; p += 64u) {
        var value = 1.0;
        if (column < COLS - 1u) {
            value = readInput(i32((p % W) * S + k / IC) - P, i32((p / W) * S + ky) - P, k % IC);
        }
        sum += deltas[p * C + channel] * value;
    }
    sums[local] = sum; workgroupBarrier();
    for (var step = 32u; step > 0u; step /= 2u) {
        if (local < step) { sums[local] += sums[local + step]; }
        workgroupBarrier();
    }
    if (local == 0u) { gradient[group.x] += sums[0]; }
}`;
        shaders[`cameraUpdate${l}`] = `
@group(0) @binding(0) var<storage, read_write> weights: array<f32>;
@group(0) @binding(1) var<storage, read> gradient: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= ${channels * columns}u) { return; }
    weights[id.x] = clamp(weights[id.x] + gradient[id.x], -${config.learning.weightLimit.toExponential()}f, ${config.learning.weightLimit.toExponential()}f);
}`;
    });
    return shaders;
}
