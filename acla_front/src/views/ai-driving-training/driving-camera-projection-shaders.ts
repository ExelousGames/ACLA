import { DrivingModelConfig, drivingCameraInputs, drivingCameraLayers, drivingCameraPooledInputs } from './driving-model-config';

/** Adaptive average pooling keeps spatial cells separate; a linear projection
 * learns a different coefficient for every (y, x, channel) position. */
export function drivingCameraProjectionShaders(config: DrivingModelConfig): Record<string, string> {
    const layers = drivingCameraLayers(config), last = layers[layers.length - 1];
    const { width, height } = config.network.cameraPooling;
    const pooled = drivingCameraPooledInputs(config), projected = drivingCameraInputs(config);
    const grid = `
const W = ${last.width}u; const H = ${last.height}u; const C = ${last.channels}u;
const PW = ${width}u; const PH = ${height}u;
// Floor start and ceil end cover every pixel, including non-divisible dimensions.
fn bounds(x: u32, y: u32) -> vec4u {
    return vec4u(x * W / PW, y * H / PH,
        ((x + 1u) * W + PW - 1u) / PW, ((y + 1u) * H + PH - 1u) / PH);
}
`;
    const projection = `const INPUTS = ${pooled}u; const OUTPUTS = ${projected}u; const COLS = ${pooled + 1}u;\n`;
    return {
        cameraPool: grid + `
@group(0) @binding(0) var<storage, read> inputs: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputs: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= PW * PH * C) { return; }
    let c = id.x % C; let b = bounds((id.x / C) % PW, id.x / (C * PW));
    var sum = 0.0;
    for (var y = b.y; y < b.w; y++) {
        for (var x = b.x; x < b.z; x++) { sum += inputs[(y * W + x) * C + c]; }
    }
    outputs[id.x] = sum / f32((b.z - b.x) * (b.w - b.y));
}`,
        cameraPoolBackward: grid + `
@group(0) @binding(0) var<storage, read> deltas: array<f32>;
@group(0) @binding(1) var<storage, read_write> gradient: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= W * H * C) { return; }
    let c = id.x % C; let x = (id.x / C) % W; let y = id.x / (C * W);
    var sum = 0.0;
    for (var py = 0u; py < PH; py++) {
        for (var px = 0u; px < PW; px++) {
            let b = bounds(px, py);
            if (x >= b.x && x < b.z && y >= b.y && y < b.w) {
                sum += deltas[(py * PW + px) * C + c] / f32((b.z - b.x) * (b.w - b.y));
            }
        }
    }
    gradient[id.x] = sum;
}`,
        cameraProject: projection + `
@group(0) @binding(0) var<storage, read> inputs: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputs: array<f32>;
var<workgroup> sums: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) local: u32) {
    var sum = 0.0;
    for (var column = local; column < INPUTS; column += 64u) {
        sum += inputs[column] * weights[group.x * COLS + column];
    }
    sums[local] = sum; workgroupBarrier();
    for (var step = 32u; step > 0u; step /= 2u) {
        if (local < step) { sums[local] += sums[local + step]; }
        workgroupBarrier();
    }
    if (local == 0u) { outputs[group.x] = sums[0] + weights[group.x * COLS + INPUTS]; }
}`,
        cameraProjectionBackward: projection + `
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> deltas: array<f32>;
@group(0) @binding(2) var<storage, read_write> gradient: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= INPUTS) { return; }
    var sum = 0.0;
    for (var row = 0u; row < OUTPUTS; row++) { sum += weights[row * COLS + id.x] * deltas[row]; }
    gradient[id.x] = sum;
}`,
        cameraProjectionWeightGradient: projection + `
@group(0) @binding(0) var<storage, read> inputs: array<f32>;
@group(0) @binding(1) var<storage, read> deltas: array<f32>;
@group(0) @binding(2) var<storage, read_write> gradient: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= OUTPUTS * COLS) { return; }
    let column = id.x % COLS;
    var value = 1.0;
    if (column < INPUTS) { value = inputs[column]; }
    gradient[id.x] += deltas[id.x / COLS] * value;
}`,
        cameraProjectionUpdate: projection + `
@group(0) @binding(0) var<storage, read_write> weights: array<f32>;
@group(0) @binding(1) var<storage, read> gradient: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= OUTPUTS * COLS) { return; }
    weights[id.x] = clamp(weights[id.x] + gradient[id.x], -${config.learning.weightLimit.toExponential()}f, ${config.learning.weightLimit.toExponential()}f);
}`,
    };
}
