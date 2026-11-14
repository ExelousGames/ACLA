import { DrivingModelConfig, drivingPolicyInputs } from './driving-model-config';

export function drivingLstmShaders(config: DrivingModelConfig): Record<string, string> {
    const constants = `
const INPUTS = ${drivingPolicyInputs(config)}u;
const H = ${config.network.hiddenUnits}u;
const COLS = INPUTS + H + 1u;
const OUTPUTS = ${config.actions.planTimesSeconds.length * config.actions.outputs.length + 1}u;
const LIMIT = ${Number(config.learning.weightLimit).toExponential()}f;

// GPU tanh may evaluate sinh/cosh and overflow for large finite inputs.
// At |x| >= 10, tanh rounds to +/-1 in f32, so 1 - tanh(x)^2 is already zero.
fn stableTanh(value: f32) -> f32 {
    return tanh(clamp(value, -10.0, 10.0));
}
fn stableSigmoid(value: f32) -> f32 {
    let exponential = exp(-abs(value));
    return select(exponential / (1.0 + exponential), 1.0 / (1.0 + exponential), value >= 0.0);
}
`;
    const shaders = {
        forward: `
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> features: array<f32>;
@group(0) @binding(2) var<storage, read_write> states: array<f32>;
@group(0) @binding(3) var<storage, read> time: array<u32>;
var<workgroup> sums: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) local: u32) {
    let previous = time[0] * 6u * H; let current = previous + 6u * H;
    var gates: array<f32, 4>;
    for (var gate = 0u; gate < 4u; gate++) {
        let row = (gate * H + group.x) * COLS;
        var sum = 0.0;
        for (var j = local; j < INPUTS + H; j += 64u) {
            var value = 0.0;
            if (j < INPUTS) { value = features[time[0] * INPUTS + j]; }
            else { value = states[previous + j - INPUTS]; }
            sum += weights[row + j] * value;
        }
        sums[local] = sum; workgroupBarrier();
        for (var stride = 32u; stride > 0u; stride /= 2u) {
            if (local < stride) { sums[local] += sums[local + stride]; }
            workgroupBarrier();
        }
        if (local == 0u) {
            let value = sums[0] + weights[row + COLS - 1u];
            if (gate == 2u) { gates[gate] = stableTanh(value); }
            else { gates[gate] = stableSigmoid(value); }
            states[current + (gate + 2u) * H + group.x] = gates[gate];
        }
        workgroupBarrier();
    }
    if (local == 0u) {
        let cell = gates[1] * states[previous + H + group.x] + gates[0] * gates[2];
        states[current + H + group.x] = cell;
        states[current + group.x] = gates[3] * stableTanh(cell);
    }
}`,
        output: `
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> states: array<f32>;
@group(0) @binding(2) var<storage, read> count: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputs: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= OUTPUTS) { return; }
    var sum = weights[id.x * (H + 1u) + H];
    for (var i = 0u; i < H; i++) { sum += weights[id.x * (H + 1u) + i] * states[count[0] * 6u * H + i]; }
    outputs[id.x] = sum;
}`,
        hiddenGradient: `
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> gradient: array<f32>;
@group(0) @binding(2) var<storage, read_write> upstream: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= H) { return; }
    var sum = 0.0;
    for (var i = 0u; i < OUTPUTS; i++) { sum += gradient[i] * weights[i * (H + 1u) + id.x]; }
    upstream[id.x] = sum;
}`,
        gateGradient: `
@group(0) @binding(0) var<storage, read> states: array<f32>;
@group(0) @binding(1) var<storage, read_write> upstream: array<f32>;
@group(0) @binding(2) var<storage, read_write> deltas: array<f32>;
@group(0) @binding(3) var<storage, read> time: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= H) { return; }
    let previous = time[0] * 6u * H + id.x; let current = previous + 6u * H;
    let i = states[current + 2u * H]; let f = states[current + 3u * H];
    let g = states[current + 4u * H]; let o = states[current + 5u * H];
    let cell = stableTanh(states[current + H]); let dh = upstream[id.x];
    let dc = upstream[H + id.x] + dh * o * (1.0 - cell * cell);
    let offset = time[0] * 4u * H + id.x;
    deltas[offset] = dc * g * i * (1.0 - i);
    deltas[offset + H] = dc * states[previous + H] * f * (1.0 - f);
    deltas[offset + 2u * H] = dc * i * (1.0 - g * g);
    deltas[offset + 3u * H] = dh * cell * o * (1.0 - o);
    upstream[H + id.x] = dc * f;
}`,
        recurrentGradient: `
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> deltas: array<f32>;
@group(0) @binding(2) var<storage, read_write> upstream: array<f32>;
@group(0) @binding(3) var<storage, read> time: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= H) { return; }
    var sum = 0.0;
    for (var row = 0u; row < 4u * H; row++) {
        sum += weights[row * COLS + INPUTS + id.x] * deltas[time[0] * 4u * H + row];
    }
    upstream[id.x] = sum;
}`,
        updateLstm: `
@group(0) @binding(0) var<storage, read_write> weights: array<f32>;
@group(0) @binding(1) var<storage, read> features: array<f32>;
@group(0) @binding(2) var<storage, read> states: array<f32>;
@group(0) @binding(3) var<storage, read> deltas: array<f32>;
@group(0) @binding(4) var<storage, read> count: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= 4u * H * COLS) { return; }
    let row = id.x / COLS; let column = id.x % COLS;
    var sum = 0.0;
    for (var t = 0u; t < count[0]; t++) {
        var value = 1.0;
        if (column < INPUTS) { value = features[t * INPUTS + column]; }
        else if (column < INPUTS + H) { value = states[t * 6u * H + column - INPUTS]; }
        sum += deltas[t * 4u * H + row] * value;
    }
    weights[id.x] = clamp(weights[id.x] + sum, -LIMIT, LIMIT);
}`,
        updateOutput: `
@group(0) @binding(0) var<storage, read_write> weights: array<f32>;
@group(0) @binding(1) var<storage, read> states: array<f32>;
@group(0) @binding(2) var<storage, read> gradient: array<f32>;
@group(0) @binding(3) var<storage, read> count: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= OUTPUTS * (H + 1u)) { return; }
    let column = id.x % (H + 1u); var value = 1.0;
    if (column < H) { value = states[count[0] * 6u * H + column]; }
    weights[id.x] = clamp(weights[id.x] + gradient[id.x / (H + 1u)] * value, -LIMIT, LIMIT);
}`,
    };
    return Object.fromEntries(Object.entries(shaders).map(([name, source]) => [name, constants + source]));
}
