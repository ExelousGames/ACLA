import { DRIVING_GRU_GATES, DrivingModelConfig, drivingPolicyInputs } from './driving-model-config';

// Each step stores h, update, reset, candidate, and the candidate recurrent affine sum.
export const drivingGruStateWidth = DRIVING_GRU_GATES + 2;
// Candidate input/bias and recurrent weights receive different derivatives.
export const drivingGruDeltaWidth = DRIVING_GRU_GATES + 1;

export function drivingGruShaders(config: DrivingModelConfig): Record<string, string> {
    const constants = `
const INPUTS = ${drivingPolicyInputs(config)}u;
const H = ${config.network.hiddenUnits}u;
const COLS = INPUTS + H + 1u;
const OUTPUTS = ${config.actions.planTimesSeconds.length * config.actions.outputs.length + 1}u;
const GATE_SIZE = ${DRIVING_GRU_GATES}u * H;
const STATE_STRIDE = ${drivingGruStateWidth}u * H;
const DELTA_STRIDE = ${drivingGruDeltaWidth}u * H;
const CACHE_STRIDE = GATE_SIZE + ${DRIVING_GRU_GATES}u;
const NORMALIZATION_EPSILON = ${Number(config.network.layerNormEpsilon).toExponential()}f;

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
    const forwardAffines = `
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> features: array<f32>;
@group(0) @binding(2) var<storage, read_write> states: array<f32>;
@group(0) @binding(3) var<storage, read> time: array<u32>;
var<workgroup> sums: array<vec4f, 64>;

// Keep the candidate's input and recurrent terms separate so the reset gate
// multiplies only the recurrent term.
fn gateAffines(unit: u32, local: u32) -> vec4f {
    let updateRow = unit * COLS;
    let resetRow = (H + unit) * COLS;
    let candidateRow = (2u * H + unit) * COLS;
    let previous = time[0] * STATE_STRIDE;
    var sum = vec4f(0.0);
    for (var j = local; j < INPUTS + H; j += 64u) {
        let updateWeight = weights[updateRow + j];
        let resetWeight = weights[resetRow + j];
        let candidateWeight = weights[candidateRow + j];
        if (j < INPUTS) {
            sum += vec4f(updateWeight, resetWeight, candidateWeight, 0.0) * features[time[0] * INPUTS + j];
        } else {
            sum += vec4f(updateWeight, resetWeight, 0.0, candidateWeight) * states[previous + j - INPUTS];
        }
    }
    sums[local] = sum; workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if (local < stride) { sums[local] += sums[local + stride]; }
        workgroupBarrier();
    }
    return sums[0] + vec4f(weights[updateRow + COLS - 1u], weights[resetRow + COLS - 1u],
        weights[candidateRow + COLS - 1u], 0.0);
}
`;
    const normBackward = (firstGate: number) => `
@group(0) @binding(0) var<storage, read> cache: array<f32>;
@group(0) @binding(1) var<storage, read> affine: array<f32>;
@group(0) @binding(2) var<storage, read_write> deltas: array<f32>;
@group(0) @binding(3) var<storage, read> time: array<u32>;
@group(0) @binding(4) var<storage, read_write> gradient: array<f32>;
var<workgroup> sums: array<vec2f, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) local: u32) {
    let gate = group.x + ${firstGate}u;
    let base = time[0] * CACHE_STRIDE;
    let offset = time[0] * DELTA_STRIDE;
    var sum = vec2f(0.0);
    for (var unit = local; unit < H; unit += 64u) {
        let index = gate * H + unit;
        let weighted = deltas[offset + index] * affine[index];
        sum += vec2f(weighted, weighted * cache[base + index]);
    }
    sums[local] = sum; workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if (local < stride) { sums[local] += sums[local + stride]; }
        workgroupBarrier();
    }
    let means = sums[0] / f32(H);
    let inverseStd = cache[base + GATE_SIZE + gate];
    for (var unit = local; unit < H; unit += 64u) {
        let index = gate * H + unit;
        let normalized = cache[base + index];
        let delta = deltas[offset + index];
        gradient[index] += delta * normalized;
        gradient[GATE_SIZE + index] += delta;
        deltas[offset + index] = inverseStd * (delta * affine[index] - means.x - normalized * means.y);
    }
}`;
    const shaders = {
        forward: forwardAffines + `
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) local: u32) {
    let raw = gateAffines(group.x, local);
    if (local == 0u) {
        let previous = time[0] * STATE_STRIDE + group.x; let current = previous + STATE_STRIDE;
        let update = stableSigmoid(raw.x);
        let reset = stableSigmoid(raw.y);
        let candidate = stableTanh(raw.z + reset * raw.w);
        states[current] = update * states[previous] + (1.0 - update) * candidate;
        states[current + H] = update;
        states[current + 2u * H] = reset;
        states[current + 3u * H] = candidate;
        states[current + 4u * H] = raw.w;
    }
}`,
        gateSums: forwardAffines + `
@group(0) @binding(4) var<storage, read_write> cache: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) local: u32) {
    let raw = gateAffines(group.x, local);
    if (local == 0u) {
        let base = time[0] * CACHE_STRIDE + group.x;
        cache[base] = raw.x;
        cache[base + H] = raw.y;
        cache[base + 2u * H] = raw.z;
        states[(time[0] + 1u) * STATE_STRIDE + 4u * H + group.x] = raw.w;
    }
}`,
        // Layer normalization runs across the H units of each gate. The reset
        // gate must be applied before normalizing the combined candidate affine.
        normalizedForward: `
@group(0) @binding(0) var<storage, read_write> cache: array<f32>;
@group(0) @binding(1) var<storage, read> affine: array<f32>;
@group(0) @binding(2) var<storage, read_write> states: array<f32>;
@group(0) @binding(3) var<storage, read> time: array<u32>;
var<workgroup> sums: array<f32, 64>;

fn reduceSum(value: f32, local: u32) -> f32 {
    sums[local] = value; workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if (local < stride) { sums[local] += sums[local + stride]; }
        workgroupBarrier();
    }
    let result = sums[0];
    workgroupBarrier();
    return result;
}

fn normalizeGate(base: u32, gate: u32, local: u32) {
    let offset = base + gate * H;
    var sum = 0.0;
    for (var unit = local; unit < H; unit += 64u) { sum += cache[offset + unit]; }
    let mean = reduceSum(sum, local) / f32(H);
    var squared = 0.0;
    for (var unit = local; unit < H; unit += 64u) {
        let centered = cache[offset + unit] - mean;
        squared += centered * centered;
    }
    let inverseStd = inverseSqrt(reduceSum(squared, local) / f32(H) + NORMALIZATION_EPSILON);
    if (local == 0u) { cache[base + GATE_SIZE + gate] = inverseStd; }
    for (var unit = local; unit < H; unit += 64u) {
        cache[offset + unit] = (cache[offset + unit] - mean) * inverseStd;
    }
}

fn normalizedAffine(base: u32, index: u32) -> f32 {
    return cache[base + index] * affine[index] + affine[GATE_SIZE + index];
}

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) local: u32) {
    let previous = time[0] * STATE_STRIDE; let current = previous + STATE_STRIDE;
    let base = time[0] * CACHE_STRIDE;
    normalizeGate(base, 0u, local);
    normalizeGate(base, 1u, local);
    for (var unit = local; unit < H; unit += 64u) {
        let reset = stableSigmoid(normalizedAffine(base, H + unit));
        states[current + H + unit] = stableSigmoid(normalizedAffine(base, unit));
        states[current + 2u * H + unit] = reset;
        cache[base + 2u * H + unit] += reset * states[current + 4u * H + unit];
    }
    normalizeGate(base, 2u, local);
    for (var unit = local; unit < H; unit += 64u) {
        let update = states[current + H + unit];
        let candidate = stableTanh(normalizedAffine(base, 2u * H + unit));
        states[current + 3u * H + unit] = candidate;
        states[current + unit] = update * states[previous + unit] + (1.0 - update) * candidate;
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
    for (var i = 0u; i < H; i++) { sum += weights[id.x * (H + 1u) + i] * states[count[0] * STATE_STRIDE + i]; }
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
    let previous = time[0] * STATE_STRIDE + id.x; let current = previous + STATE_STRIDE;
    let update = states[current + H]; let candidate = states[current + 3u * H];
    let dh = upstream[id.x];
    let offset = time[0] * DELTA_STRIDE + id.x;
    deltas[offset] = dh * (states[previous] - candidate) * update * (1.0 - update);
    deltas[offset + 2u * H] = dh * (1.0 - update) * (1.0 - candidate * candidate);
    upstream[id.x] = dh * update;
}`,
        // Propagate the candidate through its normalization before differentiating
        // the reset gate. The update and reset normalization run after that.
        candidateNormBackward: normBackward(2),
        resetGradient: `
@group(0) @binding(0) var<storage, read> states: array<f32>;
@group(0) @binding(1) var<storage, read_write> deltas: array<f32>;
@group(0) @binding(2) var<storage, read> time: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= H) { return; }
    let current = (time[0] + 1u) * STATE_STRIDE + id.x;
    let reset = states[current + 2u * H];
    let offset = time[0] * DELTA_STRIDE + id.x;
    let candidateDelta = deltas[offset + 2u * H];
    deltas[offset + H] = candidateDelta * states[current + 4u * H] * reset * (1.0 - reset);
    deltas[offset + 3u * H] = candidateDelta * reset;
}`,
        gateNormBackward: normBackward(0),
        recurrentGradient: `
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> deltas: array<f32>;
@group(0) @binding(2) var<storage, read_write> upstream: array<f32>;
@group(0) @binding(3) var<storage, read> time: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= H) { return; }
    let offset = time[0] * DELTA_STRIDE;
    var sum = 0.0;
    for (var row = 0u; row < 2u * H; row++) {
        sum += weights[row * COLS + INPUTS + id.x] * deltas[offset + row];
    }
    for (var unit = 0u; unit < H; unit++) {
        sum += weights[(2u * H + unit) * COLS + INPUTS + id.x] * deltas[offset + 3u * H + unit];
    }
    upstream[id.x] += sum;
}`,
        gruWeightGradient: `
@group(0) @binding(0) var<storage, read_write> gradient: array<f32>;
@group(0) @binding(1) var<storage, read> features: array<f32>;
@group(0) @binding(2) var<storage, read> states: array<f32>;
@group(0) @binding(3) var<storage, read> deltas: array<f32>;
@group(0) @binding(4) var<storage, read> count: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= GATE_SIZE * COLS) { return; }
    let row = id.x / COLS; let column = id.x % COLS;
    let recurrent = column >= INPUTS && column < INPUTS + H;
    let deltaRow = select(row, row + H, recurrent && row >= 2u * H);
    var sum = 0.0;
    for (var t = 0u; t < count[0]; t++) {
        var value = 1.0;
        if (column < INPUTS) { value = features[t * INPUTS + column]; }
        else if (recurrent) { value = states[t * STATE_STRIDE + column - INPUTS]; }
        sum += deltas[t * DELTA_STRIDE + deltaRow] * value;
    }
    gradient[id.x] += sum;
}`,
        outputWeightGradient: `
@group(0) @binding(0) var<storage, read_write> weightGradient: array<f32>;
@group(0) @binding(1) var<storage, read> states: array<f32>;
@group(0) @binding(2) var<storage, read> gradient: array<f32>;
@group(0) @binding(3) var<storage, read> count: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= OUTPUTS * (H + 1u)) { return; }
    let column = id.x % (H + 1u); var value = 1.0;
    if (column < H) { value = states[count[0] * STATE_STRIDE + column]; }
    weightGradient[id.x] += gradient[id.x / (H + 1u)] * value;
}`,
    };
    return Object.fromEntries(Object.entries(shaders).map(([name, source]) => [name, constants + source]));
}
