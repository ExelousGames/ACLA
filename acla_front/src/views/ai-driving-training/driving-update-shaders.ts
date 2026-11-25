import { DrivingModelConfig, drivingCameraInputs, drivingFeatureSizes, drivingPolicyInputs } from './driving-model-config';

export const drivingLearningRate = (config: DrivingModelConfig) =>
    Math.max(config.learning.actorRate, config.learning.criticRate, config.learning.demonstrationRate);

const telemetryInputGains: Record<string, readonly number[]> = {
    localVelocity: [5, 5, 5], speed: [5], controls: [2, 3, 6], gear: [8], rpm: [16], yawRate: [8], gForce: [5, 5, 5],
};

function recordedGruGainShader(config: DrivingModelConfig): string {
    const sizes = drivingFeatureSizes(config);
    const columns = new Map<number, number[]>();
    let offset = 2; // Presence and age precede the observation groups.
    for (const name of config.observation.features) {
        (telemetryInputGains[name] ?? []).forEach((gain, index) => {
            const matching = columns.get(gain);
            if (matching) matching.push(offset + index);
            else columns.set(gain, [offset + index]);
        });
        offset += name === 'camera' ? drivingCameraInputs(config) : sizes[name];
    }
    const stride = drivingPolicyInputs(config) + config.network.hiddenUnits + 1;
    const cases = Array.from(columns, ([gain, indices]) =>
        `case ${indices.map((index) => `${index}u`).join(', ')}: { return ${gain}.0; }`).join('\n        ');
    return `
fn recordedGruGain(index: u32) -> f32 {
    switch (index % ${stride}u) {
        ${cases}
        default: { return 1.0; }
    }
}
`;
}

/** Reduce every accumulated parameter gradient before writing any weights. */
export function drivingUpdateShaders(config: DrivingModelConfig): Record<string, string> {
    const reduction = `
var<workgroup> values: array<f32, 64>;
fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
fn maximum(value: f32, local: u32) -> f32 {
    values[local] = value; workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if (local < stride) { values[local] = max(values[local], values[local + stride]); }
        workgroupBarrier();
    }
    let result = values[0]; workgroupBarrier(); return result;
}
fn total(value: f32, local: u32) -> f32 {
    values[local] = value; workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if (local < stride) { values[local] += values[local + stride]; }
        workgroupBarrier();
    }
    let result = values[0]; workgroupBarrier(); return result;
}
`;
    const apply = (limit: number) => `
@group(0) @binding(0) var<storage, read_write> weights: array<f32>;
@group(0) @binding(1) var<storage, read> gradient: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= arrayLength(&weights) || stats[5] == 0.0 || stats[3] == 0.0) { return; }
    let delta = gradient[id.x] * stats[1] * stats[3];
    weights[id.x] = clamp(weights[id.x] + delta, -${limit.toExponential()}f, ${limit.toExponential()}f);
}`;
    const adamRate = config.learning.demonstrationRate.toExponential();
    const gruGain = recordedGruGainShader(config);
    const applyAdam = (limit: number, gru = false) => reduction + (gru ? gruGain : '') + `
@group(0) @binding(0) var<storage, read_write> weights: array<f32>;
@group(0) @binding(1) var<storage, read> gradient: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read_write> firstMoment: array<f32>;
@group(0) @binding(4) var<storage, read_write> secondMoment: array<f32>;
@group(0) @binding(5) var<storage, read> correction: array<f32>;
@group(0) @binding(6) var<storage, read_write> partials: array<f32>;
@group(0) @binding(7) var<storage, read> offset: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) local: u32) {
    let start = group.x * 256u;
    let end = min(start + 256u, arrayLength(&weights));
    var squares = 0.0;
    if (stats[5] != 0.0 && stats[3] > 0.0) {
        for (var i = start + local; i < end; i += 64u) {
            let g = gradient[i] * stats[1];
            let m = 0.9 * firstMoment[i] + 0.1 * g;
            let v = 0.999 * secondMoment[i] + 0.001 * g * g;
            firstMoment[i] = m;
            secondMoment[i] = v;
            let delta = ${gru ? 'recordedGruGain(i) * ' : ''}stats[3] * m * correction[2] / (sqrt(v * correction[3]) + 1e-8);
            let old = weights[i];
            let updated = clamp(old + delta, -${limit.toExponential()}f, ${limit.toExponential()}f);
            weights[i] = updated;
            let applied = updated - old;
            squares += applied * applied;
        }
    }
    let subtotal = total(squares, local);
    if (local == 0u) { partials[offset[0] + group.x] = subtotal; }
}`;
    return {
        gradientPartialNorm: reduction + `
@group(0) @binding(0) var<storage, read> gradient: array<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<f32>;
@group(0) @binding(2) var<storage, read> offset: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) local: u32) {
    let start = group.x * 256u; let end = min(start + 256u, arrayLength(&gradient));
    var largest = 0.0; var invalid = 0.0;
    for (var i = start + local; i < end; i += 64u) {
        if (!finite(gradient[i])) { invalid = 1.0; }
        else { largest = max(largest, abs(gradient[i])); }
    }
    let bad = maximum(invalid, local);
    let scale = maximum(largest, local);
    var squares = 0.0;
    for (var i = start + local; i < end; i += 64u) {
        if (bad == 0.0 && scale > 0.0) { let value = gradient[i] / scale; squares += value * value; }
    }
    let norm = scale * sqrt(total(squares, local));
    if (local == 0u) { partials[offset[0] + group.x] = select(-1.0, norm, bad == 0.0 && finite(norm)); }
}`,
        gradientGlobalNorm: reduction + `
@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read> count: array<u32>;
// Mean gradient norm, clip/averaging scale, proposed update norm, base rate, sample count, finite flag.
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) local: u32) {
    var largest = 0.0; var invalid = 0.0;
    for (var i = local; i < arrayLength(&partials); i += 64u) {
        if (partials[i] < 0.0) { invalid = 1.0; }
        largest = max(largest, partials[i]);
    }
    let bad = maximum(invalid, local); let scale = maximum(largest, local);
    var squares = 0.0;
    for (var i = local; i < arrayLength(&partials); i += 64u) {
        if (scale > 0.0) { let value = partials[i] / scale; squares += value * value; }
    }
    let samples = f32(max(1u, count[0]));
    let norm = scale * sqrt(total(squares, local)) / samples;
    let valid = bad == 0.0 && finite(norm);
    let clip = min(1.0, ${config.learning.gradientNormLimit.toExponential()}f / max(norm, 1e-30));
    if (local == 0u) {
        stats[0] = norm; stats[1] = select(0.0, clip / samples, valid);
        stats[2] = select(0.0, norm * clip * ${drivingLearningRate(config).toExponential()}f, valid);
        stats[3] = ${drivingLearningRate(config).toExponential()}f;
        stats[4] = samples; stats[5] = select(0.0, 1.0, valid);
    }
}`,
        applyGradient: apply(config.learning.weightLimit),
        applyNormGradient: apply(Math.max(1, config.learning.weightLimit)),
        recordedAdamAdvance: `
@group(0) @binding(0) var<storage, read_write> stats: array<f32>;
@group(0) @binding(1) var<storage, read_write> correction: array<f32>;
@compute @workgroup_size(1)
fn main() {
    stats[3] = ${adamRate}f;
    if (stats[5] == 0.0 || stats[3] == 0.0) { return; }
    correction[0] *= 0.9;
    correction[1] *= 0.999;
    correction[2] = 1.0 / (1.0 - correction[0]);
    correction[3] = 1.0 / (1.0 - correction[1]);
}`,
        recordedAdamApply: applyAdam(config.learning.weightLimit),
        recordedAdamApplyNorm: applyAdam(Math.max(1, config.learning.weightLimit)),
        recordedAdamApplyGru: applyAdam(config.learning.weightLimit, true),
        recordedAdamConditionGru: gruGain + `
@group(0) @binding(0) var<storage, read_write> gradient: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x < arrayLength(&gradient)) { gradient[id.x] *= recordedGruGain(id.x); }
}`,
        recordedAdamUpdateNorm: reduction + `
@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> stats: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) local: u32) {
    var squares = 0.0;
    for (var i = local; i < arrayLength(&partials); i += 64u) {
        squares += max(0.0, partials[i]);
    }
    let norm = sqrt(total(squares, local));
    if (local == 0u) { stats[2] = norm; }
}`,
    };
}
