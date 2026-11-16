import { DrivingModelConfig } from './driving-model-config';

export const drivingLearningRate = (config: DrivingModelConfig) =>
    Math.max(config.learning.actorRate, config.learning.criticRate, config.learning.demonstrationRate);

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
// Summed gradient norm, clip scale, proposed update norm, base rate, sample count, finite flag.
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
    let norm = scale * sqrt(total(squares, local));
    let valid = bad == 0.0 && finite(norm);
    let clip = min(1.0, ${config.learning.gradientNormLimit.toExponential()}f / max(norm, 1e-30));
    if (local == 0u) {
        stats[0] = norm; stats[1] = select(0.0, clip, valid);
        stats[2] = select(0.0, norm * clip * ${drivingLearningRate(config).toExponential()}f, valid);
        stats[3] = ${drivingLearningRate(config).toExponential()}f;
        stats[4] = samples; stats[5] = select(0.0, 1.0, valid);
    }
}`,
        applyGradient: apply(config.learning.weightLimit),
        applyNormGradient: apply(Math.max(1, config.learning.weightLimit)),
    };
}
