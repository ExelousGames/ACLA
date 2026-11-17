import { DrivingModelConfig, drivingExplorationSigmas } from './driving-model-config';
import { drivingLearningRate } from './driving-update-shaders';

/** SAC on the existing clipped-tanh pedals, tanh steering and categorical gear.
 * Clipped pedals have a point mass at zero. The score-function estimator below
 * differentiates that mixed distribution, using fresh CURRENT-policy samples. */
export function drivingSacShaders(config: DrivingModelConfig): Record<string, string> {
    const learning = config.learning, rate = drivingLearningRate(config) || 1;
    const constants = `
const H = ${config.network.hiddenUnits}u;
const QCOLS = H + 7u;
const QSIZE = H * QCOLS + H + 1u;
const OUTPUTS = ${config.actions.planTimesSeconds.length * config.actions.outputs.length + 1}u;
const ALPHA = ${learning.entropyTemperature.toExponential()}f;
const ACTOR_RATE = ${(learning.actorRate / rate).toExponential()}f;
const CRITIC_RATE = ${(learning.criticRate / rate).toExponential()}f;
const POLICY_LIMIT = ${learning.policyGradientLimit.toExponential()}f;
const CRITIC_LIMIT = ${learning.criticGradientLimit.toExponential()}f;
const TEMPERATURE = ${learning.shiftTemperature.toExponential()}f;
// Retain the user-facing exploration controls as multipliers on learned spread.
const STD_OFFSETS = vec3f(${drivingExplorationSigmas(config).map((sigma) => `${Math.log(sigma / 0.05).toExponential()}f`).join(', ')});
fn stableTanh(value: f32) -> f32 { return tanh(clamp(value, -10.0, 10.0)); }
`;
    const actionFunctions = `
// Evaluation 0 is the replayed action, 1..3 the current policy and 4..6 the next policy.
fn actionValue(evaluation: u32, axis: u32) -> f32 {
    if (evaluation == 0u) { return transition[11u + axis]; }
    return samples[select(0u, 13u, evaluation >= 4u) + axis];
}
fn gearChoice(evaluation: u32) -> u32 {
    if (evaluation == 0u) { return u32(transition[14]); }
    return (evaluation - 1u) % 3u;
}
fn inputValue(evaluation: u32, column: u32) -> f32 {
    if (column < H) { return features[select(0u, H, evaluation >= 4u) + column]; }
    if (column < H + 3u) { return actionValue(evaluation, column - H); }
    if (column < H + 6u) { return select(0.0, 1.0, column - H - 3u == gearChoice(evaluation)); }
    return 1.0;
}
`;
    return {
        sacSample: constants + `
@group(0) @binding(0) var<storage, read> outputs: array<f32>;
@group(0) @binding(1) var<storage, read> nextOutputs: array<f32>;
@group(0) @binding(2) var<storage, read> logStds: array<f32>;
@group(0) @binding(3) var<storage, read> transition: array<f32>;
// Each policy: controls[3], analog log probability, mean scores[3], std scores[3], gear probabilities[3].
@group(0) @binding(4) var<storage, read_write> samples: array<f32>;
fn logNormalCdf(z: f32) -> f32 {
    // Evaluate the tail in log space so a near-impossible zero pedal remains finite.
    let t = 1.0 / (1.0 + 0.2316419 * abs(z));
    let polynomial = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let tail = -0.5 * z * z - 0.9189385332 + log(polynomial);
    if (z <= 0.0) { return tail; }
    return log(max(1e-30, 1.0 - exp(tail)));
}
fn normalHazard(z: f32, logCdf: f32) -> f32 {
    if (z <= 0.0) {
        let t = 1.0 / (1.0 + 0.2316419 * abs(z));
        let polynomial = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
        return 1.0 / polynomial;
    }
    return exp(-0.5 * z * z - 0.9189385332 - logCdf);
}
fn outputValue(next: bool, index: u32) -> f32 {
    if (next) { return nextOutputs[index]; }
    return outputs[index];
}
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let next = id.x == 1u; let base = id.x * 13u;
    let slot = select(u32(transition[3]), 0u, next);
    var logProbability = 0.0;
    for (var axis = 0u; axis < 3u; axis++) {
        let mean = outputValue(next, slot * 6u + axis);
        let logStd = clamp(logStds[slot * 3u + axis], -7.0, 1.0) + STD_OFFSETS[axis];
        let sigma = exp(logStd); let noise = transition[15u + id.x * 3u + axis];
        let latent = mean + sigma * noise; let squashed = stableTanh(latent);
        var meanScore = noise / sigma; var stdScore = noise * noise - 1.0;
        var logDensity = -0.5 * noise * noise - logStd - 0.9189385332;
        // Stable log(1 - tanh(latent)^2), including saturated actions.
        let magnitude = abs(latent);
        logDensity -= 2.0 * (0.6931471806 - magnitude - log(1.0 + exp(-2.0 * magnitude)));
        var action = squashed;
        if (axis < 2u && latent <= 0.0) {
            action = 0.0;
            let z = -mean / sigma; logDensity = logNormalCdf(z);
            let hazard = normalHazard(z, logDensity);
            meanScore = -hazard / sigma; stdScore = -z * hazard;
        }
        samples[base + axis] = action;
        samples[base + 4u + axis] = meanScore;
        samples[base + 7u + axis] = stdScore;
        logProbability += logDensity;
    }
    samples[base + 3u] = logProbability;
    let mask = select(7u, 9u, next);
    let allowed = array<bool, 3>(true, transition[mask] != 0.0, transition[mask + 1u] != 0.0);
    var maximum = outputValue(next, slot * 6u + 3u);
    for (var gear = 1u; gear < 3u; gear++) {
        if (allowed[gear]) { maximum = max(maximum, outputValue(next, slot * 6u + 3u + gear)); }
    }
    var total = 0.0;
    for (var gear = 0u; gear < 3u; gear++) {
        var probability = 0.0;
        if (allowed[gear]) { probability = exp((outputValue(next, slot * 6u + 3u + gear) - maximum) / TEMPERATURE); }
        samples[base + 10u + gear] = probability; total += probability;
    }
    for (var gear = 0u; gear < 3u; gear++) { samples[base + 10u + gear] /= total; }
}`,
        sacCriticForward: constants + `
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> targetWeights: array<f32>;
@group(0) @binding(2) var<storage, read> features: array<f32>;
@group(0) @binding(3) var<storage, read> transition: array<f32>;
@group(0) @binding(4) var<storage, read> samples: array<f32>;
@group(0) @binding(5) var<storage, read_write> activations: array<f32>;
@group(0) @binding(6) var<storage, read_write> values: array<f32>;
var<workgroup> sums: array<f32, 64>;
${actionFunctions}
fn weightValue(useTarget: bool, index: u32) -> f32 {
    if (useTarget) { return targetWeights[index]; }
    return weights[index];
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) local: u32) {
    let evaluation = group.x / 2u; let critic = group.x % 2u;
    let useTarget = evaluation >= 4u; let offset = critic * QSIZE;
    var result = 0.0;
    for (var unit = local; unit < H; unit += 64u) {
        var sum = 0.0;
        for (var column = 0u; column < QCOLS; column++) {
            sum += weightValue(useTarget, offset + unit * QCOLS + column) * inputValue(evaluation, column);
        }
        let hidden = stableTanh(sum); activations[group.x * H + unit] = hidden;
        result += hidden * weightValue(useTarget, offset + H * QCOLS + unit);
    }
    sums[local] = result; workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if (local < stride) { sums[local] += sums[local + stride]; }
        workgroupBarrier();
    }
    if (local == 0u) { values[group.x] = sums[0] + weightValue(useTarget, offset + QSIZE - 1u); }
}`,
        sacLoss: constants + `
@group(0) @binding(0) var<storage, read> transition: array<f32>;
@group(0) @binding(1) var<storage, read> samples: array<f32>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var<storage, read_write> errors: array<f32>;
@group(0) @binding(4) var<storage, read_write> outputGradient: array<f32>;
@group(0) @binding(5) var<storage, read_write> stdGradient: array<f32>;
@compute @workgroup_size(1)
fn main() {
    var nextValue = 0.0; var currentValue = 0.0;
    var gearObjectives: array<f32, 3>;
    for (var gear = 0u; gear < 3u; gear++) {
        let probability = samples[10u + gear];
        let objective = min(values[(1u + gear) * 2u], values[(1u + gear) * 2u + 1u]) - ALPHA * log(max(probability, 1e-30));
        gearObjectives[gear] = objective; currentValue += probability * objective;
        let nextProbability = samples[23u + gear];
        nextValue += nextProbability * (min(values[(4u + gear) * 2u], values[(4u + gear) * 2u + 1u]) - ALPHA * log(max(nextProbability, 1e-30)));
    }
    var bellmanTarget = transition[0];
    if (transition[2] == 0.0 && ${learning.discount.toExponential()}f > 0.0) {
        bellmanTarget += pow(${learning.discount.toExponential()}f, transition[1] / ${learning.discountIntervalSeconds.toExponential()}f) * (nextValue - ALPHA * samples[16]);
    }
    errors[0] = transition[6] * CRITIC_RATE * clamp(bellmanTarget - values[0], -CRITIC_LIMIT, CRITIC_LIMIT);
    errors[1] = transition[6] * CRITIC_RATE * clamp(bellmanTarget - values[1], -CRITIC_LIMIT, CRITIC_LIMIT);
    let slot = u32(transition[3]);
    // Only fresh policy actions enter this objective; the replay action trains Q above.
    let objective = currentValue - ALPHA * (samples[3] + 1.0);
    for (var axis = 0u; axis < 3u; axis++) {
        outputGradient[slot * 6u + axis] += transition[4] * ACTOR_RATE * clamp(samples[4u + axis] * objective, -POLICY_LIMIT, POLICY_LIMIT);
        stdGradient[slot * 3u + axis] += transition[4] * ACTOR_RATE * clamp(samples[7u + axis] * objective, -POLICY_LIMIT, POLICY_LIMIT);
    }
    for (var gear = 0u; gear < 3u; gear++) {
        outputGradient[slot * 6u + 3u + gear] += transition[5] * ACTOR_RATE * clamp(samples[10u + gear] * (gearObjectives[gear] - currentValue) / TEMPERATURE, -POLICY_LIMIT, POLICY_LIMIT);
    }
}`,
        sacCriticWeightGradient: constants + `
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> features: array<f32>;
@group(0) @binding(2) var<storage, read> transition: array<f32>;
@group(0) @binding(3) var<storage, read> samples: array<f32>;
@group(0) @binding(4) var<storage, read> activations: array<f32>;
@group(0) @binding(5) var<storage, read> errors: array<f32>;
@group(0) @binding(6) var<storage, read_write> gradients: array<f32>;
${actionFunctions}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= 2u * QSIZE) { return; }
    let critic = id.x / QSIZE; let index = id.x % QSIZE;
    var derivative = 1.0;
    if (index < H * QCOLS) {
        let unit = index / QCOLS; let activation = activations[critic * H + unit];
        derivative = weights[critic * QSIZE + H * QCOLS + unit] * (1.0 - activation * activation) * inputValue(0u, index % QCOLS);
    } else if (index < QSIZE - 1u) { derivative = activations[critic * H + index - H * QCOLS]; }
    gradients[id.x] += errors[critic] * derivative;
}`,
        sacCriticHiddenGradient: constants + `
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> activations: array<f32>;
@group(0) @binding(2) var<storage, read> errors: array<f32>;
@group(0) @binding(3) var<storage, read_write> upstream: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= H) { return; }
    var sum = 0.0;
    for (var critic = 0u; critic < 2u; critic++) {
        for (var unit = 0u; unit < H; unit++) {
            let activation = activations[critic * H + unit];
            sum += errors[critic] * weights[critic * QSIZE + H * QCOLS + unit] * (1.0 - activation * activation) * weights[critic * QSIZE + unit * QCOLS + id.x];
        }
    }
    upstream[id.x] += sum;
}`,
        sacAddHidden: constants + `
@group(0) @binding(0) var<storage, read> critic: array<f32>;
@group(0) @binding(1) var<storage, read_write> upstream: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x < H) { upstream[id.x] += critic[id.x]; }
}`,
        sacForecastGradient: constants + `
@group(0) @binding(0) var<storage, read> outputs: array<f32>;
@group(0) @binding(1) var<storage, read> teacher: array<f32>;
// Per future slot: analog weight, categorical weight, up allowed, down allowed.
@group(0) @binding(2) var<storage, read> targets: array<f32>;
@group(0) @binding(3) var<storage, read> logStds: array<f32>;
@group(0) @binding(4) var<storage, read_write> stdGradient: array<f32>;
@group(0) @binding(5) var<storage, read_write> gradient: array<f32>;
fn probability(offset: u32, gear: u32, allowed: array<bool, 3>, teaching: bool) -> f32 {
    if (!allowed[gear]) { return 0.0; }
    var logits: array<f32, 3>;
    for (var i = 0u; i < 3u; i++) {
        logits[i] = outputs[offset + i];
        if (teaching) { logits[i] = teacher[3u + i]; }
    }
    var maximum = logits[0];
    for (var i = 1u; i < 3u; i++) { if (allowed[i]) { maximum = max(maximum, logits[i]); } }
    var total = 0.0;
    for (var i = 0u; i < 3u; i++) { if (allowed[i]) { total += exp((logits[i] - maximum) / TEMPERATURE); } }
    return exp((logits[gear] - maximum) / TEMPERATURE) / total;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= OUTPUTS) { return; }
    gradient[id.x] = 0.0;
    if (id.x == OUTPUTS - 1u) { return; }
    let slot = id.x / 6u; let axis = id.x % 6u;
    if (slot == 0u) { return; }
    let base = slot * 4u;
    if (axis < 3u) {
        gradient[id.x] = targets[base] * ACTOR_RATE * clamp(teacher[axis] - outputs[id.x], -POLICY_LIMIT, POLICY_LIMIT);
        stdGradient[slot * 3u + axis] += targets[base] * ACTOR_RATE * clamp(logStds[axis] - logStds[slot * 3u + axis], -POLICY_LIMIT, POLICY_LIMIT);
    } else {
        let allowed = array<bool, 3>(true, targets[base + 2u] != 0.0, targets[base + 3u] != 0.0);
        let score = (probability(slot * 6u + 3u, axis - 3u, allowed, true) - probability(slot * 6u + 3u, axis - 3u, allowed, false)) / TEMPERATURE;
        gradient[id.x] = targets[base + 1u] * ACTOR_RATE * clamp(score, -POLICY_LIMIT, POLICY_LIMIT);
    }
}`,
        sacApplyLogStd: `
@group(0) @binding(0) var<storage, read_write> weights: array<f32>;
@group(0) @binding(1) var<storage, read> gradients: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= arrayLength(&weights) || stats[5] == 0.0) { return; }
    weights[id.x] = clamp(weights[id.x] + gradients[id.x] * stats[1] * stats[3], -7.0, 1.0);
}`,
        sacTargetUpdate: `
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read_write> targetWeights: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= arrayLength(&weights) || stats[5] == 0.0 || stats[3] == 0.0) { return; }
    targetWeights[id.x] += ${learning.targetTau.toExponential()}f * (weights[id.x] - targetWeights[id.x]);
}`,
    };
}
