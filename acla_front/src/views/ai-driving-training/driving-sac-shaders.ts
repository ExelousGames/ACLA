import { DRIVING_ANALOG_AXES, DRIVING_GEAR_COUNT, DrivingModelConfig, drivingExplorationSigmas } from './driving-model-config';
import { drivingGruStateWidth } from './driving-gru-shaders';
import { drivingLearningRate } from './driving-update-shaders';

export const sacPolicySampleFloats = 5 * DRIVING_ANALOG_AXES + 1 + DRIVING_GEAR_COUNT;
export const sacCurrentGearLimitOffset = 7;
export const sacNextGearLimitOffset = 8;

/** Hybrid SAC with reparameterized analog controls and exact gear expectations. */
export function drivingSacShaders(config: DrivingModelConfig): Record<string, string> {
    const learning = config.learning, rate = drivingLearningRate(config) || 1;
    const constants = `
const H = ${config.network.hiddenUnits}u;
const AXES = ${DRIVING_ANALOG_AXES}u;
const ACTIONS = ${config.actions.outputs.length}u;
const GEARS = ACTIONS - AXES;
const NEXT_EVALUATION = 1u + GEARS;
const BEHAVIOR_ACTION = 9u;
const NOISE = BEHAVIOR_ACTION + AXES + 1u;
const QCOLS = H + ACTIONS + 1u;
const QSIZE = H * QCOLS + H + 1u;
const ALPHA = ${learning.entropyTemperature.toExponential()}f;
const ACTOR_RATE = ${(learning.actorRate / rate).toExponential()}f;
const CRITIC_RATE = ${(learning.criticRate / rate).toExponential()}f;
const POLICY_LIMIT = ${learning.policyGradientLimit.toExponential()}f;
const CRITIC_LIMIT = ${learning.criticGradientLimit.toExponential()}f;
const TEMPERATURE = ${learning.shiftTemperature.toExponential()}f;
// Retain the user-facing exploration controls as multipliers on learned spread.
const STD_OFFSETS = vec2f(${drivingExplorationSigmas(config).map((sigma) => `${Math.log(sigma / 0.05).toExponential()}f`).join(', ')});
fn stableTanh(value: f32) -> f32 { return tanh(clamp(value, -10.0, 10.0)); }
`;
    const sampleType = `
struct PolicySample {
    action: array<f32, AXES>,
    logProbability: f32,
    actionMeanDerivative: array<f32, AXES>,
    actionLogStdDerivative: array<f32, AXES>,
    logProbabilityMeanGradient: array<f32, AXES>,
    logProbabilityLogStdGradient: array<f32, AXES>,
    gearProbabilities: array<f32, GEARS>,
}
`;
    const actionFunctions = `
// The replayed action comes first, then the current and next gear distributions.
fn actionValue(evaluation: u32, axis: u32) -> f32 {
    if (evaluation == 0u) { return transition[BEHAVIOR_ACTION + axis]; }
    return samples[select(0u, 1u, evaluation >= NEXT_EVALUATION)].action[axis];
}
fn gearChoice(evaluation: u32) -> u32 {
    if (evaluation == 0u) { return u32(transition[BEHAVIOR_ACTION + AXES]); }
    return (evaluation - 1u) % GEARS;
}
fn inputValue(evaluation: u32, column: u32) -> f32 {
    if (column < H) { return features[select(0u, H, evaluation >= NEXT_EVALUATION) + column]; }
    if (column < H + AXES) { return actionValue(evaluation, column - H); }
    if (column < H + ACTIONS) { return select(0.0, 1.0, column - H - AXES == gearChoice(evaluation)); }
    return 1.0;
}
`;
    return {
        sacSample: constants + sampleType + `
@group(0) @binding(0) var<storage, read> outputs: array<f32>;
@group(0) @binding(1) var<storage, read> nextOutputs: array<f32>;
@group(0) @binding(2) var<storage, read> logStds: array<f32>;
@group(0) @binding(3) var<storage, read> transition: array<f32>;
@group(0) @binding(4) var<storage, read_write> samples: array<PolicySample>;
fn outputValue(next: bool, index: u32) -> f32 {
    if (next) { return nextOutputs[index]; }
    return outputs[index];
}
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let next = id.x == 1u;
    let slot = select(u32(transition[3]), 0u, next);
    var logProbability = 0.0;
    for (var axis = 0u; axis < AXES; axis++) {
        let mean = outputValue(next, slot * ACTIONS + axis);
        let logStd = clamp(logStds[slot * AXES + axis], -7.0, 1.0) + STD_OFFSETS[axis];
        let sigma = exp(logStd); let noise = transition[NOISE + id.x * AXES + axis];
        let perturbation = sigma * noise;
        let latent = mean + perturbation; let squashed = stableTanh(latent);
        var logDensity = -0.5 * noise * noise - logStd - 0.9189385332;
        // Stable log(1 - tanh(latent)^2), including saturated actions.
        let magnitude = abs(latent);
        logDensity -= 2.0 * (0.6931471806 - magnitude - log(1.0 + exp(-2.0 * magnitude)));
        let actionDerivative = select(0.0, 1.0 - squashed * squashed, magnitude < 10.0);
        let logProbabilityGradient = vec2f(2.0 * squashed, 2.0 * squashed * perturbation - 1.0);
        samples[id.x].action[axis] = squashed;
        samples[id.x].actionMeanDerivative[axis] = actionDerivative;
        samples[id.x].actionLogStdDerivative[axis] = actionDerivative * perturbation;
        samples[id.x].logProbabilityMeanGradient[axis] = logProbabilityGradient.x;
        samples[id.x].logProbabilityLogStdGradient[axis] = logProbabilityGradient.y;
        logProbability += logDensity;
    }
    samples[id.x].logProbability = logProbability;
    // An interpolated future pedal/steering endpoint did not choose the gear.
    // Evaluate that analog sample under the gear command that actually ran.
    if (slot > 0u && transition[5] == 0.0) {
        for (var gear = 0u; gear < GEARS; gear++) {
            samples[id.x].gearProbabilities[gear] = select(0.0, 1.0, gear == u32(transition[BEHAVIOR_ACTION + AXES]));
        }
        return;
    }
    let allowed = u32(transition[select(${sacCurrentGearLimitOffset}u, ${sacNextGearLimitOffset}u, next)]);
    var maximum = outputValue(next, slot * ACTIONS + AXES);
    for (var gear = 1u; gear < allowed; gear++) {
        maximum = max(maximum, outputValue(next, slot * ACTIONS + AXES + gear));
    }
    var total = 0.0;
    for (var gear = 0u; gear < GEARS; gear++) {
        var probability = 0.0;
        if (gear < allowed) { probability = exp((outputValue(next, slot * ACTIONS + AXES + gear) - maximum) / TEMPERATURE); }
        samples[id.x].gearProbabilities[gear] = probability; total += probability;
    }
    for (var gear = 0u; gear < GEARS; gear++) { samples[id.x].gearProbabilities[gear] /= total; }
}`,
        sacCriticForward: constants + sampleType + `
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> targetWeights: array<f32>;
@group(0) @binding(2) var<storage, read> features: array<f32>;
@group(0) @binding(3) var<storage, read> transition: array<f32>;
@group(0) @binding(4) var<storage, read> samples: array<PolicySample>;
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
    let useTarget = evaluation >= NEXT_EVALUATION; let offset = critic * QSIZE;
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
        sacScoreState: constants + `
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> states: array<f32>;
@group(0) @binding(2) var<storage, read> count: array<u32>;
@group(0) @binding(3) var<storage, read_write> stateSums: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let unit = id.x;
    let critic = id.y;
    if (unit >= H) { return; }
    let row = critic * QSIZE + unit * QCOLS;
    let state = count[0] * ${drivingGruStateWidth}u * H;
    var sum = 0.0;
    for (var column = 0u; column < H; column++) {
        sum += weights[row + column] * states[state + column];
    }
    stateSums[critic * H + unit] = sum + weights[row + QCOLS - 1u];
}`,
        sacScoreActions: constants + `
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> stateSums: array<f32>;
@group(0) @binding(2) var<storage, read> candidates: array<f32>;
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;
var<workgroup> sums: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) local: u32) {
    let candidate = group.x / 2u * (AXES + 1u);
    let critic = group.x % 2u;
    let offset = critic * QSIZE;
    let gear = u32(candidates[candidate + AXES]);
    var result = 0.0;
    for (var unit = local; unit < H; unit += 64u) {
        let row = offset + unit * QCOLS;
        var sum = stateSums[critic * H + unit];
        for (var axis = 0u; axis < AXES; axis++) {
            sum += weights[row + H + axis] * candidates[candidate + axis];
        }
        sum += weights[row + H + AXES + gear];
        result += stableTanh(sum) * weights[offset + H * QCOLS + unit];
    }
    sums[local] = result;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if (local < stride) { sums[local] += sums[local + stride]; }
        workgroupBarrier();
    }
    if (local == 0u) { scores[group.x] = sums[0] + weights[offset + QSIZE - 1u]; }
}`,
        sacLoss: constants + sampleType + `
@group(0) @binding(0) var<storage, read> transition: array<f32>;
@group(0) @binding(1) var<storage, read> samples: array<PolicySample>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var<storage, read_write> errors: array<f32>;
@group(0) @binding(4) var<storage, read_write> outputGradient: array<f32>;
@group(0) @binding(5) var<storage, read_write> stdGradient: array<f32>;
@group(0) @binding(6) var<storage, read> weights: array<f32>;
@group(0) @binding(7) var<storage, read> activations: array<f32>;

fn criticActionGradient(evaluation: u32) -> vec2f {
    let pair = evaluation * 2u;
    let critic = select(0u, 1u, values[pair + 1u] < values[pair]);
    let offset = critic * QSIZE;
    var gradient = vec2f(0.0);
    for (var unit = 0u; unit < H; unit++) {
        let hidden = activations[(pair + critic) * H + unit];
        let derivative = weights[offset + H * QCOLS + unit] * (1.0 - hidden * hidden);
        let column = offset + unit * QCOLS + H;
        gradient += derivative * vec2f(weights[column], weights[column + 1u]);
    }
    return gradient;
}

@compute @workgroup_size(1)
fn main() {
    var currentValue = 0.0;
    var actionGradient = vec2f(0.0);
    var gearObjectives: array<f32, GEARS>;
    let referenceQ = min(values[2], values[3]);
    for (var gear = 0u; gear < GEARS; gear++) {
        let probability = samples[0].gearProbabilities[gear];
        let evaluation = 1u + gear;
        let q = min(values[evaluation * 2u], values[evaluation * 2u + 1u]);
        let objective = (q - referenceQ) - ALPHA * log(max(probability, 1e-30));
        gearObjectives[gear] = objective; currentValue += probability * objective;
        if (probability > 0.0 && transition[4] > 0.0) {
            actionGradient += probability * criticActionGradient(evaluation);
        }
    }
    let slot = u32(transition[3]);
    // Fit the critics once at the execution state. Future-slot passes consume
    // the same critics for policy improvement and need no bootstrap evaluation.
    if (slot == 0u) {
        var bellmanTarget = transition[0];
        if (transition[2] == 0.0 && ${learning.discount.toExponential()}f > 0.0) {
            var nextValue = 0.0;
            for (var gear = 0u; gear < GEARS; gear++) {
                let p = samples[1].gearProbabilities[gear];
                let pair = (NEXT_EVALUATION + gear) * 2u;
                nextValue += p * (min(values[pair], values[pair + 1u]) - ALPHA * log(max(p, 1e-30)));
            }
            bellmanTarget += pow(${learning.discount.toExponential()}f, transition[1] / ${learning.discountIntervalSeconds.toExponential()}f) * (nextValue - ALPHA * samples[1].logProbability);
        }
        errors[0] = transition[6] * CRITIC_RATE * clamp(bellmanTarget - values[0], -CRITIC_LIMIT, CRITIC_LIMIT);
        errors[1] = transition[6] * CRITIC_RATE * clamp(bellmanTarget - values[1], -CRITIC_LIMIT, CRITIC_LIMIT);
    }
    // Ascend Q minus alpha * log(pi) through the fixed-noise action sample.
    for (var axis = 0u; axis < AXES; axis++) {
        let meanGradient = actionGradient[axis] * samples[0].actionMeanDerivative[axis] - ALPHA * samples[0].logProbabilityMeanGradient[axis];
        let logStdGradient = actionGradient[axis] * samples[0].actionLogStdDerivative[axis] - ALPHA * samples[0].logProbabilityLogStdGradient[axis];
        outputGradient[slot * ACTIONS + axis] += transition[4] * ACTOR_RATE * clamp(meanGradient, -POLICY_LIMIT, POLICY_LIMIT);
        stdGradient[slot * AXES + axis] += transition[4] * ACTOR_RATE * clamp(logStdGradient, -POLICY_LIMIT, POLICY_LIMIT);
    }
    for (var gear = 0u; gear < GEARS; gear++) {
        outputGradient[slot * ACTIONS + AXES + gear] += transition[5] * ACTOR_RATE * clamp(samples[0].gearProbabilities[gear] * (gearObjectives[gear] - currentValue) / TEMPERATURE, -POLICY_LIMIT, POLICY_LIMIT);
    }
}`,
        sacCriticWeightGradient: constants + sampleType + `
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> features: array<f32>;
@group(0) @binding(2) var<storage, read> transition: array<f32>;
@group(0) @binding(3) var<storage, read> samples: array<PolicySample>;
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
