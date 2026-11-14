import { DrivingModelConfig } from './driving-model-config';

/** Learning targets enter from the CPU; predictions and output gradients stay on the GPU. */
export function drivingLearningShaders(config: DrivingModelConfig): Record<string, string> {
    const l = config.learning;
    const constants = `
const ACTIONS = ${config.actions.outputs.length}u;
const AXES = 2u;
const CRITIC = ${config.actions.planTimesSeconds.length * config.actions.outputs.length}u;
const EPSILON = ${l.actionEpsilon.toExponential()}f;
const DEMONSTRATION_RATE = ${l.demonstrationRate.toExponential()}f;
const DEMONSTRATION_LIMIT = ${l.demonstrationGradientLimit.toExponential()}f;
const ACTOR_RATE = ${l.actorRate.toExponential()}f;
const CRITIC_RATE = ${l.criticRate.toExponential()}f;
const ADVANTAGE_LIMIT = ${l.advantageLimit.toExponential()}f;
const POLICY_LIMIT = ${l.policyGradientLimit.toExponential()}f;
const VARIANCE = ${(l.explorationSigma ** 2).toExponential()}f;
const DISCOUNT = ${l.discount.toExponential()}f;
const DISCOUNT_INTERVAL = ${l.discountIntervalSeconds.toExponential()}f;
@group(0) @binding(0) var<storage, read> outputs: array<f32>;

fn shiftProbability(offset: u32, choice: u32, up: f32, down: f32) -> f32 {
    let allowed = array<bool, 3>(true, up != 0.0, down != 0.0);
    if (!allowed[choice]) { return 0.0; }
    var maximum = outputs[offset];
    for (var i = 1u; i < 3u; i++) {
        if (allowed[i]) { maximum = max(maximum, outputs[offset + i]); }
    }
    var total = 0.0;
    for (var i = 0u; i < 3u; i++) {
        if (allowed[i]) { total += exp(outputs[offset + i] - maximum); }
    }
    return exp(outputs[offset + choice] - maximum) / total;
}
`;
    return {
        demonstrationGradient: constants + `
// Per plan slot: present, throttle, brake, steering, shift choice, up allowed, down allowed.
@group(0) @binding(1) var<storage, read> targets: array<f32>;
@group(0) @binding(2) var<storage, read_write> gradients: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x > CRITIC) { return; }
    gradients[id.x] = 0.0;
    if (id.x == CRITIC) { return; }
    let slot = id.x / ACTIONS; let axis = id.x % ACTIONS; let base = slot * 7u;
    if (targets[base] == 0.0) { return; }
    if (axis < AXES) {
        let actionValue = select(targets[base + 3u], targets[base + 1u] - targets[base + 2u], axis == 0u);
        let p = clamp(actionValue, EPSILON - 1.0, 1.0 - EPSILON);
        let desired = 0.5 * log(max(EPSILON, 1.0 + p) / max(EPSILON, 1.0 - p));
        gradients[id.x] = DEMONSTRATION_RATE * clamp(desired - outputs[id.x], -DEMONSTRATION_LIMIT, DEMONSTRATION_LIMIT);
    } else {
        let choice = axis - AXES;
        let probability = shiftProbability(slot * ACTIONS + AXES, choice, targets[base + 5u], targets[base + 6u]);
        gradients[id.x] = DEMONSTRATION_RATE * (select(0.0, 1.0, choice == u32(targets[base + 4u])) - probability);
    }
}`,
        transitionGradient: constants + `
@group(0) @binding(1) var<storage, read> nextOutputs: array<f32>;
// Reward, elapsed seconds, terminal; then axis weight, latent axes, shift, mask and shift weight per slot.
@group(0) @binding(2) var<storage, read> transition: array<f32>;
@group(0) @binding(3) var<storage, read_write> gradients: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x > CRITIC) { return; }
    gradients[id.x] = 0.0;
    var bootstrap = 0.0;
    if (transition[2] == 0.0 && DISCOUNT > 0.0) {
        bootstrap = pow(DISCOUNT, transition[1] / DISCOUNT_INTERVAL) * nextOutputs[CRITIC];
    }
    let advantage = clamp(transition[0] + bootstrap - outputs[CRITIC],
        -ADVANTAGE_LIMIT, ADVANTAGE_LIMIT);
    if (id.x == CRITIC) { gradients[id.x] = CRITIC_RATE * advantage; return; }
    let slot = id.x / ACTIONS; let axis = id.x % ACTIONS; let base = 3u + slot * 7u;
    let weight = select(transition[base + 6u], transition[base], axis < AXES);
    if (weight == 0.0) { return; }
    if (axis < AXES) {
        gradients[id.x] = weight * ACTOR_RATE * advantage * clamp((transition[base + 1u + axis] - outputs[id.x]) / VARIANCE, -POLICY_LIMIT, POLICY_LIMIT);
    } else {
        let choice = axis - AXES;
        let probability = shiftProbability(slot * ACTIONS + AXES, choice, transition[base + 4u], transition[base + 5u]);
        gradients[id.x] = weight * ACTOR_RATE * advantage * (select(0.0, 1.0, choice == u32(transition[base + 3u])) - probability);
    }
}`,
    };
}
