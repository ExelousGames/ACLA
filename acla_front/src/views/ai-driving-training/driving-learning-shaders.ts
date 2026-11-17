import { DrivingModelConfig } from './driving-model-config';
import { drivingLearningRate } from './driving-update-shaders';

/** Learning targets enter from the CPU; predictions and output gradients stay on the GPU. */
export function drivingLearningShaders(config: DrivingModelConfig): Record<string, string> {
    const l = config.learning;
    // Preserve relative actor/critic rates while clipping before the common base rate.
    const rate = drivingLearningRate(config) || 1;
    const constants = `
const ACTIONS = ${config.actions.outputs.length}u;
const AXES = 3u;
const CRITIC = ${config.actions.planTimesSeconds.length * config.actions.outputs.length}u;
const EPSILON = ${l.actionEpsilon.toExponential()}f;
const DEMONSTRATION_RATE = ${(l.demonstrationRate / rate).toExponential()}f;
const DEMONSTRATION_LIMIT = ${l.demonstrationGradientLimit.toExponential()}f;
const SHIFT_TEMPERATURE = ${l.shiftTemperature.toExponential()}f;
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
        if (allowed[i]) { total += exp((outputs[offset + i] - maximum) / SHIFT_TEMPERATURE); }
    }
    return exp((outputs[offset + choice] - maximum) / SHIFT_TEMPERATURE) / total;
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
        let actionValue = targets[base + 1u + axis];
        let p = clamp(actionValue, EPSILON - 1.0, 1.0 - EPSILON);
        let desired = 0.5 * log(max(EPSILON, 1.0 + p) / max(EPSILON, 1.0 - p));
        gradients[id.x] = DEMONSTRATION_RATE * clamp(desired - outputs[id.x], -DEMONSTRATION_LIMIT, DEMONSTRATION_LIMIT);
    } else {
        let choice = axis - AXES;
        let probability = shiftProbability(slot * ACTIONS + AXES, choice, targets[base + 5u], targets[base + 6u]);
        let score = (select(0.0, 1.0, choice == u32(targets[base + 4u])) - probability) / SHIFT_TEMPERATURE;
        gradients[id.x] = DEMONSTRATION_RATE * clamp(score, -DEMONSTRATION_LIMIT, DEMONSTRATION_LIMIT);
    }
}`,
    };
}
