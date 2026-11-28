import { DRIVING_ANALOG_AXES, DrivingModelConfig } from './driving-model-config';
import { drivingLearningRate } from './driving-update-shaders';

export const drivingDemonstrationTargetStride = DRIVING_ANALOG_AXES + 3;

/** Learning targets enter from the CPU; predictions and output gradients stay on the GPU. */
export function drivingLearningShaders(config: DrivingModelConfig): Record<string, string> {
    const l = config.learning;
    // Preserve relative actor/critic rates while clipping before the common base rate.
    const rate = drivingLearningRate(config) || 1;
    const constants = `
const ACTIONS = ${config.actions.outputs.length}u;
const AXES = ${DRIVING_ANALOG_AXES}u;
const GEARS = ACTIONS - AXES;
const CRITIC = ${config.actions.planTimesSeconds.length * config.actions.outputs.length}u;
const EPSILON = ${l.actionEpsilon.toExponential()}f;
const DEMONSTRATION_RATE = ${(l.demonstrationRate / rate).toExponential()}f;
const DEMONSTRATION_LIMIT = ${l.demonstrationGradientLimit.toExponential()}f;
const GEAR_TEMPERATURE = ${l.shiftTemperature.toExponential()}f;
@group(0) @binding(0) var<storage, read> outputs: array<f32>;

fn gearProbability(offset: u32, choice: u32) -> f32 {
    var maximum = outputs[offset];
    for (var i = 1u; i < GEARS; i++) {
        maximum = max(maximum, outputs[offset + i]);
    }
    var total = 0.0;
    for (var i = 0u; i < GEARS; i++) {
        total += exp((outputs[offset + i] - maximum) / GEAR_TEMPERATURE);
    }
    return exp((outputs[offset + choice] - maximum) / GEAR_TEMPERATURE) / total;
}
`;
    const gradient = (recorded: boolean) => constants + `
// Per plan slot: weight, signed pedal, steering, gear index, gear weight.
@group(0) @binding(1) var<storage, read> targets: array<f32>;
@group(0) @binding(2) var<storage, read_write> gradients: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x > CRITIC) { return; }
    gradients[id.x] = 0.0;
    if (id.x == CRITIC) { return; }
    let slot = id.x / ACTIONS; let axis = id.x % ACTIONS;
    let base = slot * ${drivingDemonstrationTargetStride}u;
    let weight = targets[base];
    if (weight == 0.0) { return; }
    if (axis < AXES) {
        let actionValue = targets[base + 1u + axis];
        ${recorded ? `// Negative derivative of softplus(2 * logit) - (1 + target) * logit.
        // It still corrects a prediction saturated in the wrong direction.
        let predicted = tanh(clamp(outputs[id.x], -10.0, 10.0));
        let residual = actionValue - predicted;` : `
        let p = clamp(actionValue, EPSILON - 1.0, 1.0 - EPSILON);
        let desired = 0.5 * log(max(EPSILON, 1.0 + p) / max(EPSILON, 1.0 - p));
        let residual = desired - outputs[id.x];`}
        gradients[id.x] = DEMONSTRATION_RATE * clamp(residual, -DEMONSTRATION_LIMIT, DEMONSTRATION_LIMIT) * weight;
    } else {
        let choice = axis - AXES;
        let expected = u32(targets[base + AXES + 1u]);
        let gearWeight = targets[base + AXES + 2u];
        if (gearWeight == 0.0) { return; }
        let probability = gearProbability(slot * ACTIONS + AXES, choice);
        let score = (select(0.0, 1.0, choice == expected) - probability) / GEAR_TEMPERATURE;
        gradients[id.x] = DEMONSTRATION_RATE * clamp(score, -DEMONSTRATION_LIMIT, DEMONSTRATION_LIMIT) * weight * gearWeight;
    }
}`;
    return { demonstrationGradient: gradient(false), recordedDemonstrationGradient: gradient(true) };
}
