import { DrivingGpuModel, GpuPolicyPass } from './driving-gpu-model';
import type { PolicyModelState } from './driving-policy';

/** GPU boundary stub for UI/coordination tests in jsdom. Returns preset output
 * biases and a persistence sentinel; it implements no CNN, LSTM or learning math.
 * Actual numerical behavior is tested in scripts/check-driving-gpu.cjs. */
export function mockDrivingGpu() {
    return jest.spyOn(DrivingGpuModel, 'create').mockImplementation(async (config, state) => {
        const weights = {
            cameraWeights: state.cameraWeights,
            cameraProjectionWeights: state.cameraProjectionWeights,
            lstmWeights: state.lstmWeights,
            outputWeights: state.outputWeights,
        };
        const changed = () => { weights.lstmWeights[0][0] += 0.00001; };
        return {
            forward: jest.fn(async () => ({ pass: { released: false } as GpuPolicyPass,
                outputs: weights.outputWeights.map((row) => row[config.network.hiddenUnits]) })),
            learnTransition: jest.fn(() => { if (config.learning.actorRate || config.learning.criticRate) changed(); }),
            learnTargets: jest.fn(() => { if (config.learning.demonstrationRate) changed(); }),
            release: jest.fn((pass: GpuPolicyPass) => { pass.released = true; }),
            finishTraining: jest.fn(async () => undefined),
            weights: jest.fn(async (): Promise<Pick<PolicyModelState, 'cameraWeights' | 'cameraProjectionWeights' | 'lstmWeights' | 'outputWeights'>> =>
                JSON.parse(JSON.stringify(weights))),
            dispose: jest.fn(),
        } as unknown as DrivingGpuModel;
    });
}
