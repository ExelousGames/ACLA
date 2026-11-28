import { DrivingGpuModel, GpuPolicyPass } from './driving-gpu-model';
import type { PolicyWeights } from './driving-policy';
import { drivingCameraLayers } from './driving-model-config';

/** GPU boundary stub for UI/coordination tests in jsdom. Returns preset output
 * biases and a persistence sentinel; it implements no CNN, GRU or learning math.
 * Actual numerical behavior is tested in scripts/check-driving-gpu.cjs. */
export function mockDrivingGpu() {
    return jest.spyOn(DrivingGpuModel, 'create').mockImplementation(async (config, state) => {
        const weights = {
            cameraWeights: state.cameraWeights,
            cameraProjectionWeights: state.cameraProjectionWeights,
            ...(state.cameraNormWeights && { cameraNormWeights: state.cameraNormWeights }),
            ...(state.gruNormWeights && { gruNormWeights: state.gruNormWeights }),
            gruWeights: state.gruWeights,
            outputWeights: state.outputWeights,
            sac: state.sac,
        };
        let pendingChanges: number | undefined;
        const changed = () => {
            if (pendingChanges !== undefined) pendingChanges++;
            else weights.gruWeights[0][0] += 0.00001;
        };
        return {
            inspectCamera: jest.fn(async () => ({ layers: drivingCameraLayers(config).map(({ width, height, channels }, index) =>
                ({ index, width, height, channels, values: new Float32Array(width * height * channels) })) })),
            forward: jest.fn(async () => ({ pass: { released: false } as GpuPolicyPass,
                outputs: weights.outputWeights.map((row) => row[config.network.hiddenUnits]), logStds: weights.sac.logStds.slice() })),
            scoreActions: jest.fn(async (_pass: GpuPolicyPass, actions: readonly unknown[]) => actions.map((): [number, number] => [0, 0])),
            learnTransition: jest.fn(() => { if (config.learning.actorRate || config.learning.criticRate) changed(); }),
            learnTargets: jest.fn(() => { if (config.learning.demonstrationRate) changed(); }),
            observeDemonstration: jest.fn(),
            replay: jest.fn(() => ({ released: false } as GpuPolicyPass)),
            beginSequence: jest.fn(() => { pendingChanges = 0; }),
            beginRecordedSequence: jest.fn(() => { pendingChanges = 0; }),
            endSequence: jest.fn((commit: boolean) => {
                if (commit) weights.gruWeights[0][0] += (pendingChanges ?? 0) * 0.00001;
                pendingChanges = undefined;
            }),
            release: jest.fn((pass: GpuPolicyPass) => { pass.released = true; }),
            finishTraining: jest.fn(async () => undefined),
            trainingDiagnostics: jest.fn(async () => undefined),
            weights: jest.fn(async (): Promise<PolicyWeights> =>
                JSON.parse(JSON.stringify(weights))),
            dispose: jest.fn(),
        } as unknown as DrivingGpuModel;
    });
}
