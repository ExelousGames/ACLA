import { DrivingGpuModel, GpuPolicyPass, MAX_DRIVING_SCORE_ACTIONS } from './driving-gpu-model';
import { DrivingPolicy } from './driving-policy';
import { cloneConfig, drivingCameraLayers, drivingMemorySlots, drivingObservationInputs } from './driving-model-config';
import { drivingMemoryInputs, drivingMemorySlotInputs } from './driving-memory';
import { drivingDemonstrationTargetStride } from './driving-learning-shaders';

function deferred<T>() {
    let resolve!: (value: T) => void, reject!: (error: Error) => void;
    const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
    return { promise, resolve, reject };
}

const models: DrivingGpuModel[] = [];
const gpuDescriptor = Object.getOwnPropertyDescriptor(navigator, 'gpu');
afterEach(() => {
    models.splice(0).forEach((model) => model.dispose());
    if (gpuDescriptor) Object.defineProperty(navigator, 'gpu', gpuDescriptor);
    else Reflect.deleteProperty(navigator, 'gpu');
});

async function setup() {
    Object.assign(globalThis, { GPUBufferUsage: { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, STORAGE: 128 }, GPUMapMode: { READ: 1 } });
    const lost = deferred<Pick<GPUDeviceLostInfo, 'reason' | 'message'>>();
    const mapping = jest.fn(() => Promise.resolve());
    const compute = { setPipeline: jest.fn(), setBindGroup: jest.fn(), dispatchWorkgroups: jest.fn(), end: jest.fn() };
    const device = {
        lost: lost.promise, addEventListener: jest.fn(), destroy: jest.fn(),
        createBuffer: jest.fn(({ size, usage }: GPUBufferDescriptor) => ({ size, usage,
            destroy: jest.fn(), mapAsync: jest.fn(() => mapping()), unmap: jest.fn(),
            getMappedRange: jest.fn(() => new ArrayBuffer(size)) })),
        createShaderModule: jest.fn(),
        createComputePipelineAsync: jest.fn(async () => ({ getBindGroupLayout: jest.fn(() => ({})) })),
        createBindGroup: jest.fn((descriptor) => descriptor),
        createCommandEncoder: jest.fn(() => ({ beginComputePass: () => compute, clearBuffer: jest.fn(),
            copyBufferToBuffer: jest.fn(), finish: jest.fn(() => ({})) })),
        queue: { writeBuffer: jest.fn(), submit: jest.fn(), onSubmittedWorkDone: jest.fn(async () => undefined) },
    };
    Object.defineProperty(navigator, 'gpu', { configurable: true,
        value: { requestAdapter: async () => ({ requestDevice: async () => device }) } });
    const config = cloneConfig();
    config.network.hiddenUnits = 2;
    config.network.cameraLayers.forEach((layer) => { layer.channels = 1; });
    config.network.cameraProjectionUnits = 32;
    config.observation.features = ['camera', 'speed', 'centerline']; config.memory.features = ['camera', 'speed', 'centerline'];
    config.memory.windowMs = 400;
    const state = new DrivingPolicy({ game: 'acc', car: 'GT3', map: 'monza' }, drivingObservationInputs(config), () => 0.6, config).checkpoint().model;
    const model = await DrivingGpuModel.create(config, state);
    models.push(model);
    const input = (history = true, distinct = false) => {
        const bins = drivingMemorySlots(config), slot = drivingMemorySlotInputs(config);
        const features = new Array(drivingObservationInputs(config) + drivingMemoryInputs(config)).fill(0);
        const camera = { width: 384, height: 216, rgb: new Uint8Array(384 * 216 * 3) };
        const cameras = Array.from({ length: bins + 1 }, (_, i) => i < bins && !history ? null : distinct ? { ...camera } : camera);
        for (let i = 0; i < bins; i++) { features[i * slot] = Number(history); features[i * slot + 1] = (i + 1) / bins; }
        return { features, cameras };
    };
    const allocations = () => [device.createBuffer.mock.calls.length, device.createBindGroup.mock.calls.length];
    const buffers = () => device.createBuffer.mock.results.filter(({ type }) => type === 'return').map(({ value }) => value);
    return { model, device, config, input, allocations, buffers, mapping, lost, compute };
}

it('reads every convolution map with its real dimensions without running learning or writing model weights', async () => {
    const { model, device, config, input, buffers } = await setup();
    const camera = input().cameras[0]!;
    const weights = buffers();
    device.queue.writeBuffer.mockClear();
    device.createComputePipelineAsync.mockClear();
    const inspection = await model.inspectCamera(camera);
    expect(inspection.layers.map(({ values, ...shape }) => ({ ...shape, count: values.length }))).toEqual(
        drivingCameraLayers(config).map(({ width, height, channels }, index) => ({ index, width, height, channels, count: width * height * channels })));
    expect(device.queue.writeBuffer).toHaveBeenCalledTimes(1);
    expect(device.queue.writeBuffer.mock.calls[0][2]).toBe(camera.rgb);
    expect(device.queue.writeBuffer.mock.calls.some(([buffer]) => weights.includes(buffer))).toBe(false);
    expect(device.createComputePipelineAsync).not.toHaveBeenCalled();
    const bindings = device.createBindGroup.mock.calls.map(([descriptor]) => descriptor.entries.map((entry: GPUBindGroupEntry) => entry.binding));
    expect(bindings).toHaveLength(config.network.cameraLayers.length + 3);
    expect(bindings.slice(0, config.network.cameraLayers.length)).toEqual(config.network.cameraLayers.map(() => [0, 1, 2]));
});

it('preserves HWC map readback values and separates layers at their exact sizes', async () => {
    const { model, config, input, mapping, buffers } = await setup();
    mapping.mockImplementationOnce(async () => {
        const staging = buffers().find((buffer) => buffer.usage & GPUBufferUsage.MAP_READ)!;
        staging.getMappedRange.mockReturnValue(Float32Array.from({ length: staging.size / 4 }, (_, i) => i + 0.5).buffer);
    });
    const inspection = await model.inspectCamera(input().cameras[0]!);
    let offset = 0;
    drivingCameraLayers(config).forEach((layer, index) => {
        const count = layer.width * layer.height * layer.channels;
        expect(inspection.layers[index].values[0]).toBe(offset + 0.5);
        expect(inspection.layers[index].values[count - 1]).toBe(offset + count - 0.5);
        offset += count;
    });
});

it('keeps overlapping inspections exclusive and reuses their resources after readback', async () => {
    const { model, input, mapping, buffers, device, allocations } = await setup();
    const firstMap = deferred<void>(), secondMap = deferred<void>();
    mapping.mockImplementationOnce(() => firstMap.promise).mockImplementationOnce(() => secondMap.promise);
    device.queue.writeBuffer.mockClear();
    const first = model.inspectCamera(input().cameras[0]!), second = model.inspectCamera(input().cameras[0]!);
    const writes = device.queue.writeBuffer.mock.calls;
    expect(writes[0][0]).not.toBe(writes[1][0]);
    expect(buffers().filter((buffer) => buffer.usage & GPUBufferUsage.MAP_READ)).toHaveLength(2);
    secondMap.resolve(); await second;
    firstMap.resolve(); await first;
    // Retiring the excess readback drops cached bindings; warm them once again.
    await model.inspectCamera(input().cameras[0]!);
    const warmed = allocations();
    await model.inspectCamera(input().cameras[0]!);
    expect(allocations()).toEqual(warmed);
});

it('rejects incompatible images and retires resources after invalid activations', async () => {
    const { model, input, mapping, buffers } = await setup();
    const camera = input().cameras[0]!;
    await expect(model.inspectCamera({ ...camera, width: 1 })).rejects.toThrow(/RGB pixels/);
    await expect(model.inspectCamera({ ...camera, rgb: new Uint8Array(3) })).rejects.toThrow(/RGB pixels/);
    mapping.mockImplementationOnce(async () => {
        const staging = buffers().find((buffer) => buffer.usage & GPUBufferUsage.MAP_READ)!;
        const values = new Float32Array(staging.size / 4); values[0] = NaN;
        staging.getMappedRange.mockReturnValue(values.buffer);
    });
    await expect(model.inspectCamera(camera)).rejects.toThrow(/invalid camera activations/);
    model.dispose();
    expect(buffers().every((buffer) => buffer.destroy.mock.calls.length === 1)).toBe(true);
});

const targets = [{ step: 0, action: { pedal: -0.1, steering: -0.3, gear: 4 } }];
const transition = { reward: 1, elapsedSeconds: 0.2, terminal: false,
    nextGears: { max: 5 },
    actions: [{ step: 0, weight: 0.25, action: { pedal: -0.2, steering: -0.2, gear: -1 }, gear: -1, gears: { max: 10 } },
        { step: 2, weight: 0.75, action: { pedal: -0.4, steering: -0.4, gear: 0 }, gear: 0, gears: { max: 7 } }] };

it('queues predictions in episode order and does not remember a failed capture', async () => {
    const { model, input, mapping } = await setup();
    const episode = {};
    const frame = (capturedAt: number, historyTimes: [number, number] | [null, null]) => ({
        ...input(historyTimes[0] !== null), recurrent: { episode, capturedAt, historyTimes },
    });
    const firstMap = deferred<void>();
    mapping.mockImplementationOnce(() => firstMap.promise);
    const first = model.forward(frame(200, [null, null]));
    const second = model.forward(frame(450, [200, 50]));
    expect(mapping).toHaveBeenCalledTimes(1);
    firstMap.resolve();
    model.release((await first).pass);
    const next = (await second).pass;
    expect(next.steps).toBe(1);
    model.release(next);
    const duplicate = (await model.forward(frame(450, [200, 50]))).pass;
    expect(duplicate.steps).toBe(0);
    model.release(duplicate);

    mapping.mockRejectedValueOnce(new Error('Readback failed'));
    await expect(model.forward(frame(600, [450, 200]))).rejects.toThrow('Readback failed');
    const afterFailure = (await model.forward(frame(700, [600, 450]))).pass;
    expect(afterFailure.steps).toBe(2);
    model.release(afterFailure);
});

it('scores joint actions in one GPU batch using the active recurrent prediction and both critics', async () => {
    const { model, input, device, mapping, buffers, allocations, compute } = await setup();
    const { pass } = await model.forward(input());
    device.queue.writeBuffer.mockClear();
    device.queue.submit.mockClear();
    mapping.mockClear();
    compute.dispatchWorkgroups.mockClear();
    mapping.mockImplementationOnce(async () => {
        const staging = buffers().find((buffer) => (buffer.usage & GPUBufferUsage.MAP_READ)
            && buffer.size === 2 * 2 * 4)!;
        const values = new Float32Array(staging.size / 4);
        values.set([1, -2, 3, 4]);
        staging.getMappedRange.mockReturnValue(values.buffer);
    });
    const actions = [{ pedal: -0.5, steering: 0.25, gear: -1 }, { pedal: 0.75, steering: -0.25, gear: 4 }];
    await expect(model.scoreActions(pass, actions)).resolves.toEqual([[1, -2], [3, 4]]);
    expect(device.queue.writeBuffer).toHaveBeenCalledTimes(1);
    expect(device.queue.writeBuffer.mock.calls[0][2]).toEqual(new Float32Array([-0.5, 0.25, 0, 0.75, -0.25, 5]));
    expect(device.queue.submit).toHaveBeenCalledTimes(1);
    expect(mapping).toHaveBeenCalledTimes(1);
    expect(compute.dispatchWorkgroups.mock.calls).toEqual([[1, 2], [4, 1]]);
    const [projection, scoring] = device.createBindGroup.mock.calls.slice(-2).map(([descriptor]) => descriptor.entries);
    expect(projection[1].resource.buffer).toBe(pass.states);
    expect(projection[2].resource.buffer).toBe(pass.count);
    expect(scoring[1].resource.buffer).toBe(projection[3].resource.buffer);

    const warmed = allocations();
    await expect(model.scoreActions(pass, actions.slice(0, 1))).resolves.toEqual([[1, -2]]);
    expect(allocations()).toEqual(warmed);
    model.release(pass);
    await expect(model.scoreActions(pass, actions)).rejects.toThrow(/active prediction/);
});

it('grows directed scoring to the maximum grid and reads back only active scores', async () => {
    const { model, input, device, mapping, allocations, compute } = await setup();
    const { pass } = await model.forward(input());
    const action = { pedal: 1, steering: 1, gear: 1 };
    await model.scoreActions(pass, [action]);

    const actions = Array(MAX_DRIVING_SCORE_ACTIONS).fill(action);
    device.queue.submit.mockClear();
    mapping.mockClear();
    compute.dispatchWorkgroups.mockClear();
    await expect(model.scoreActions(pass, actions)).resolves.toHaveLength(MAX_DRIVING_SCORE_ACTIONS);
    expect(compute.dispatchWorkgroups.mock.calls).toEqual([[1, 2], [MAX_DRIVING_SCORE_ACTIONS * 2, 1]]);
    expect(device.queue.submit).toHaveBeenCalledTimes(1);
    expect(mapping).toHaveBeenCalledTimes(1);

    const warmed = allocations();
    await model.scoreActions(pass, actions.slice(0, 3));
    expect(allocations()).toEqual(warmed);
    const copy = device.createCommandEncoder.mock.results.slice(-1)[0].value.copyBufferToBuffer;
    expect(copy.mock.calls[0][4]).toBe(3 * 2 * 4);
    expect(copy.mock.calls[0][2].mapAsync).toHaveBeenLastCalledWith(GPUMapMode.READ, 0, 3 * 2 * 4);
    await expect(model.scoreActions(pass, [...actions, action])).rejects.toThrow(/Invalid driving action candidates/);
    model.release(pass);
});

it('uploads executed target gear numbers and current/next gear limits for sparse SAC credits', async () => {
    const { model, input, device } = await setup();
    const { pass } = await model.forward(input());
    device.queue.writeBuffer.mockClear();
    model.learnTransition(pass, input(), { ...transition, terminal: true });
    const data = device.queue.writeBuffer.mock.calls.map(([, , values]) => values)
        .filter((values): values is Float32Array => values instanceof Float32Array && values.length === 16);
    expect(data).toHaveLength(2);
    expect(Array.from(data[0].slice(0, 12))).toEqual(Array.from(new Float32Array(
        [1, 0.2, 1, 0, 0.25, 0.25, 0.25, 12, 7, -0.2, -0.2, 0])));
    expect(Array.from(data[1].slice(0, 12))).toEqual(Array.from(new Float32Array(
        [1, 0.2, 1, 0, 0.75, 0.75, 0.75, 12, 7, -0.4, -0.4, 1])));
    expect(data.every((values) => Array.from(values.slice(12)).every(Number.isFinite))).toBe(true);
    model.release(pass);
});

it('uploads reverse and neutral as distinct recorded gear classes', async () => {
    const { model, input, device, config } = await setup();
    device.queue.writeBuffer.mockClear();
    model.learnTargets(input(), [
        { step: 0, action: { pedal: 0, steering: 0, gear: -1 } },
        { step: 1, action: { pedal: 0, steering: 0, gear: 0 } },
    ], false, 'recorded');
    const stride = drivingDemonstrationTargetStride;
    const target = device.queue.writeBuffer.mock.calls.map(([, , values]) => values).find((values) =>
        values instanceof Float32Array && values.length === config.actions.planTimesSeconds.length * stride);
    expect(Array.from(target.slice(0, 2 * stride))).toEqual([1, 0, 0, 0, 1, 0.5, 0, 0, 1, 1]);
});

it('retains distinct interpolation actions instead of replacing them with a plan endpoint', async () => {
    const { model, input, device } = await setup();
    const { pass } = await model.forward(input());
    const analog = { step: 1, weight: 0.25, gearWeight: 0, action: { pedal: -0.4, steering: -0.4, gear: 4 },
        gear: 4, gears: { max: 4 } };
    device.queue.writeBuffer.mockClear();
    model.learnTransition(pass, input(), { ...transition, terminal: true, actions: [analog,
        { ...analog, action: { pedal: 0.8, steering: 0.2, gear: 4 }, weight: 0.5, gearWeight: 0.5, gears: { max: 7 } }, analog] });
    const data = device.queue.writeBuffer.mock.calls.map(([, , values]) => values)
        .filter((values): values is Float32Array => values instanceof Float32Array && values.length === 16);
    expect(data).toHaveLength(3);
    expect(data.map((values) => values[3])).toEqual([0, 0, 0]);
    expect(data.map((values) => values[5])).toEqual([0, 0.5, 0]);
    expect(data.map((values) => Array.from(values.slice(7, 9)))).toEqual([[9, 7], [9, 7], [9, 7]]);
    expect(data.map((values) => values[11])).toEqual([5, 5, 5]);
    expect(Array.from(data[1].slice(9, 12))).toEqual(Array.from(new Float32Array([0.8, 0.2, 5])));
    expect(Array.from(data[0].slice(9, 12))).toEqual(Array.from(data[2].slice(9, 12)));
    model.release(pass);
});

it('reuses buffers and bind groups across predictions, training, history lengths and camera sharing', async () => {
    const { model, input, allocations, device } = await setup();
    const tick = async () => {
        const { pass } = await model.forward(input(true, true));
        model.learnTransition(pass, input(true, true), transition);
        model.learnTransition(pass, input(false), { ...transition, terminal: true });
        model.learnTargets(input(true, true), targets);
        model.learnTargets(input(false), targets);
        model.learnTargets(input(), targets);
        model.release(pass);
        await model.weights();
    };
    await tick(); await tick(); // Both pass workspaces have now served both roles.
    const warmed = allocations();
    await tick(); await tick();
    expect(allocations()).toEqual(warmed);
    expect(device.queue.onSubmittedWorkDone).not.toHaveBeenCalled();
});

it('keeps retained decisions exclusive and ignores stale duplicate releases', async () => {
    const { model, input, device } = await setup();
    const first = (await model.forward(input())).pass;
    device.queue.writeBuffer.mockClear();
    model.learnTargets(input(true, true), targets);
    const second = (await model.forward(input())).pass;
    const retained = [first.features, first.states, first.count, first.outputs,
        ...first.cameras.flatMap(({ offset, activations }) => [offset, ...activations])];
    expect(device.queue.writeBuffer.mock.calls.some(([buffer]) => retained.includes(buffer))).toBe(false);
    expect(second.features).not.toBe(first.features);
    model.release(first);
    const third = (await model.forward(input())).pass;
    expect(third.features).toBe(first.features);
    expect(third).not.toBe(first);
    model.release(first);
    const fourth = (await model.forward(input())).pass;
    expect(fourth.features).not.toBe(third.features);
    expect(fourth.features).not.toBe(second.features);
    expect(() => model.learnTransition(first, input(), transition)).toThrow(/already released/);
    [second, third, fourth].forEach((pass) => model.release(pass));
});

it('holds each staging buffer until its readback is copied and unmapped', async () => {
    const { model, input, mapping, buffers } = await setup();
    const firstMap = deferred<void>(), secondMap = deferred<void>();
    mapping.mockImplementationOnce(() => firstMap.promise).mockImplementationOnce(() => secondMap.promise);
    const first = model.forward(input()), second = model.forward(input());
    const staging = buffers().filter((buffer) => buffer.usage & GPUBufferUsage.MAP_READ);
    expect(staging).toHaveLength(2);
    expect(staging.every((buffer) => buffer.unmap.mock.calls.length === 0)).toBe(true);
    secondMap.resolve(); model.release((await second).pass);
    const third = await model.forward(input());
    expect(staging[0].mapAsync).toHaveBeenCalledTimes(1);
    expect(staging[1].mapAsync).toHaveBeenCalledTimes(2);
    model.release(third.pass);
    firstMap.resolve(); model.release((await first).pass);
    expect(staging[0].destroy).toHaveBeenCalledTimes(1); // Only one idle readback per size.
    expect(staging[1].destroy).not.toHaveBeenCalled();
});

it('retires a failed readback and recovers without leaking the forward lease', async () => {
    const { model, input, mapping, buffers } = await setup();
    mapping.mockRejectedValueOnce(new Error('mapping failed'));
    await expect(model.forward(input())).rejects.toThrow('mapping failed');
    const failed = buffers().find((buffer) => buffer.usage & GPUBufferUsage.MAP_READ)!;
    expect(failed.destroy).toHaveBeenCalledTimes(1);
    const { pass } = await model.forward(input()); model.release(pass);
    expect(failed.mapAsync).toHaveBeenCalledTimes(1);
    model.dispose();
    expect(buffers().every((buffer) => buffer.destroy.mock.calls.length === 1)).toBe(true);
});

it('retires partial workspaces when allocation fails', async () => {
    const { model, input, device, buffers } = await setup();
    const create = device.createBuffer.getMockImplementation()!;
    device.createBuffer.mockImplementationOnce(create).mockImplementationOnce(() => { throw new Error('allocation failed'); });
    await expect(model.forward(input())).rejects.toThrow('allocation failed');
    const partial = buffers().slice(-1)[0];
    expect(partial.destroy).toHaveBeenCalledTimes(1);
    const { pass } = await model.forward(input()); model.release(pass);
    model.dispose();
    expect(buffers().every((buffer) => buffer.destroy.mock.calls.length === 1)).toBe(true);
});

it('bounds idle forward workspaces after a burst of retained decisions', async () => {
    const { model, input, buffers, allocations } = await setup();
    const passes: GpuPolicyPass[] = [];
    for (let i = 0; i < 5; i++) passes.push((await model.forward(input())).pass);
    passes.forEach((pass) => model.release(pass));
    expect(buffers().filter((buffer) => !buffer.destroy.mock.calls.length && passes.some((pass) => pass.features === buffer))).toHaveLength(3);
    const before = allocations()[0];
    for (let i = 0; i < 3; i++) model.release((await model.forward(input())).pass);
    expect(allocations()[0]).toBe(before);
    model.dispose();
    expect(buffers().every((buffer) => buffer.destroy.mock.calls.length === 1)).toBe(true);
});

it.each(['dispose', 'device loss'])('does not recycle resources after %s during a pending map', async (failure) => {
    const { model, input, mapping, lost, device, buffers } = await setup();
    const retained = (await model.forward(input())).pass;
    model.learnTargets(input(), targets);
    const pending = deferred<void>();
    mapping.mockImplementationOnce(() => pending.promise);
    const prediction = model.forward(input());
    if (failure === 'dispose') model.dispose();
    else { lost.resolve({ reason: 'destroyed', message: 'test loss' }); await Promise.resolve(); }
    pending.resolve();
    await expect(prediction).rejects.toThrow(failure === 'dispose' ? /released/ : /GPU lost/);
    model.release(retained);
    await expect(model.forward(input())).rejects.toThrow();
    model.dispose(); model.dispose(); model.release(retained);
    expect(device.destroy).toHaveBeenCalledTimes(1);
    expect(buffers().every((buffer) => buffer.destroy.mock.calls.length === 1)).toBe(true);
});
