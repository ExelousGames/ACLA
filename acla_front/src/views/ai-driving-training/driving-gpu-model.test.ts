import { DrivingGpuModel, GpuPolicyPass } from './driving-gpu-model';
import { DrivingPolicy } from './driving-policy';
import { cloneConfig, drivingMemorySlots, drivingObservationInputs } from './driving-model-config';
import { drivingMemoryInputs, drivingMemorySlotInputs } from './driving-memory';

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
    config.observation.features = ['camera', 'speed']; config.memory.features = ['camera', 'speed'];
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
    return { model, device, input, allocations, buffers, mapping, lost };
}

const targets = [{ step: 0, action: { throttle: 0.7, brake: 0.1, steering: -0.3 }, shifts: { up: true, down: false } }];
const transition = { reward: 1, elapsedSeconds: 0.2, terminal: false,
    actions: [{ step: 0, weight: 0.25, latent: [0.1, -0.2], shift: 0, shifts: { up: true, down: true } },
        { step: 2, weight: 0.75, latent: [0.3, -0.4], shift: 1, shifts: { up: true, down: false } }] };

it('uploads sparse execution credits at their plan slots and clears unexecuted slots', async () => {
    const { model, input, device } = await setup();
    const { pass } = await model.forward(input());
    device.queue.writeBuffer.mockClear();
    model.learnTransition(pass, input(), { ...transition, terminal: true });
    const data = device.queue.writeBuffer.mock.calls[0][2] as Float32Array;
    expect(Array.from(data.slice(0, 3))).toEqual(Array.from(new Float32Array([1, 0.2, 1])));
    expect(Array.from(data.slice(3, 9))).toEqual(Array.from(new Float32Array([0.25, 0.1, -0.2, 0, 1, 1])));
    expect(Array.from(data.slice(9, 15))).toEqual([0, 0, 0, 0, 0, 0]);
    expect(Array.from(data.slice(15, 21))).toEqual(Array.from(new Float32Array([0.75, 0.3, -0.4, 1, 1, 0])));
    expect(Array.from(data.slice(21)).every((value) => value === 0)).toBe(true);
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
    expect(buffers().filter((buffer) => !buffer.destroy.mock.calls.length && passes.some((pass) => pass.features === buffer))).toHaveLength(2);
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
