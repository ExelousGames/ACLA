// Run: npx electron scripts/check-driving-gpu.cjs
// Add --soak for 1,000 training cycles each on default and saturated long-history models.
// Real hardware WGSL gradients, training, checkpoint and latency checks. No simulator controls.
const { app, BrowserWindow } = require('electron');
const fs = require('fs');
const path = require('path');
const http = require('http');
const ts = require('typescript');
const root = path.resolve(__dirname, '../src');
const modules = new Map();
function bundleModule(id) {
    if (modules.has(id)) return;
    modules.set(id, '');
    const filename = ['.ts', '.tsx', '.json', '.js'].map(extension => path.join(root, id + extension)).find(fs.existsSync);
    if (!filename) throw new Error(`Cannot bundle ${id}`);
    let code = filename.endsWith('.json') ? `module.exports = ${fs.readFileSync(filename, 'utf8')};`
        : ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
            compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.CommonJS, esModuleInterop: true },
        }).outputText;
    code = code.replace(/require\("([^"]+)"\)/g, (_, name) => {
        const resolved = name.startsWith('.') ? path.posix.normalize(path.posix.join(path.posix.dirname(id), name)) : name;
        const dependency = resolved.replace(/\.json$/, '');
        bundleModule(dependency);
        return `require(${JSON.stringify(dependency)})`;
    });
    modules.set(id, code);
}
bundleModule('views/ai-driving-training/driving-policy');
bundleModule('views/ai-driving-training/driving-memory');
const bundle = `const modules = {${Array.from(modules, ([id, code]) => `${JSON.stringify(id)}: function(require, module, exports) {\n${code}\n}`).join(',')}}, cache = {};
function require(id) { if (!cache[id]) { const module = cache[id] = { exports: {} }; modules[id](require, module, module.exports); } return cache[id].exports; }`;
const server = http.createServer((_req, res) => res.end('<!doctype html><title>Driving GPU checks</title>'));

async function checks(soakIterations) {
    const { DrivingPolicy } = require('views/ai-driving-training/driving-policy');
    const { cloneConfig, drivingCameraInputs, drivingCameraLayers, drivingCameraPooledInputs, drivingMemorySlots, drivingObservationInputs, drivingPolicyInputs } = require('views/ai-driving-training/driving-model-config');
    const { drivingMemoryInputs, drivingMemorySlotInputs } = require('views/ai-driving-training/driving-memory');
    const assert = (condition, label) => { if (!condition) throw new Error(label); };
    const errors = [];
    let readbacks = 0;
    let bufferAllocations = 0, bindGroupAllocations = 0;
    const createBuffer = GPUDevice.prototype.createBuffer, createBindGroup = GPUDevice.prototype.createBindGroup;
    GPUDevice.prototype.createBuffer = function (...args) { bufferAllocations++; return createBuffer.apply(this, args); };
    GPUDevice.prototype.createBindGroup = function (...args) { bindGroupAllocations++; return createBindGroup.apply(this, args); };
    const allocations = () => [bufferAllocations, bindGroupAllocations];
    const mapAsync = GPUBuffer.prototype.mapAsync;
    GPUBuffer.prototype.mapAsync = function (...args) { readbacks++; return mapAsync.apply(this, args); };
    const withoutReadbacks = async (label, train) => {
        const before = readbacks;
        await train();
        assert(readbacks === before, `${label}: learning must not read GPU buffers back to the CPU`);
    };
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    assert(adapter && !adapter.info.isFallbackAdapter, 'A hardware GPU is required');
    const close = (a, b, label, tolerance = 0.00003) => {
        assert(a.length === b.length, `${label}: shape`);
        let maximum = 0;
        for (let i = 0; i < a.length; i++) maximum = Math.max(maximum, Math.abs(a[i] - b[i]));
        assert(Number.isFinite(maximum) && maximum < tolerance, `${label}: max error ${maximum}`);
        return maximum;
    };
    const compare = (a, b, label) => {
        assert(a.model.steps === b.model.steps, `${label}: steps`);
        for (const key of ['cameraWeights', 'cameraProjectionWeights', 'lstmWeights', 'outputWeights']) {
            close(a.model[key].flat(2), b.model[key].flat(2), `${label}/${key}`);
        }
    };
    const changed = (a, b) => a.flat(2).some((value, i) => value !== b.flat(2)[i]);
    const finite = (checkpoint) => {
        for (const key of ['cameraWeights', 'cameraProjectionWeights', 'lstmWeights', 'outputWeights']) {
            assert(checkpoint.model[key].flat(2).every(Number.isFinite), `Finite ${key}`);
        }
    };
    const predict = async (policy, input, shifts) => {
        const decision = await policy.decideAsync(input, false, shifts);
        policy.releaseDecision(decision);
        return decision;
    };
    const assertTrained = (before, after, label) => {
        after.model.cameraWeights.forEach((rows, l) => assert(changed(rows, before.model.cameraWeights[l]), `${label}: CNN layer ${l}`));
        assert(changed(before.model.cameraProjectionWeights, after.model.cameraProjectionWeights), `${label}: spatial projection`);
        const h = after.configuration.network.hiddenUnits;
        for (let gate = 0; gate < 4; gate++) {
            assert(changed(before.model.lstmWeights.slice(gate * h, (gate + 1) * h),
                after.model.lstmWeights.slice(gate * h, (gate + 1) * h)), `${label}: LSTM gate ${gate}`);
        }
        finite(after);
    };
    const create = async (config, checkpoint) => {
        let seed = 42;
        const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
        const policy = checkpoint ? DrivingPolicy.restore(checkpoint)
            : new DrivingPolicy({ game: 'acc', car: 'GT3', map: 'monza' }, drivingObservationInputs(config), random, config);
        await policy.initializeGpu();
        policy.gpu.device.addEventListener('uncapturederror', event => errors.push(event.error.message));
        return policy;
    };
    const makeInput = (config, frame = 0, history = true, distinct = false) => {
        const bins = drivingMemorySlots(config), slots = drivingMemorySlotInputs(config);
        const features = Array.from({ length: drivingObservationInputs(config) + drivingMemoryInputs(config) }, (_, i) => Math.sin(i * 0.7) * 0.2);
        const rgb = Uint8Array.from({ length: 384 * 216 * 3 }, (_, i) => (i * 17 + frame * 31) % 256);
        const camera = { width: 384, height: 216, rgb };
        const cameras = Array.from({ length: bins + 1 }, (_, slot) =>
            slot < bins && (!history || !config.memory.features.includes('camera')) ? null : camera);
        if (distinct) cameras.forEach((image, slot) => {
            if (image && slot !== bins) cameras[slot] = { ...camera, rgb: Uint8Array.from(rgb, (value, i) => (value + slot * 19 + i % 7) % 256) };
        });
        for (let i = 0; i < bins; i++) { features[i * slots] = history ? 1 : 0; features[i * slots + 1] = (i + 1) / bins; }
        return { features, cameras };
    };
    const shifts = { up: true, down: false }, action = { throttle: 0.7, brake: 0.12, steering: -0.3, shiftUp: true };
    const credits = (decision, steps = decision.plan.map((_, step) => step)) => steps.map(step =>
        ({ step, weight: 1 / steps.length, shift: decision.plan[step].shift, shifts: decision.plan[step].shifts }));

    // Exercise the production forward/backward kernels against independent CPU
    // expectations. Finite gate sums can exceed the GPU tanh overflow threshold;
    // cell states can also exceed it with the supported 101-observation history.
    const activationConfig = cloneConfig();
    activationConfig.network.hiddenUnits = 1;
    activationConfig.network.cameraLayers.forEach(layer => { layer.channels = 1; });
    activationConfig.network.cameraProjectionUnits = 1;
    const activationPolicy = await create(activationConfig), activationGpu = activationPolicy.gpu;
    const activationInputs = drivingPolicyInputs(activationConfig), columns = activationInputs + 2;
    const gateWeights = new Array(4 * columns).fill(0);
    for (let gate = 0; gate < 4; gate++) gateWeights[gate * columns] = 1;
    const weights = activationGpu.upload(gateWeights), features = activationGpu.buffer(activationInputs * 4);
    const states = activationGpu.buffer(12 * 4), upstream = activationGpu.buffer(2 * 4), deltas = activationGpu.buffer(4 * 4);
    const activationValues = [-1e30, -1000, -100, -89, -88, -50, -10, -1, -0.001, 0, 0.001, 1, 10, 50, 88, 89, 100, 1000, 1e30];
    let activationCases = 0;
    for (const value of activationValues) {
        for (const previousCell of [-100, 0, 100]) {
            const initial = new Float32Array(12); initial[1] = previousCell;
            activationGpu.device.queue.writeBuffer(features, 0, new Float32Array([value]));
            activationGpu.device.queue.writeBuffer(states, 0, initial);
            activationGpu.device.queue.writeBuffer(upstream, 0, new Float32Array([0.125, 0.25]));
            const commands = activationGpu.device.createCommandEncoder();
            activationGpu.dispatch(commands, 'forward', [weights, features, states, activationGpu.times[0]], 1);
            activationGpu.dispatch(commands, 'gateGradient', [states, upstream, deltas, activationGpu.times[0]], 1);
            activationGpu.device.queue.submit([commands.finish()]);
            const actual = await activationGpu.download([states, deltas]);
            const x = Math.fround(value), sigmoid = 1 / (1 + Math.exp(-x)), candidate = Math.tanh(x);
            const cell = sigmoid * previousCell + sigmoid * candidate, activatedCell = Math.tanh(cell);
            close(Array.from(actual.slice(6, 12)), [sigmoid * activatedCell, cell, sigmoid, sigmoid, candidate, sigmoid],
                `Finite LSTM activations at input ${value}, cell ${previousCell}`);
            const dc = 0.25 + 0.125 * sigmoid * (1 - activatedCell * activatedCell);
            close(Array.from(actual.slice(12)), [dc * candidate * sigmoid * (1 - sigmoid),
                dc * previousCell * sigmoid * (1 - sigmoid), dc * sigmoid * (1 - candidate * candidate),
                0.125 * activatedCell * sigmoid * (1 - sigmoid)], `Finite LSTM gradients at input ${value}, cell ${previousCell}`);
            activationCases++;
        }
    }
    activationPolicy.dispose();
    const results = [];
    for (const layout of ['default', 'custom', 'boundary']) {
        const config = cloneConfig();
        if (layout !== 'default') {
            config.network.cameraLayers.forEach(layer => { layer.channels = 3; });
            config.observation.features = ['camera', 'speed', 'controls'];
            config.memory.features = ['controls', 'speed'];
            config.memory.windowMs = 400;
            config.network.hiddenUnits = 7;
            config.actions.planTimesSeconds = [0, 0.5];
            config.learning.discount = 0.8;
            config.learning.discountIntervalSeconds = 0.3;
            config.learning.advantageLimit = 0.4;
            config.learning.policyGradientLimit = 0.7;
            config.learning.demonstrationGradientLimit = 0.5;
            config.learning.actorRate = 0.002;
            config.learning.criticRate = 0.003;
            config.learning.demonstrationRate = 0.004;
        }
        if (layout === 'boundary') {
            config.actions.planTimesSeconds = Array.from({ length: 32 }, (_, i) => i * 0.5);
            config.learning.actionEpsilon = 1e-9;
            config.learning.discount = 0;
        }
        const gpu = await create(config), input = makeInput(config), next = makeInput(config, 1);
        const initial = await gpu.checkpointAsync();
        gpu.gpu.device.pushErrorScope('validation');
        for (const retired of ['decide', 'learn', 'learnDemonstration', 'learnPlannedActions', 'lstm', 'cameraEncoder']) {
            assert(!(retired in gpu), `Retired CPU implementation must be absent: ${retired}`);
        }
        for (const sample of [makeInput(config, 3, false), input]) {
            const a = await predict(gpu, sample, shifts), b = await predict(gpu, sample, shifts);
            close([a.value, ...a.plan.flatMap(step => step.means)], [b.value, ...b.plan.flatMap(step => step.means)], 'deterministic prediction');
            assert(a.plan[0].shiftProbabilities[2] === 0, 'Unavailable shifts stay masked');
        }
        await withoutReadbacks('demonstration', () => gpu.learnDemonstrationAsync(input, action, shifts));
        const demonstration = await gpu.checkpointAsync();
        assertTrained(initial, demonstration, 'demonstration');
        close(initial.model.outputWeights.slice(5).flat(), demonstration.model.outputWeights.slice(5).flat(), 'Only the demonstrated slot learns');
        await withoutReadbacks('distinct historical images', () => gpu.learnDemonstrationAsync(makeInput(config, 5, true, true), action, shifts));
        gpu.random = () => 0.99;
        for (const [reward, terminal, available] of [[0.8, false, shifts], [-20, false, { up: false, down: true }],
            [20, true, { up: false, down: false }], [-0.6, true, { up: true, down: true }]]) {
            const before = await gpu.checkpointAsync(), decision = await gpu.decideAsync(input, true, available);
            const following = terminal ? { features: [], cameras: [] } : next;
            const executed = [0, decision.plan.length - 1];
            await withoutReadbacks('actor-critic', () => gpu.learnAsync(decision, reward, following, 0.35, credits(decision, executed), terminal));
            gpu.releaseDecision(decision);
            const after = await gpu.checkpointAsync();
            assertTrained(before, after, 'actor-critic');
            for (let step = 0; step < decision.plan.length; step++) {
                const from = before.model.outputWeights.slice(step * 5, (step + 1) * 5);
                const to = after.model.outputWeights.slice(step * 5, (step + 1) * 5);
                if (executed.includes(step)) assert(changed(from, to), `Executed plan slot ${step} receives reward`);
                else close(from.flat(), to.flat(), `Unexecuted slot ${step} receives no reward`);
            }
            if (!available.up && !available.down) close(before.model.outputWeights.slice(2, 5).flat(), after.model.outputWeights.slice(2, 5).flat(), 'Forced hold receives no shift gradient');
        }
        for (const [pedal, steering, available] of [[0, -1, { up: false, down: false }], [1, 1, { up: false, down: true }]]) {
            await withoutReadbacks('boundary demonstration', () => gpu.learnDemonstrationAsync(input,
                { throttle: pedal, brake: 1 - pedal, steering, shiftDown: available.down }, available));
            finite(await gpu.checkpointAsync());
        }
        const targets = [{ step: 1, action, shifts }], finalSlot = config.actions.planTimesSeconds.length - 1;
        const beforePlan = await gpu.checkpointAsync();
        await withoutReadbacks('delayed targets', () => gpu.learnPlannedActionsAsync(input, [{ step: finalSlot, action, shifts }]));
        const afterPlan = await gpu.checkpointAsync();
        assertTrained(beforePlan, afterPlan, 'delayed target');
        assert(afterPlan.model.steps === beforePlan.model.steps, 'Delayed targets do not count transitions');
        close(beforePlan.model.outputWeights.slice(0, finalSlot * 5).flat(), afterPlan.model.outputWeights.slice(0, finalSlot * 5).flat(), 'Earlier slots receive no delayed gradient');
        close(beforePlan.model.outputWeights.slice(-1).flat(), afterPlan.model.outputWeights.slice(-1).flat(), 'No critic gradient from demonstrations');
        const twin = await create(config, afterPlan), finalTarget = { step: finalSlot, action: { throttle: 0.1, brake: 0.9, steering: 0.4 }, shifts: { up: false, down: false } };
        await gpu.learnPlannedActionsAsync(input, [{ step: finalSlot, action, shifts }, finalTarget]);
        await twin.learnPlannedActionsAsync(input, [finalTarget]);
        compare(await gpu.checkpointAsync(), await twin.checkpointAsync(), 'Duplicate target slots use the last target'); twin.dispose();
        const unchanged = await gpu.checkpointAsync(), bufferCount = gpu.gpu.buffers.size;
        await withoutReadbacks('empty targets', () => gpu.learnPlannedActionsAsync(input, []));
        for (const invalid of [[{ step: -1, action, shifts }], [{ step: 0, action: { ...action, throttle: NaN }, shifts }],
            [{ step: 0, action: { ...action, shiftDown: true }, shifts }]]) {
            let rejected = false;
            try { await gpu.learnPlannedActionsAsync(input, invalid); } catch { rejected = true; }
            assert(rejected, 'Invalid targets must be rejected');
        }
        compare(unchanged, await gpu.checkpointAsync(), 'Invalid targets preserve weights');
        assert(gpu.gpu.buffers.size === bufferCount, 'Invalid targets must not leak buffers');
        let pendingCheckpoint;
        await gpu.trainBatchAsync(async () => {
            const beforeReadbacks = readbacks;
            await gpu.learnDemonstrationAsync(input, action, shifts);
            pendingCheckpoint = gpu.checkpointAsync();
            await gpu.learnPlannedActionsAsync(input, targets);
            await gpu.learnPlannedActionsAsync(next, targets);
            assert(readbacks === beforeReadbacks, 'Checkpoint waits until the batch is complete');
        });
        const checkpoint = await gpu.checkpointAsync();
        compare(checkpoint, await pendingCheckpoint, 'Whole-batch checkpoint');
        const restored = await create(config, checkpoint);
        close((await predict(gpu, input)).plan.flatMap(step => step.means),
            (await predict(restored, input)).plan.flatMap(step => step.means), 'Checkpoint reload'); restored.dispose();
        const timing = [];
        if (layout === 'default') {
            const samples = Array.from({ length: 7 }, (_, i) => makeInput(config, i, true, true));
            let warmedAllocations, warmedBuffers;
            for (let iteration = 0; iteration < 2; iteration++) {
                const start = performance.now(), beforeReadbacks = readbacks;
                await withoutReadbacks('Full training tick', () => gpu.trainBatchAsync(async () => {
                    await gpu.learnDemonstrationAsync(samples[0], action, shifts);
                    for (const sample of samples.slice(1)) await gpu.learnPlannedActionsAsync(sample, targets);
                }));
                await predict(gpu, next);
                assert(readbacks === beforeReadbacks + 1, 'Only the control decision reads back during a training tick');
                timing.push(performance.now() - start);
                if (iteration === 0) { warmedAllocations = allocations(); warmedBuffers = gpu.gpu.buffers.size; }
                else {
                    assert(gpu.gpu.buffers.size === warmedBuffers, 'GPU buffers remain bounded across ticks');
                    assert(allocations().every((count, i) => count === warmedAllocations[i]),
                        'Warmed training ticks reuse all temporary GPU buffers and bind groups');
                }
            }
        }
        const validation = await gpu.gpu.device.popErrorScope(); assert(!validation, validation?.message);
        gpu.gpu.device.destroy(); await gpu.gpu.device.lost;
        for (const operation of [() => gpu.decideAsync(input, false), () => gpu.learnDemonstrationAsync(input, action, shifts)]) {
            let lost = false; try { await operation(); } catch (error) { lost = /GPU lost/.test(error.message); }
            assert(lost, 'Device loss must fail without a CPU path');
        }
        gpu.dispose(); gpu.dispose();
        results.push({ layout, millisecondsPerTick: timing });
    }

    // Check backpropagation against finite differences of an objective evaluated
    // by real GPU forward passes. No second model or CPU training algorithm.
    const config = cloneConfig();
    config.observation.features = ['camera', 'speed']; config.memory.features = ['camera', 'speed'];
    config.memory.windowMs = 400; config.network.hiddenUnits = 2;
    config.network.cameraLayers.forEach(layer => { layer.channels = 1; });
    config.network.cameraProjectionUnits = 32;
    config.learning.demonstrationRate = 0.05;

    // Interpolation credits may revisit a slot before and after its discrete
    // gear choice. Their accumulation must match one combined GPU target.
    const blended = await create(config), blendedSeed = await blended.checkpointAsync();
    const combined = await create(config, blendedSeed), blendedInput = makeInput(config);
    const blendedDecision = await blended.decideAsync(blendedInput, true, shifts);
    const combinedDecision = await combined.decideAsync(blendedInput, false, shifts);
    combinedDecision.plan = blendedDecision.plan;
    const analog = { step: 1, weight: 0.25, shiftWeight: 0, shift: 0, shifts: { up: false, down: false } };
    const due = { ...analog, weight: 0.5, shiftWeight: 0.5, shift: 1, shifts };
    await blended.learnAsync(blendedDecision, 1, blendedInput, 0.2, [analog, due, analog], true);
    await combined.learnAsync(combinedDecision, 1, blendedInput, 0.2, [{ ...due, weight: 1 }], true);
    compare(await blended.checkpointAsync(), await combined.checkpointAsync(), 'Repeated interpolation credits accumulate');
    const beforeAnalogOnly = await blended.checkpointAsync();
    await blended.learnAsync(blendedDecision, 1, blendedInput, 0.2, [analog], true);
    const afterAnalogOnly = await blended.checkpointAsync();
    assert(changed(beforeAnalogOnly.model.outputWeights.slice(5, 7), afterAnalogOnly.model.outputWeights.slice(5, 7)),
        'Interpolated future analog controls receive reward');
    close(beforeAnalogOnly.model.outputWeights.slice(7, 10).flat(), afterAnalogOnly.model.outputWeights.slice(7, 10).flat(),
        'Interpolation gives no gradient to an unexecuted future gear choice');
    blended.releaseDecision(blendedDecision); combined.releaseDecision(combinedDecision);
    blended.dispose(); combined.dispose();

    // A rollout must equal the sum of independent gradients on its frozen
    // collection weights, with a single frozen endpoint bootstrap. Replay
    // released decisions to exercise bounded activation storage as well.
    const sequencePolicy = await create(config), sequenceSeed = await sequencePolicy.checkpointAsync();
    const sequenceInputs = [0, 1, 2, 3].map(frame => makeInput(config, frame));
    const sequenceTimes = [1000, 1500, 3000, 4500], sequenceRewards = [0.2, 0.4, -2];
    const sequence = [];
    for (let i = 0; i < 3; i++) {
        const decision = await sequencePolicy.decideAsync(sequenceInputs[i], true, shifts);
        sequencePolicy.releaseDecision(decision);
        sequence.push({ startedAt: sequenceTimes[i], endedAt: sequenceTimes[i + 1], reward: sequenceRewards[i],
            samples: [{ decision, credits: credits(decision, [0, 2, 6]), rewardShare: 1 }] });
    }
    compare(await sequencePolicy.checkpointAsync(), sequenceSeed, 'Collection keeps every layer fixed');
    const weightKeys = ['cameraWeights', 'cameraProjectionWeights', 'lstmWeights', 'outputWeights'];
    const expectedSequence = Object.fromEntries(weightKeys.map(key => [key, sequenceSeed.model[key].flat(2)]));
    let returnReward = 0;
    for (let i = sequence.length - 1; i >= 0; i--) {
        returnReward = sequenceRewards[i] + config.learning.discount ** ((sequenceTimes[i + 1] - sequenceTimes[i]) / 1000 / config.learning.discountIntervalSeconds) * returnReward;
        const independent = await create(config, sequenceSeed);
        const retained = await independent.decideAsync(sequenceInputs[i], false, shifts);
        retained.plan = sequence[i].samples[0].decision.plan;
        await independent.learnAsync(retained, returnReward, sequenceInputs[3], (4500 - sequenceTimes[i]) / 1000, sequence[i].samples[0].credits);
        const after = await independent.checkpointAsync();
        for (const key of weightKeys) {
            const initial = sequenceSeed.model[key].flat(2), result = after.model[key].flat(2);
            expectedSequence[key] = expectedSequence[key].map((value, j) => value + result[j] - initial[j]);
        }
        independent.releaseDecision(retained); independent.dispose();
    }
    await withoutReadbacks('Multi-step sequence', () => sequencePolicy.trainBatchAsync(() => sequencePolicy.learnSequenceAsync(sequence, sequenceInputs[3])));
    const sequenceResult = await sequencePolicy.checkpointAsync();
    assert(sequenceResult.model.steps === 3, 'Sequence counts committed learning samples');
    for (const key of weightKeys) close(sequenceResult.model[key].flat(2), expectedSequence[key], `Frozen sequence gradients/${key}`);
    const beforeCancel = await sequencePolicy.checkpointAsync();
    let activeChecks = 0;
    await sequencePolicy.trainBatchAsync(() => sequencePolicy.learnSequenceAsync(sequence, sequenceInputs[3], () => ++activeChecks < 3));
    compare(await sequencePolicy.checkpointAsync(), beforeCancel, 'Interrupted sequence commits no weights or steps');
    sequencePolicy.dispose();

    // Verify signed pedal targets, including overlapping human pedals and coast,
    // against independently calculated output-bias updates on the real GPU.
    const pedalSeed = new DrivingPolicy({ game: 'acc', car: 'GT3', map: 'monza' }, drivingObservationInputs(config), () => 0.6, config).checkpoint();
    pedalSeed.model.outputWeights.forEach(row => row.fill(0));
    const pedals = await create(config, pedalSeed), pedalInput = makeInput(config);
    for (const [throttle, brake] of [[0, 0], [1, 0], [0, 1], [0.7, 0.1], [0.1, 0.7], [0.3, 0.3]]) {
        pedals.gpu.device.queue.writeBuffer(pedals.gpu.outputWeights, 0, new Float32Array(pedalSeed.model.outputWeights.flat()));
        const targetAction = { throttle, brake, steering: -0.3 };
        const available = { up: false, down: false };
        await pedals.learnDemonstrationAsync(pedalInput, targetAction, available);
        await pedals.learnPlannedActionsAsync(pedalInput, [{ step: 1, action: targetAction, shifts: available }]);
        const saved = await pedals.checkpointAsync();
        const p = Math.max(config.learning.actionEpsilon - 1, Math.min(1 - config.learning.actionEpsilon, throttle - brake));
        const expected = config.learning.demonstrationRate * Math.atanh(p);
        close([saved.model.outputWeights[0][2], saved.model.outputWeights[5][2]], [expected, expected], 'Signed demonstration and delayed pedal targets');
        close([saved.model.outputWeights[1][2], saved.model.outputWeights[6][2]],
            [config.learning.demonstrationRate * Math.atanh(-0.3), config.learning.demonstrationRate * Math.atanh(-0.3)], 'Steering stays independent of pedals');
        close(saved.model.outputWeights.slice(2, 5).flat(), pedalSeed.model.outputWeights.slice(2, 5).flat(), 'Forced hold uses the shifted logit indices');
    }
    pedals.dispose();
    // Compare reused storage with a newly restored GPU for every update. Vary
    // history length and capture sharing to expose stale state/gradient contents.
    const reused = await create(config);
    for (let iteration = 0; iteration < 4; iteration++) {
        const sample = makeInput(config, iteration, iteration % 2 === 0, iteration === 2);
        const fresh = await create(config, await reused.checkpointAsync());
        const held = await reused.decideAsync(sample, false), expected = await fresh.decideAsync(sample, false);
        const replay = makeInput(config, iteration + 1, true, true);
        await reused.learnDemonstrationAsync(replay, action, shifts);
        await fresh.learnDemonstrationAsync(replay, action, shifts);
        await reused.learnAsync(held, 0.4, replay, 0.2, credits(held), iteration % 2 === 0);
        await fresh.learnAsync(expected, 0.4, replay, 0.2, credits(expected), iteration % 2 === 0);
        reused.releaseDecision(held); fresh.releaseDecision(expected);
        compare(await reused.checkpointAsync(), await fresh.checkpointAsync(), 'Reused workspace versus fresh GPU');
        fresh.dispose();
    }
    reused.dispose();
    const seedPolicy = new DrivingPolicy({ game: 'acc', car: 'GT3', map: 'monza' }, drivingObservationInputs(config), () => 0.6, config);
    const seed = seedPolicy.checkpoint();
    seed.model.cameraWeights.forEach((rows, l) => rows.forEach(row => { row.fill([0.005, 0.03, 0.05, 0.05][l]); row[row.length - 1] = 0.05; }));
    seed.model.outputWeights.forEach(row => { row[0] = 0.1; row[1] = -0.05; });
    const gpu = await create(config, seed), input = makeInput(config, 1, true, true);
    const target = { throttle: Math.tanh(0.4), brake: 0, steering: Math.tanh(0.15), shiftUp: true };
    const objective = async () => {
        const decision = await predict(gpu, input);
        const means = decision.plan[0].means;
        return -0.5 * means.reduce((sum, mean, i) => sum + (mean - [0.4, 0.15][i]) ** 2, 0)
            + Math.log(decision.plan[0].shiftProbabilities[1]);
    };
    const before = await gpu.checkpointAsync(), coordinates = [];
    config.network.cameraLayers.forEach((_, l) => {
        const row = before.model.cameraWeights[l][0];
        for (const column of [0, Math.floor((row.length - 1) / 2), row.length - 2, row.length - 1]) coordinates.push({ key: 'cameraWeights', layer: l, row: 0, column,
            buffer: gpu.gpu.cameraWeights[l], index: column, value: row[column] });
    });
    for (const row of [0, 31]) {
        for (const column of [0, 7, 24, 31, 32]) coordinates.push({ key: 'cameraProjectionWeights', row, column,
            buffer: gpu.gpu.cameraProjectionWeights, index: row * 33 + column, value: before.model.cameraProjectionWeights[row][column] });
    }
    const inputs = drivingPolicyInputs(config), h = config.network.hiddenUnits;
    for (let gate = 0; gate < 4; gate++) {
        const row = gate * h;
        for (const column of [0, inputs, inputs + h]) coordinates.push({ key: 'lstmWeights', row, column,
            buffer: gpu.gpu.lstmWeights, index: row * (inputs + h + 1) + column, value: before.model.lstmWeights[row][column] });
    }
    for (const column of [0, h]) coordinates.push({ key: 'outputWeights', row: 0, column,
        buffer: gpu.gpu.outputWeights, index: column, value: before.model.outputWeights[0][column] });
    for (const coordinate of coordinates) {
        const epsilon = 0.01;
        gpu.gpu.device.queue.writeBuffer(coordinate.buffer, coordinate.index * 4, new Float32Array([coordinate.value + epsilon]));
        const plus = await objective();
        gpu.gpu.device.queue.writeBuffer(coordinate.buffer, coordinate.index * 4, new Float32Array([coordinate.value - epsilon]));
        const minus = await objective();
        gpu.gpu.device.queue.writeBuffer(coordinate.buffer, coordinate.index * 4, new Float32Array([coordinate.value]));
        coordinate.derivative = (plus - minus) / (2 * epsilon);
    }
    await withoutReadbacks('Finite-difference update', () => gpu.learnDemonstrationAsync(input, target, { up: true, down: true }));
    const after = await gpu.checkpointAsync();
    let maxDerivativeError = 0;
    for (const coordinate of coordinates) {
        const rows = coordinate.key === 'cameraWeights' ? after.model.cameraWeights[coordinate.layer] : after.model[coordinate.key];
        const actual = (rows[coordinate.row][coordinate.column] - coordinate.value) / config.learning.demonstrationRate;
        const error = Math.abs(actual - coordinate.derivative);
        assert(error < 0.00005 + Math.abs(coordinate.derivative) * 0.02,
            `Finite differences ${coordinate.key}/${coordinate.layer ?? ''}/${coordinate.row}/${coordinate.column}: ${actual} vs ${coordinate.derivative}`);
        maxDerivativeError = Math.max(maxDerivativeError, error);
    }
    gpu.dispose();

    // Spatial and temporal regressions exercise the same GPU kernels used by training.
    const spatialSeed = JSON.parse(JSON.stringify(seed));
    const layers = drivingCameraLayers(config);
    spatialSeed.model.cameraWeights.forEach((rows, l) => rows.forEach(row => {
        row.fill(0); const layer = layers[l];
        row[(layer.padding * layer.kernelSize + layer.padding) * layer.inputChannels] = 1;
    }));
    spatialSeed.model.cameraProjectionWeights.forEach((row, i) => { row.fill(0); row[i] = 1; });
    const spatial = await create(config, spatialSeed);
    const cameraFeatures = async (pixel) => {
        const sample = makeInput(config, 0, false), rgb = new Uint8Array(384 * 216 * 3);
        rgb[pixel] = 255;
        sample.cameras[sample.cameras.length - 1] = { width: 384, height: 216, rgb };
        const { pass } = await spatial.gpu.forward(sample);
        try { return await spatial.gpu.download([pass.features]); }
        finally { spatial.gpu.release(pass); }
    };
    const left = await cameraFeatures(0), right = await cameraFeatures(368 * 3);
    close([left[2], right[2 + 7]], [1 / 12, 1 / 12], 'Spatial pooling averages each cell');
    assert(left[2 + 7] === 0 && right[2] === 0, 'Projection preserves left versus right positions');
    assert((await cameraFeatures(3))[2] === 0,
        'Convolutions distinguish equal-average images with different pixel positions');
    spatialSeed.model.cameraWeights.forEach((rows, l) => {
        rows.forEach(row => { row.fill(0); row[[56, 8, 8, 5][l]] = 1; });
        spatial.gpu.device.queue.writeBuffer(spatial.gpu.cameraWeights[l], 0, new Float32Array(rows.flat()));
    });
    close([(await cameraFeatures(384 * 216 * 3 - 1))[2 + 31]], [1 / 12],
        'The bottom-right RGB pixel reaches the pooled projection through padded convolutions');
    // A vertical ramp has independently known averages in the four overlapping
    // pooling bands of the 14-row map; boundary rows receive both gradients.
    const ramp = spatial.gpu.upload(Array.from({ length: 24 * 14 }, (_, i) => Math.floor(i / 24)));
    const pooled = spatial.gpu.buffer(drivingCameraPooledInputs(config) * 4);
    const poolDeltas = spatial.gpu.upload(new Array(32).fill(1)), rampGradient = spatial.gpu.buffer(24 * 14 * 4);
    const commands = spatial.gpu.device.createCommandEncoder();
    spatial.gpu.dispatch(commands, 'cameraPool', [ramp, pooled], 1);
    spatial.gpu.dispatch(commands, 'cameraPoolBackward', [poolDeltas, rampGradient], Math.ceil(24 * 14 / 64));
    spatial.gpu.device.queue.submit([commands.finish()]);
    close(Array.from(await spatial.gpu.download([pooled])), [1.5, 4.5, 8.5, 11.5].flatMap(value => new Array(8).fill(value)),
        'Adaptive pooling covers non-divisible height');
    const rampDeltas = await spatial.gpu.download([rampGradient]);
    close([rampDeltas[0], rampDeltas[3 * 24], rampDeltas[10 * 24], rampDeltas[13 * 24]],
        [1 / 12, 1 / 6, 1 / 6, 1 / 12], 'Overlapping pooling gradients accumulate');
    [ramp, pooled, poolDeltas, rampGradient].forEach(buffer => spatial.gpu.destroy(buffer));
    spatialSeed.model.lstmWeights.forEach(row => row.fill(0));
    spatialSeed.model.lstmWeights[2 * h][2 + drivingCameraInputs(config)] = 1;
    spatialSeed.model.outputWeights.forEach(row => row.fill(0));
    spatialSeed.model.outputWeights[0][0] = 1;
    spatial.gpu.device.queue.writeBuffer(spatial.gpu.lstmWeights, 0, new Float32Array(spatialSeed.model.lstmWeights.flat()));
    spatial.gpu.device.queue.writeBuffer(spatial.gpu.outputWeights, 0, new Float32Array(spatialSeed.model.outputWeights.flat()));
    const temporal = makeInput(config), slotSize = drivingMemorySlotInputs(config);
    temporal.features[2] = 0.8; temporal.features[slotSize + 2] = 0.1;
    temporal.features[drivingMemoryInputs(config)] = 0.4;
    const first = (await predict(spatial, temporal)).plan[0].means[0];
    temporal.features[2] = 0.1; temporal.features[slotSize + 2] = 0.8;
    assert(Math.abs((await predict(spatial, temporal)).plan[0].means[0] - first) > 0.001,
        'Prediction depends on historical order with identical current observations');
    temporal.features[2] = 0.8; temporal.features[slotSize + 2] = 0.1;
    close([first], [(await predict(spatial, temporal)).plan[0].means[0]], 'Recurrent state resets between predictions');
    spatial.dispose();

    for (const cameraHistory of [true, false]) {
        const c = cloneConfig(config);
        if (!cameraHistory) c.memory.features = ['speed'];
        const policy = await create(c), sample = makeInput(c);
        sample.cameras[sample.cameras.length - 1] = { width: 384, height: 216, rgb: new Uint8Array(384 * 216 * 3) };
        const start = await policy.checkpointAsync();
        await policy.learnDemonstrationAsync(sample, action, shifts);
        const finish = await policy.checkpointAsync();
        assert(changed(start.model.cameraWeights[0].map(row => row.slice(0, -1)),
            finish.model.cameraWeights[0].map(row => row.slice(0, -1))) === cameraHistory, 'CNN gradients reach historical pixels only when present');
        policy.dispose();
    }
    for (const variant of ['zero-rates', 'large-logits']) {
        const c = cloneConfig(config);
        if (variant === 'zero-rates') { c.learning.actorRate = c.learning.criticRate = c.learning.demonstrationRate = 0; }
        else c.learning.weightLimit = 2000;
        const document = new DrivingPolicy(seed.model.context, drivingObservationInputs(c), () => 0.6, c).checkpoint();
        if (variant === 'large-logits') { document.model.outputWeights[3].fill(0); document.model.outputWeights[3][c.network.hiddenUnits] = 1000; }
        const policy = await create(c, document), sample = makeInput(c), start = await policy.checkpointAsync();
        const available = { up: false, down: false };
        if (variant === 'large-logits') assert((await predict(policy, sample)).plan[0].shiftProbabilities[1] === 1, 'Large logits remain stable');
        await policy.learnDemonstrationAsync(sample, { throttle: 0, brake: 1, steering: -1 }, available);
        if (variant === 'zero-rates') {
            const decision = await policy.decideAsync(sample, true);
            await policy.learnAsync(decision, 2, sample, 0.2, credits(decision)); policy.releaseDecision(decision);
            await policy.learnPlannedActionsAsync(sample, [{ step: 1, action, shifts }]);
        }
        const finish = await policy.checkpointAsync(); finite(finish);
        if (variant === 'zero-rates') {
            for (const key of ['cameraWeights', 'cameraProjectionWeights', 'lstmWeights', 'outputWeights']) close(start.model[key].flat(2), finish.model[key].flat(2), 'Zero learning rates', 1e-12);
            assert(finish.model.steps === 2, 'Count completed transitions with zero rates');
        } else close(start.model.outputWeights.slice(2, 5).flat(), finish.model.outputWeights.slice(2, 5).flat(), 'Masked large logits');
        policy.dispose();
    }
    const soak = [];
    if (soakIterations) for (const variant of ['default', 'saturated-long-history']) {
        const c = cloneConfig();
        if (variant === 'saturated-long-history') {
            c.network.cameraLayers.forEach(layer => { layer.channels = 1; });
            c.network.cameraProjectionUnits = 32; c.network.hiddenUnits = 2;
            c.observation.features = ['camera', 'speed']; c.memory.features = ['camera', 'speed'];
            c.memory.windowMs = 30000;
        }
        let policy = await create(c);
        if (variant === 'saturated-long-history') {
            const saturated = await policy.checkpointAsync(); policy.dispose();
            // Legal, finite weights amplify camera features beyond the tanh
            // overflow range and accumulate cells near +/-101 across history.
            saturated.model.cameraWeights.forEach(rows => rows.forEach(row => row.fill(0.25)));
            saturated.model.cameraProjectionWeights.forEach(row => row.fill(1));
            saturated.model.lstmWeights.forEach((row, index) => {
                row.fill(0); row[2] = index === 5 ? -1 : 1;
            });
            policy = await create(c, saturated);
        }
        const samples = Array.from({ length: 6 }, (_, i) => makeInput(c, i, true, i % 2 === 1 && variant === 'default'));
        const initial = await policy.checkpointAsync(), started = performance.now();
        let checkpointReloads = 0, checkedDecisions = 0, maxCameraFeature = 0, maxCell = 0;
        const checkDecision = async (sample) => {
            const decision = await policy.decideAsync(sample, true);
            assert([decision.value, ...decision.plan.flatMap(step => [...step.means, ...step.latent, ...step.shiftProbabilities,
                step.action.throttle, step.action.brake, step.action.steering])].every(Number.isFinite), `${variant}: finite decision`);
            checkedDecisions++;
            return decision;
        };
        for (let iteration = 0; iteration < soakIterations; iteration++) {
            const sample = samples[iteration % samples.length], next = samples[(iteration + 1) % samples.length];
            const decision = await checkDecision(sample);
            if (iteration === 0 && variant === 'saturated-long-history') {
                const features = await policy.gpu.download([decision.gpu.features]);
                const states = await policy.gpu.download([decision.gpu.states]);
                maxCameraFeature = Math.max(...features.map(Math.abs));
                for (let t = 1; t <= decision.gpu.steps; t++) {
                    for (let unit = 0; unit < c.network.hiddenUnits; unit++) {
                        maxCell = Math.max(maxCell, Math.abs(states[t * 6 * c.network.hiddenUnits + c.network.hiddenUnits + unit]));
                    }
                }
                assert(maxCameraFeature > 100 && maxCell > 89, 'Stress fixture must reach large camera features and cell states');
            }
            const target = { throttle: iteration % 3 ? 1 : 0, brake: iteration % 3 ? 0 : 1,
                steering: Math.sin(iteration * 0.1), shiftUp: iteration % 3 === 0, shiftDown: iteration % 3 === 1 };
            const available = { up: true, down: true };
            try {
                await withoutReadbacks('Soak training', () => policy.trainBatchAsync(async () => {
                    await policy.learnAsync(decision, (iteration % 21) - 10, next, 0.2, credits(decision), iteration % 17 === 0);
                    await policy.learnDemonstrationAsync(sample, target, available);
                    await policy.learnPlannedActionsAsync(sample, [{ step: c.actions.planTimesSeconds.length - 1, action: target, shifts: available }]);
                }));
            } finally { policy.releaseDecision(decision); }
            if ((iteration + 1) % 100 === 0) {
                const checkpoint = await policy.checkpointAsync(); finite(checkpoint);
                assert(checkpoint.model.steps === (iteration + 1) * 2, 'Every soak cycle records both learning steps');
                if ((iteration + 1) % 500 === 0) {
                    const restored = await create(c, JSON.parse(JSON.stringify(checkpoint)));
                    close((await predict(policy, next)).plan.flatMap(step => step.means),
                        (await predict(restored, next)).plan.flatMap(step => step.means), 'Soak checkpoint reload');
                    policy.dispose(); policy = restored; checkpointReloads++;
                }
                console.log(`Training soak ${variant}: ${iteration + 1}/${soakIterations} cycles, finite predictions and weights`);
            }
        }
        policy.releaseDecision(await checkDecision(samples[0]));
        const final = await policy.checkpointAsync(); finite(final);
        if (variant === 'default') assertTrained(initial, final, 'Default soak trains every layer');
        else assert(changed(initial.model.outputWeights, final.model.outputWeights), 'Saturated model continues learning');
        soak.push({ variant, cycles: soakIterations, trainingUpdates: soakIterations * 3, checkedDecisions, checkpointReloads,
            maxCameraFeature, maxCell, seconds: (performance.now() - started) / 1000 });
        policy.dispose();
    }
    assert(errors.length === 0, errors.join('\n'));
    return { adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture },
        checks: 'GPU-only prediction and learning, zero intermediate readbacks, CNN/projection/LSTM/output finite differences, padded borders, spatial ordering and adaptive pooling gradients, all layers and gates train, historical camera gradients, masks, terminal/clipped transitions, delayed plans, checkpoint reload/synchronization, custom layouts/rates, allocation-free warmed ticks, reused versus fresh workspaces, bounded buffers and device loss passed',
        activationCases, sequenceChecks: 'Frozen collection, time-discounted returns, all-layer gradient accumulation, replay without readbacks, and interrupted-batch rollback passed',
        soak, finiteDifferences: { weightsChecked: coordinates.length, maxDerivativeError }, results };
}

app.whenReady().then(async () => {
    let window;
    try {
        await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
        window = new BrowserWindow({ show: false, webPreferences: { contextIsolation: true, nodeIntegration: false, backgroundThrottling: false } });
        window.webContents.on('console-message', (_event, _level, message) => process.stderr.write(message + '\n'));
        await window.loadURL(`http://127.0.0.1:${server.address().port}/`);
        const result = await window.webContents.executeJavaScript(`(async () => { ${bundle}\nreturn await (${checks.toString()})(${process.argv.includes('--soak') ? 1000 : 0}); })()`);
        process.stdout.write(JSON.stringify(result, null, 2) + '\n');
        app.exit(0);
    } catch (error) { process.stderr.write(String(error.stack || error) + '\n'); app.exit(1); }
    finally { server.close(); window?.destroy(); }
});
