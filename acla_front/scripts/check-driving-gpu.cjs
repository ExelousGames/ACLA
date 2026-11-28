// Run: npx electron scripts/check-driving-gpu.cjs
// Add --soak for 1,000 training cycles each on default and saturated long-history models.
// Add --target-gears for the gear distributions, SAC, and recorded gear target checks.
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

async function checks(soakIterations, targetGearsOnly) {
    const { DrivingPolicy } = require('views/ai-driving-training/driving-policy');
    const { cloneConfig, DEFAULT_DRIVING_CONFIG, DRIVING_ANALOG_AXES, DRIVING_GEAR_COUNT, DRIVING_MAX_GEAR, DRIVING_MIN_GEAR, MAX_DIRECTED_CONTROL_SAMPLES,
        drivingCameraInputs, drivingCameraLayers, drivingCameraPooledInputs, drivingGearIndex: gearIndex, drivingGearFromIndex,
        drivingMemorySlots, drivingObservationInputs, drivingPolicyInputs } = require('views/ai-driving-training/driving-model-config');
    const { drivingMemoryInputs, drivingMemorySlotInputs } = require('views/ai-driving-training/driving-memory');
    const { drivingGruStateWidth } = require('views/ai-driving-training/driving-gru-shaders');
    const { sacPolicySampleFloats } = require('views/ai-driving-training/driving-sac-shaders');
    const actionOutputs = DEFAULT_DRIVING_CONFIG.actions.outputs.length;
    const gearRow = gear => DRIVING_ANALOG_AXES + gearIndex(gear);
    // Exercise the optional unnormalized encoder as well as the default encoder.
    const unnormalizedConfig = (source) => {
        const config = cloneConfig(source);
        config.network.cameraLayerNorm = config.network.gruLayerNorm = false;
        config.learning.gradientNormLimit = 1e6;
        return config;
    };
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
        for (const key of ['cameraWeights', 'cameraProjectionWeights', 'cameraNormWeights', 'gruWeights', 'gruNormWeights', 'outputWeights']) {
            if (!a.model[key] && !b.model[key]) continue;
            close(a.model[key].flat(2), b.model[key].flat(2), `${label}/${key}`);
        }
        for (const key of ['logStds', 'criticWeights', 'targetCriticWeights']) close(a.model.sac[key].flat(), b.model.sac[key].flat(), `${label}/sac/${key}`);
        for (const key of Object.keys(a.model.sac.targetEncoder)) close(a.model.sac.targetEncoder[key].flat(2), b.model.sac.targetEncoder[key].flat(2), `${label}/sac/targetEncoder/${key}`);
    };
    const changed = (a, b) => a.flat(2).some((value, i) => value !== b.flat(2)[i]);
    const finite = (checkpoint) => {
        for (const key of ['cameraWeights', 'cameraProjectionWeights', 'cameraNormWeights', 'gruWeights', 'gruNormWeights', 'outputWeights']) {
            if (!checkpoint.model[key]) continue;
            assert(checkpoint.model[key].flat(2).every(Number.isFinite), `Finite ${key}`);
        }
        assert([checkpoint.model.sac.logStds, checkpoint.model.sac.criticWeights,
            checkpoint.model.sac.targetCriticWeights, ...Object.values(checkpoint.model.sac.targetEncoder)]
            .flat(3).every(Number.isFinite), 'Finite SAC parameters and targets');
    };
    const predict = async (policy, input, gears) => {
        const decision = await policy.decideAsync(input, false, gears);
        policy.releaseDecision(decision);
        return decision;
    };
    const assertTrained = (before, after, label) => {
        after.model.cameraWeights.forEach((rows, l) => assert(changed(rows, before.model.cameraWeights[l]), `${label}: CNN layer ${l}`));
        assert(changed(before.model.cameraProjectionWeights, after.model.cameraProjectionWeights), `${label}: spatial projection`);
        const h = after.configuration.network.hiddenUnits;
        for (let gate = 0; gate < 3; gate++) {
            assert(changed(before.model.gruWeights.slice(gate * h, (gate + 1) * h),
                after.model.gruWeights.slice(gate * h, (gate + 1) * h)), `${label}: GRU gate ${gate}`);
        }
        finite(after);
    };
    const create = async (config, checkpoint) => {
        let seed = 42;
        const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
        const policy = checkpoint ? DrivingPolicy.restore(checkpoint)
            : new DrivingPolicy({ game: 'acc', car: 'GT3', map: 'monza' }, drivingObservationInputs(config), random, config);
        await policy.initializeGpu();
        policy.gpu.random = () => 0.61;
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
    const reverseOnly = { max: -1 }, reverseAndNeutral = { max: 0 }, throughSecond = { max: 2 }, allGears = { max: DRIVING_MAX_GEAR };
    const action = { pedal: 0.7, steering: -0.3, gear: 2 };
    const credits = (decision, steps = decision.plan.map((_, step) => step)) => steps.map(step =>
        ({ step, weight: 1 / steps.length, action: decision.plan[step].action, gear: decision.plan[step].action.gear, gears: decision.plan[step].gears }));

    const explorationConfig = unnormalizedConfig();
    explorationConfig.network.hiddenUnits = 1;
    explorationConfig.network.cameraLayers.forEach(layer => { layer.channels = 1; });
    explorationConfig.network.cameraProjectionUnits = 1;
    const explorationSeed = new DrivingPolicy({ game: 'acc', car: 'GT3', map: 'monza' },
        drivingObservationInputs(explorationConfig), () => 0.5, explorationConfig).checkpoint();
    explorationSeed.model.gruWeights[2][drivingPolicyInputs(explorationConfig) + 1] = 0.5;
    const freshExploration = await create(explorationConfig, explorationSeed);
    explorationSeed.model.sac.criticWeights = [
        [0.75, -1, 0.5, 0, -1, 1, 0, 2, 0.1],
        [-0.5, -2, 0.25, 0, -0.5, 0.5, 0, 1, -0.2],
    ].map(row => [...row.slice(0, 3), ...Array(gearIndex(1)).fill(0), ...row.slice(3, 6),
        ...Array(DRIVING_GEAR_COUNT - gearIndex(3) - 1).fill(0), ...row.slice(6)]);
    const restoredExploration = await create(explorationConfig, explorationSeed);
    freshExploration.random = restoredExploration.random = () => 0.98;
    const explorationInput = makeInput(explorationConfig, 0, false), beforeExploration = readbacks;
    const freshPlan = await freshExploration.decideAsync(explorationInput, true);
    const restoredPlan = await restoredExploration.decideAsync(explorationInput, true);
    assert(JSON.stringify(freshPlan.plan) === JSON.stringify(restoredPlan.plan), 'Restored critic weights do not change the sampled plan');
    assert(readbacks === beforeExploration + 2, 'Each exploration plan uses one actor readback');
    const scoreGpu = restoredExploration.gpu;
    const hidden = (await scoreGpu.download([restoredPlan.gpu.states]))[restoredPlan.gpu.steps * drivingGruStateWidth];
    close([hidden], [Math.tanh(0.5) / (1 + Math.exp(1))], 'Scoring fixture current recurrent state');
    const controls = Array.from({ length: MAX_DIRECTED_CONTROL_SAMPLES }, (_, index) =>
        -1 + 2 * index / (MAX_DIRECTED_CONTROL_SAMPLES - 1));
    const candidates = [{ pedal: -0.45, steering: 0.2, gear: 2 }];
    for (let gear = DRIVING_MIN_GEAR; gear <= DRIVING_MAX_GEAR; gear++) {
        for (const pedal of controls) {
            for (const steering of controls) candidates.push({ pedal, steering, gear });
        }
    }
    const candidateReadbacks = readbacks;
    const candidateScores = await scoreGpu.scoreActions(restoredPlan.gpu, candidates);
    assert(readbacks === candidateReadbacks + 1, 'Joint action scoring reads both critics in one batch');
    const candidateColumns = 1 + actionOutputs + 1;
    const expectedScores = candidates.map(({ pedal, steering, gear }) => {
        const criticInput = [hidden, pedal, steering, ...Array.from({ length: DRIVING_GEAR_COUNT }, (_, index) => Number(index === gearIndex(gear))), 1];
        return explorationSeed.model.sac.criticWeights.map(weights => Math.tanh(
            criticInput.reduce((sum, value, index) => sum + value * weights[index], 0)) * weights[candidateColumns] + weights[candidateColumns + 1]);
    });
    close(candidateScores.flat(), expectedScores.flat(), 'Twin-critic joint action scores against CPU oracle');
    freshExploration.releaseDecision(freshPlan); restoredExploration.releaseDecision(restoredPlan);
    freshExploration.dispose(); restoredExploration.dispose();

    // Check the production kernels against a CPU oracle, including gate inputs
    // beyond the GPU tanh overflow threshold and a nonzero previous hidden state.
    const activationConfig = unnormalizedConfig();
    activationConfig.network.hiddenUnits = 1;
    activationConfig.network.cameraLayers.forEach(layer => { layer.channels = 1; });
    activationConfig.network.cameraProjectionUnits = 1;
    const activationPolicy = await create(activationConfig), activationGpu = activationPolicy.gpu;
    const activationInputs = drivingPolicyInputs(activationConfig), columns = activationInputs + 2;
    const gateWeights = new Array(3 * columns).fill(0);
    for (let gate = 0; gate < 3; gate++) gateWeights[gate * columns] = 1;
    const hiddenWeights = [0.25, -0.5, 0.75];
    hiddenWeights.forEach((weight, gate) => { gateWeights[gate * columns + activationInputs] = weight; });
    const weights = activationGpu.upload(gateWeights), features = activationGpu.buffer(activationInputs * 4);
    const states = activationGpu.buffer(10 * 4), upstream = activationGpu.buffer(4), deltas = activationGpu.buffer(4 * 4);
    const sigmoid = x => 1 / (1 + Math.exp(-x));
    const activationValues = [-1e30, -1000, -100, -89, -88, -50, -10, -1, -0.001, 0, 0.001, 1, 10, 50, 88, 89, 100, 1000, 1e30];
    let activationCases = 0;
    for (const value of activationValues) {
        for (const previousHidden of [-1, 0, 1]) {
            const initial = new Float32Array(10); initial[0] = previousHidden;
            const dh = 0.125;
            activationGpu.device.queue.writeBuffer(features, 0, new Float32Array([value]));
            activationGpu.device.queue.writeBuffer(states, 0, initial);
            activationGpu.device.queue.writeBuffer(upstream, 0, new Float32Array([dh]));
            const commands = activationGpu.device.createCommandEncoder();
            activationGpu.dispatch(commands, 'forward', [weights, features, states, activationGpu.times[0]], 1);
            activationGpu.dispatch(commands, 'gateGradient', [states, upstream, deltas, activationGpu.times[0]], 1);
            activationGpu.dispatch(commands, 'resetGradient', [states, deltas, activationGpu.times[0]], 1);
            activationGpu.dispatch(commands, 'recurrentGradient', [weights, deltas, upstream, activationGpu.times[0]], 1);
            activationGpu.device.queue.submit([commands.finish()]);
            const actual = await activationGpu.download([states, deltas, upstream]);
            const x = Math.fround(value);
            const update = sigmoid(x + hiddenWeights[0] * previousHidden);
            const reset = sigmoid(x + hiddenWeights[1] * previousHidden);
            const recurrence = hiddenWeights[2] * previousHidden;
            const candidate = Math.tanh(x + reset * recurrence);
            close(Array.from(actual.slice(5, 10)), [update * previousHidden + (1 - update) * candidate,
                update, reset, candidate, recurrence], `Finite GRU activations at input ${value}, previous hidden ${previousHidden}`);
            const dUpdate = dh * (previousHidden - candidate) * update * (1 - update);
            const dCandidate = dh * (1 - update) * (1 - candidate * candidate);
            const dReset = dCandidate * recurrence * reset * (1 - reset);
            const dRecurrence = dCandidate * reset;
            const dHidden = dh * update + hiddenWeights[0] * dUpdate + hiddenWeights[1] * dReset + hiddenWeights[2] * dRecurrence;
            close(Array.from(actual.slice(10)), [dUpdate, dReset, dCandidate, dRecurrence, dHidden],
                `Finite GRU gradients at input ${value}, previous hidden ${previousHidden}`);
            activationCases++;
        }
    }
    activationPolicy.dispose();

    const results = [];
    for (const layout of ['default', 'custom', 'boundary']) {
        const config = unnormalizedConfig();
        if (layout !== 'default') {
            config.network.cameraLayers.forEach(layer => { layer.channels = 3; });
            config.observation.features = ['camera', 'speed', 'controls', 'centerline'];
            config.memory.features = ['controls', 'speed'];
            config.memory.windowMs = 400;
            config.network.hiddenUnits = 7;
            config.actions.planTimesSeconds = [0, 0.5];
            config.learning.discount = 0.8;
            config.learning.discountIntervalSeconds = 0.3;
            config.learning.criticGradientLimit = 0.4;
            config.learning.policyGradientLimit = 0.7;
            config.learning.demonstrationGradientLimit = 0.5;
            config.learning.actorRate = 0.002;
            config.learning.criticRate = 0.003;
            config.learning.demonstrationRate = 0.004;
            config.learning.shiftTemperature = 0.5;
            config.learning.pedalExplorationSigma = 0.4;
            config.learning.steeringExplorationSigma = 0.8;
        }
        if (layout === 'boundary') {
            config.actions.planTimesSeconds = Array.from({ length: 32 }, (_, i) => i * 0.5);
            config.learning.actionEpsilon = 1e-9;
            config.learning.discount = 0;
        }
        const gpu = await create(config), input = makeInput(config), next = makeInput(config, 1);
        const initial = await gpu.checkpointAsync();
        gpu.gpu.device.pushErrorScope('validation');
        for (const retired of ['decide', 'learn', 'learnDemonstration', 'learnPlannedActions', 'gru', 'cameraEncoder']) {
            assert(!(retired in gpu), `Retired CPU implementation must be absent: ${retired}`);
        }
        for (const sample of [makeInput(config, 3, false), input]) {
            const a = await predict(gpu, sample, throughSecond), b = await predict(gpu, sample, throughSecond);
            close([a.value, ...a.plan.flatMap(step => step.means)], [b.value, ...b.plan.flatMap(step => step.means)], 'deterministic prediction');
            const probabilities = a.plan[0].gearProbabilities;
            assert(probabilities.slice(gearIndex(3)).every(p => p === 0), 'Targets above the gear limit stay masked');
            assert(probabilities[gearIndex(-1)] > 0 && probabilities[gearIndex(0)] > 0, 'Reverse and neutral stay available with a forward cap');
        }
        await withoutReadbacks('demonstration', () => gpu.learnDemonstrationAsync(input, action));
        const demonstration = await gpu.checkpointAsync();
        assertTrained(initial, demonstration, 'demonstration');
        close(initial.model.outputWeights.slice(actionOutputs).flat(), demonstration.model.outputWeights.slice(actionOutputs).flat(), 'Only the demonstrated slot learns');
        await withoutReadbacks('distinct historical images', () => gpu.learnDemonstrationAsync(makeInput(config, 5, true, true), action));
        gpu.random = () => 0.99;
        for (const [reward, terminal, available] of [[0.8, false, throughSecond], [-20, false, { max: 4 }],
            [20, true, reverseOnly], [-0.6, true, allGears]]) {
            const before = await gpu.checkpointAsync(), decision = await gpu.decideAsync(input, true, available);
            const following = terminal ? { features: [], cameras: [] } : next;
            const executed = [0, decision.plan.length - 1];
            await withoutReadbacks('SAC', () => gpu.learnAsync(decision, reward, following, 0.35, credits(decision, executed), terminal));
            gpu.releaseDecision(decision);
            const after = await gpu.checkpointAsync();
            assertTrained(before, after, 'SAC');
            after.model.sac.criticWeights.forEach((rows, index) =>
                assert(changed(before.model.sac.criticWeights[index], rows), `SAC critic ${index} learns`));
            for (const key of ['criticWeights', 'targetCriticWeights']) assert(changed(before.model.sac[key], after.model.sac[key]), `SAC ${key} updates`);
            for (let step = 0; step < decision.plan.length; step++) {
                const from = before.model.outputWeights.slice(step * actionOutputs, (step + 1) * actionOutputs);
                const to = after.model.outputWeights.slice(step * actionOutputs, (step + 1) * actionOutputs);
                if (executed.includes(step)) assert(changed(from, to), `Executed plan slot ${step} receives reward`);
                else close(from.flat(), to.flat(), `Unexecuted slot ${step} receives no reward`);
            }
            if (available.max === reverseOnly.max) close(before.model.outputWeights.slice(DRIVING_ANALOG_AXES, actionOutputs).flat(), after.model.outputWeights.slice(DRIVING_ANALOG_AXES, actionOutputs).flat(), 'A single available gear receives no categorical gradient');
        }
        for (const [pedal, steering, gear] of [[-1, -1, DRIVING_MIN_GEAR], [1, 1, DRIVING_MAX_GEAR]]) {
            await withoutReadbacks('boundary demonstration', () => gpu.learnDemonstrationAsync(input,
                { pedal, steering, gear }));
            finite(await gpu.checkpointAsync());
        }
        const targets = [{ step: 1, action }], finalSlot = config.actions.planTimesSeconds.length - 1;
        const beforePlan = await gpu.checkpointAsync();
        await withoutReadbacks('delayed targets', () => gpu.learnPlannedActionsAsync(input, [{ step: finalSlot, action }]));
        const afterPlan = await gpu.checkpointAsync();
        assertTrained(beforePlan, afterPlan, 'delayed target');
        assert(afterPlan.model.steps === beforePlan.model.steps, 'Delayed targets do not count transitions');
        close(beforePlan.model.outputWeights.slice(0, finalSlot * actionOutputs).flat(), afterPlan.model.outputWeights.slice(0, finalSlot * actionOutputs).flat(), 'Earlier slots receive no delayed gradient');
        close(beforePlan.model.outputWeights.slice(-1).flat(), afterPlan.model.outputWeights.slice(-1).flat(), 'No critic gradient from demonstrations');
        const twin = await create(config, afterPlan), finalTarget = { step: finalSlot, action: { pedal: -0.9, steering: 0.4, gear: 1 } };
        await gpu.learnPlannedActionsAsync(input, [{ step: finalSlot, action }, finalTarget]);
        await twin.learnPlannedActionsAsync(input, [finalTarget]);
        compare(await gpu.checkpointAsync(), await twin.checkpointAsync(), 'Duplicate target slots use the last target'); twin.dispose();
        const unchanged = await gpu.checkpointAsync(), bufferCount = gpu.gpu.buffers.size;
        await withoutReadbacks('empty targets', () => gpu.learnPlannedActionsAsync(input, []));
        for (const invalid of [[{ step: -1, action }], [{ step: 0, action: { ...action, pedal: NaN } }],
            [{ step: 0, action: { ...action, gear: DRIVING_MIN_GEAR - 1 } }],
            [{ step: 0, action: { ...action, gear: DRIVING_MAX_GEAR + 1 } }]]) {
            let rejected = false;
            try { await gpu.learnPlannedActionsAsync(input, invalid); } catch { rejected = true; }
            assert(rejected, 'Invalid targets must be rejected');
        }
        compare(unchanged, await gpu.checkpointAsync(), 'Invalid targets preserve weights');
        assert(gpu.gpu.buffers.size === bufferCount, 'Invalid targets must not leak buffers');
        let pendingCheckpoint;
        await gpu.trainBatchAsync(async () => {
            const beforeReadbacks = readbacks;
            await gpu.learnDemonstrationAsync(input, action);
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
                    await gpu.learnDemonstrationAsync(samples[0], action);
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
        for (const operation of [() => gpu.decideAsync(input, false), () => gpu.learnDemonstrationAsync(input, action)]) {
            let lost = false; try { await operation(); } catch (error) { lost = /GPU lost/.test(error.message); }
            assert(lost, 'Device loss must fail without a CPU path');
        }
        gpu.dispose(); gpu.dispose();
        results.push({ layout, millisecondsPerTick: timing });
    }

    // Check backpropagation against finite differences of an objective evaluated
    // by real GPU forward passes. No second model or CPU training algorithm.
    const config = unnormalizedConfig();
    config.observation.features = ['camera', 'speed', 'centerline']; config.memory.features = ['camera', 'speed', 'centerline'];
    config.memory.windowMs = 400; config.network.hiddenUnits = 2;
    config.network.cameraLayers.forEach(layer => { layer.channels = 1; });
    config.network.cameraProjectionUnits = 32;
    config.learning.demonstrationRate = 0.05;

    // Independent Bellman oracle: zero online Q heads and constant target Qs
    // isolate the minimum, elapsed-time discount, terminal mask and Polyak step.
    const sacConfig = cloneConfig(config);
    sacConfig.learning.entropyTemperature = 1e-12;
    sacConfig.learning.criticGradientLimit = 100;
    sacConfig.actions.planTimesSeconds = [0, 0.1, 0.2];
    const sacSeed = new DrivingPolicy({ game: 'acc', car: 'GT3', map: 'monza' }, drivingObservationInputs(sacConfig), () => 0.6, sacConfig).checkpoint();
    sacSeed.model.sac.criticWeights.forEach(row => row.fill(0));
    sacSeed.model.sac.targetCriticWeights.forEach((row, i) => { row.fill(0); row[row.length - 1] = 2 + i * 2; });
    const sacInput = makeInput(sacConfig), sacNext = makeInput(sacConfig, 1);
    const sacAction = { pedal: 0.4, steering: -0.2, gear: -1 };
    const sacCredit = { step: 0, weight: 1, gear: -1, gears: reverseOnly, action: sacAction };
    const directSac = async (policy, next, terminal, actions = [sacCredit], reward = 0.25) => {
        const pass = policy.gpu.replay(sacInput);
        try { policy.gpu.learnTransition(pass, next, { reward, elapsedSeconds: 0.4, terminal, actions,
            nextGears: reverseOnly }); }
        finally { policy.gpu.release(pass); }
        await policy.gpu.finishTraining();
    };
    for (const terminal of [true, false]) {
        const policy = await create(sacConfig, sacSeed), before = await policy.checkpointAsync();
        await withoutReadbacks('SAC Bellman target', () => directSac(policy, terminal ? { features: [], cameras: [] } : sacNext, terminal));
        const after = await policy.checkpointAsync();
        const target = 0.25 + (terminal ? 0 : 2 * sacConfig.learning.discount ** (0.4 / sacConfig.learning.discountIntervalSeconds));
        for (let q = 0; q < 2; q++) {
            const weights = after.model.sac.criticWeights[q], expected = sacConfig.learning.criticRate * target;
            close(weights.slice(-1), [expected], `SAC twin critic ${q} Bellman oracle (terminal=${terminal})`, 0.0000002);
            close(after.model.sac.targetCriticWeights[q], before.model.sac.targetCriticWeights[q].map((old, i) =>
                old * (1 - sacConfig.learning.targetTau) + weights[i] * sacConfig.learning.targetTau), 'Polyak critic targets', 0.000001);
        }
        close(before.model.outputWeights.flat(), after.model.outputWeights.flat(), 'Reward does not directly weight old actor actions', 1e-10);
        close(before.model.sac.logStds, after.model.sac.logStds, 'Zero Q and entropy leave actor deviation unchanged', 1e-10);
        finite(after); policy.dispose();
    }

    // Different behavior actions change Q regression, while fresh-policy actor
    // gradients are identical for the same current state, weights and noise.
    const actionSeed = JSON.parse(JSON.stringify(sacSeed));
    actionSeed.model.outputWeights[0][sacConfig.network.hiddenUnits] = 0.5;
    actionSeed.model.sac.criticWeights.forEach(row => {
        row[sacConfig.network.hiddenUnits] = 0.4;
        row[sacConfig.network.hiddenUnits * (sacConfig.network.hiddenUnits + actionOutputs + 1)] = 0.7;
    });
    const lowAction = await create(sacConfig, actionSeed), highAction = await create(sacConfig, actionSeed);
    await directSac(lowAction, sacNext, true, [{ ...sacCredit, action: { ...sacAction, pedal: 0.1 } }]);
    await directSac(highAction, sacNext, true, [{ ...sacCredit, action: { ...sacAction, pedal: 0.9 } }]);
    const lowAfter = await lowAction.checkpointAsync(), highAfter = await highAction.checkpointAsync();
    close(lowAfter.model.outputWeights.flat(), highAfter.model.outputWeights.flat(), 'Actor uses current samples independent of replay action', 1e-9);
    close(lowAfter.model.sac.logStds, highAfter.model.sac.logStds, 'SAC deviation gradient uses current samples', 1e-9);
    assert(changed(actionSeed.model.outputWeights, lowAfter.model.outputWeights), 'Current actor sample exercises Q policy improvement');
    assert(changed(lowAfter.model.sac.criticWeights, highAfter.model.sac.criticWeights), 'Critics consume actual replay controls');
    lowAction.dispose(); highAction.dispose();

    // Shift all Q scores with their output biases. Reverse and neutral select
    // different critics, and both critics prefer less brake.
    const pedalGradients = [];
    for (const offset of [0, -4]) {
        const seed = JSON.parse(JSON.stringify(sacSeed)), h = sacConfig.network.hiddenUnits;
        seed.model.outputWeights.slice(0, actionOutputs).forEach(row => row.fill(0));
        seed.model.outputWeights[0][h] = -0.4;
        seed.model.sac.criticWeights.forEach((row, q) => {
            row[h] = q === 0 ? 0.5 : 1;
            row[h + gearRow(0)] = q === 0 ? 1 : -0.25;
            row[h * (h + actionOutputs + 1)] = 1;
            row[row.length - 1] = offset + q * 0.25;
        });
        seed.model.sac.targetCriticWeights = seed.model.sac.criticWeights.map(row => row.slice());
        const policy = await create(sacConfig, seed), gpu = policy.gpu;
        await directSac(policy, sacNext, true, [{ ...sacCredit, gears: reverseAndNeutral }]);
        pedalGradients.push(Array.from(await gpu.download([gpu.training.get('outputGradient'), gpu.training.get('logStdGradient')])));
        policy.dispose();
    }
    close(pedalGradients[0], pedalGradients[1], 'Adding a constant to every Q score preserves the actor gradient', 1e-7);
    assert(pedalGradients[0][0] > 0, 'The actor follows the critics toward releasing the brake');
    const noise = Math.sqrt(-2 * Math.log(0.61)) * Math.cos(2 * Math.PI * 0.61);
    const pedalObjective = (mean, logStd) => {
        const brake = -Math.tanh(mean + Math.exp(logStd) * noise);
        const first = Math.min(Math.tanh(-0.5 * brake), Math.tanh(-brake) + 0.25);
        const second = Math.min(Math.tanh(1 - 0.5 * brake), Math.tanh(-0.25 - brake) + 0.25);
        return (first + second) / 2;
    };
    const pedalLogStd = Math.log(sacConfig.learning.pedalExplorationSigma), delta = 1e-4;
    const expectedPedalGradients = [pedalObjective(-0.4 + delta, pedalLogStd) - pedalObjective(-0.4 - delta, pedalLogStd),
        pedalObjective(-0.4, pedalLogStd + delta) - pedalObjective(-0.4, pedalLogStd - delta)]
        .map(value => value / (2 * delta) * sacConfig.learning.actorRate / sacConfig.learning.demonstrationRate);
    close([pedalGradients[0][0], pedalGradients[0][sacSeed.model.outputWeights.length]],
        expectedPedalGradients, 'Signed pedal mean and log deviation follow finite differences of the twin-Q objective', 1e-7);
    assert(pedalGradients[0][gearRow(-1)] > 0 && pedalGradients[0][gearRow(0)] < 0
        && pedalGradients[0].slice(gearRow(1), actionOutputs).every(value => value === 0),
        'Gear gradient favors reverse over neutral and masks forward gears');

    const entropySeed = JSON.parse(JSON.stringify(sacSeed));
    entropySeed.configuration.learning.entropyTemperature = 0.02;
    const entropyPolicy = await create(entropySeed.configuration, entropySeed);
    await directSac(entropyPolicy, sacNext, true);
    const entropyAfter = await entropyPolicy.checkpointAsync();
    assert(changed(entropySeed.model.sac.logStds, entropyAfter.model.sac.logStds), 'SAC learns exploration deviation from entropy');
    finite(entropyAfter); entropyPolicy.dispose();

    // Check the signed policy density, action Jacobian, and entropy gradients
    // with different configured spreads for pedal and steering.
    const distributionConfig = cloneConfig(sacConfig);
    distributionConfig.learning.pedalExplorationSigma = 0.1;
    const distribution = await create(distributionConfig), distributionGpu = distribution.gpu;
    const outputValues = new Array(distributionConfig.actions.planTimesSeconds.length * actionOutputs + 1).fill(0);
    outputValues.splice(0, DRIVING_ANALOG_AXES, 0, 0.1);
    const distributionOutputs = distributionGpu.upload(outputValues);
    const distributionStds = distributionGpu.upload(new Array(distributionConfig.actions.planTimesSeconds.length * DRIVING_ANALOG_AXES).fill(Math.log(0.05)));
    const distributionTransition = distributionGpu.upload([0, 0.2, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, -1, -0.5, -1, -0.5]);
    const distributionSamples = distributionGpu.buffer(2 * sacPolicySampleFloats * 4), sampleCommands = distributionGpu.device.createCommandEncoder();
    distributionGpu.dispatch(sampleCommands, 'sacSample', [distributionOutputs, distributionOutputs, distributionStds, distributionTransition, distributionSamples], 2);
    distributionGpu.device.queue.submit([sampleCommands.finish()]);
    const measuredSamples = await distributionGpu.download([distributionSamples]);
    const [signedPedal, steering] = [Math.tanh(-0.1), Math.tanh(0.075)];
    const [pedalJacobian, steeringJacobian] = [1 - signedPedal ** 2, 1 - steering ** 2];
    const expectedLogp = -0.5 - 0.125 - Math.log(0.1 * 0.05 * 2 * Math.PI * pedalJacobian * steeringJacobian);
    const expectedSample = [signedPedal, steering, expectedLogp, pedalJacobian, steeringJacobian,
        pedalJacobian * -0.1, steeringJacobian * -0.025, 2 * signedPedal, 2 * steering,
        2 * signedPedal * -0.1 - 1, 2 * steering * -0.025 - 1, 1, ...Array(DRIVING_GEAR_COUNT - 1).fill(0)];
    close(Array.from(measuredSamples.slice(0, sacPolicySampleFloats)), expectedSample, 'Signed pedal density, action Jacobian and entropy gradient');
    close(Array.from(measuredSamples.slice(sacPolicySampleFloats)), expectedSample, 'Next actor distribution matches current sampler');
    distribution.dispose();

    // Fit a critic from an observed outcome, then check policy improvement at
    // every horizon. The other critic is optimistic and favors the opposite pedal.
    for (const reward of [-1, 1]) {
        const seed = JSON.parse(JSON.stringify(sacSeed)), h = sacConfig.network.hiddenUnits;
        seed.configuration.learning.criticRate = 0.1;
        seed.configuration.learning.actorRate = 0.05;
        seed.model.outputWeights.forEach(row => row.fill(0));
        const [lower, optimistic] = seed.model.sac.criticWeights;
        lower[h] = 0.8;
        lower[h + 1] = -0.5;
        lower[h + gearRow(0)] = 0.6;
        optimistic[h] = -1;
        optimistic[h * (h + actionOutputs + 1)] = 1;
        optimistic[optimistic.length - 1] = 3;
        const policy = await create(seed.configuration, seed);
        const neutral = { ...sacCredit, gear: 0, gears: reverseAndNeutral, action: { ...sacAction, gear: 0 } };
        await directSac(policy, sacNext, true, [neutral], reward);
        const replay = policy.gpu.replay(sacNext);
        const steps = sacConfig.actions.planTimesSeconds;
        policy.gpu.learnTransition(replay, sacNext, { reward, elapsedSeconds: 0.4, terminal: true,
            actions: steps.map((_, step) => ({ ...neutral, step, weight: 1 / steps.length })), nextGears: reverseAndNeutral, originInput: sacInput });
        policy.gpu.release(replay);
        const learned = await policy.checkpointAsync();
        policy.random = () => 0.98;
        const decision = await policy.decideAsync(sacInput, true, reverseAndNeutral);
        decision.plan.forEach(({ means, latent }, step) => {
            assert(reward * means[0] > 0 && reward * means[1] < 0, `Outcome steers preferred controls at step ${step}`);
            const rows = learned.model.outputWeights.slice(step * actionOutputs + DRIVING_ANALOG_AXES, (step + 1) * actionOutputs);
            assert(reward * rows[gearIndex(0)][h] > 0 && reward * rows[gearIndex(-1)][h] < 0
                && rows.slice(gearIndex(1)).every(row => row[h] === 0), `Outcome steers allowed gears at step ${step}`);
            assert(reward * (learned.model.sac.logStds[step * DRIVING_ANALOG_AXES] - Math.fround(Math.log(0.05))) < 0,
                `Outcome changes exploration spread at step ${step}`);
            assert(latent.every((value, axis) => value !== means[axis]), `Learned step ${step} still explores`);
        });
        policy.releaseDecision(decision); policy.dispose();
    }

    const blended = await create(config), blendedInput = makeInput(config);
    const blendedDecision = await blended.decideAsync(blendedInput, true, throughSecond);
    const analog = { step: 1, weight: 0.25, gearWeight: 0, gear: -1, gears: reverseOnly, action: sacAction };
    const beforeAnalogOnly = await blended.checkpointAsync();
    await blended.learnAsync(blendedDecision, 1, blendedInput, 0.2, [analog], true);
    const afterAnalogOnly = await blended.checkpointAsync();
    assert(changed(beforeAnalogOnly.model.outputWeights.slice(actionOutputs, actionOutputs + DRIVING_ANALOG_AXES), afterAnalogOnly.model.outputWeights.slice(actionOutputs, actionOutputs + DRIVING_ANALOG_AXES)),
        'Interpolated future analog controls receive an SAC update');
    close(beforeAnalogOnly.model.outputWeights.slice(actionOutputs + DRIVING_ANALOG_AXES, 2 * actionOutputs).flat(), afterAnalogOnly.model.outputWeights.slice(actionOutputs + DRIVING_ANALOG_AXES, 2 * actionOutputs).flat(),
        'Interpolation gives no gradient to an unexecuted future gear choice');
    blended.releaseDecision(blendedDecision); blended.dispose();

    // Multi-step returns use the observed horizon endpoint even after their
    // GPU decisions are released. Interrupted batches cannot commit any weights.
    const sequencePolicy = await create(config), sequenceSeed = await sequencePolicy.checkpointAsync();
    const sequenceInputs = [0, 1, 2, 3].map(frame => makeInput(config, frame));
    const sequenceTimes = [1000, 1500, 3000, 4500], sequenceRewards = [0.2, 0.4, -2];
    const sequence = [];
    for (let i = 0; i < 3; i++) {
        const decision = await sequencePolicy.decideAsync(sequenceInputs[i], true, throughSecond);
        sequencePolicy.releaseDecision(decision);
        sequence.push({ startedAt: sequenceTimes[i], endedAt: sequenceTimes[i + 1], reward: sequenceRewards[i],
            input: sequenceInputs[i], next: sequenceInputs[i + 1], nextGears: throughSecond,
            samples: [{ decision, credits: credits(decision, [0, 2, 6]), rewardShare: 1 }] });
    }
    compare(await sequencePolicy.checkpointAsync(), sequenceSeed, 'Collection keeps every layer fixed');
    await withoutReadbacks('SAC replay batch', () => sequencePolicy.trainBatchAsync(() => sequencePolicy.learnSequenceAsync(sequence, sequenceInputs[3])));
    const sequenceResult = await sequencePolicy.checkpointAsync();
    assert(sequenceResult.model.steps > 0, 'Replay counts committed learning samples');
    assert(changed(sequenceSeed.model.sac.criticWeights, sequenceResult.model.sac.criticWeights), 'Delayed replay trains twin critics');
    const beforeCancel = await sequencePolicy.checkpointAsync();
    let activeChecks = 0;
    await sequencePolicy.trainBatchAsync(() => sequencePolicy.learnSequenceAsync(sequence, sequenceInputs[3], () => ++activeChecks < 3));
    compare(await sequencePolicy.checkpointAsync(), beforeCancel, 'Interrupted sequence commits no weights or steps');
    sequencePolicy.dispose();

    // Verify signed pedal targets at full brake, full throttle, and coast,
    // against independently calculated output-bias updates on the real GPU.
    const pedalSeed = new DrivingPolicy({ game: 'acc', car: 'GT3', map: 'monza' }, drivingObservationInputs(config), () => 0.6, config).checkpoint();
    pedalSeed.model.outputWeights.forEach(row => row.fill(0));
    const pedals = await create(config, pedalSeed), pedalInput = makeInput(config);
    for (const pedal of [-1, -0.7, 0, 0.7, 1]) {
        pedals.gpu.device.queue.writeBuffer(pedals.gpu.outputWeights, 0, new Float32Array(pedalSeed.model.outputWeights.flat()));
        const targetAction = { pedal, steering: -0.3, gear: 1 };
        await pedals.learnPlannedActionsAsync(pedalInput, [0, 1].map(step => ({ step, action: targetAction, gearWeight: 0 })));
        const saved = await pedals.checkpointAsync();
        const limit = 1 - config.learning.actionEpsilon;
        const expected = config.learning.demonstrationRate * Math.atanh(Math.max(-limit, Math.min(limit, pedal)));
        close([saved.model.outputWeights[0][2], saved.model.outputWeights[actionOutputs][2]], [expected, expected],
            'Signed immediate and delayed pedal targets');
        close([saved.model.outputWeights[1][2], saved.model.outputWeights[actionOutputs + 1][2]],
            [config.learning.demonstrationRate * Math.atanh(-0.3), config.learning.demonstrationRate * Math.atanh(-0.3)], 'Steering learns independently');
        close(saved.model.outputWeights.slice(DRIVING_ANALOG_AXES, actionOutputs).flat(), pedalSeed.model.outputWeights.slice(DRIVING_ANALOG_AXES, actionOutputs).flat(), 'Missing gear labels leave logits unchanged');
    }
    pedals.dispose();

    // Check unweighted recorded control gradients on the real GPU, including
    // the last slot's target stride and recovery from wrong saturation.
    const recordedSeed = JSON.parse(JSON.stringify(pedalSeed));
    recordedSeed.configuration.learning.weightLimit = 100;
    const biasColumn = config.network.hiddenUnits;
    const finalRecordedSlot = config.actions.planTimesSeconds.length - 1;
    const recordedCases = [
        { step: 0, action: { pedal: -1, steering: 0, gear: -1 },
            means: [25, 0.4] },
        { step: 1, action: { pedal: 0.7, steering: 0.3, gear: 1 }, gearWeight: 0,
            means: [-25, -25] },
        { step: finalRecordedSlot, action: { pedal: -0.35, steering: -0.2, gear: DRIVING_MAX_GEAR },
            means: [0.7, 0.2] },
    ];
    for (const { step, means } of recordedCases) {
        means.forEach((value, axis) => { recordedSeed.model.outputWeights[step * actionOutputs + axis][biasColumn] = value; });
    }
    const recorded = await create(recordedSeed.configuration, recordedSeed);
    const standardTarget = { step: 2, action: { pedal: 0.7, steering: -0.3, gear: 1 }, gearWeight: 0 };
    await withoutReadbacks('Recorded analog update', () => recorded.trainBatchAsync(async () => {
        recorded.gpu.learnTargets(pedalInput, [recordedCases[0]], true, 'recorded');
        recorded.gpu.learnTargets(pedalInput, [standardTarget]);
        recorded.gpu.learnTargets(pedalInput, recordedCases.slice(1), false, 'recorded');
    }));
    const recordedAfter = await recorded.checkpointAsync();
    const recordedGradient = (mean, action) => {
        const objective = (z) => (1 + action) * z - Math.max(0, 2 * z) - Math.log1p(Math.exp(-Math.abs(2 * z)));
        const epsilon = 1e-4;
        const gradient = (objective(mean + epsilon) - objective(mean - epsilon)) / (2 * epsilon);
        const limit = config.learning.demonstrationGradientLimit;
        return Math.max(-limit, Math.min(limit, gradient));
    };
    const expectedRecordedBias = ({ action, means }, scale = 1) => [action.pedal, action.steering]
        .map((value, axis) => means[axis] + config.learning.demonstrationRate * scale * recordedGradient(means[axis], value));
    const analogBias = (model, step) => model.outputWeights.slice(step * actionOutputs, step * actionOutputs + DRIVING_ANALOG_AXES)
        .map((row) => row[biasColumn]);
    for (const target of recordedCases) {
        close(analogBias(recordedAfter.model, target.step), expectedRecordedBias(target), `Recorded analog finite differences at slot ${target.step}`);
    }
    const expectedGearGradient = (target, scale = 1) => Array.from({ length: DRIVING_GEAR_COUNT }, (_, i) =>
        scale * (Number(i === gearIndex(target)) - 1 / DRIVING_GEAR_COUNT) * config.learning.demonstrationRate / config.learning.shiftTemperature);
    close(recordedAfter.model.outputWeights.slice(DRIVING_ANALOG_AXES, actionOutputs).map((row) => row[biasColumn]),
        expectedGearGradient(-1),
        'Recorded reverse uses the categorical cross-entropy derivative');
    close(recordedAfter.model.outputWeights.slice(finalRecordedSlot * actionOutputs + DRIVING_ANALOG_AXES, (finalRecordedSlot + 1) * actionOutputs)
        .map((row) => row[biasColumn]), expectedGearGradient(DRIVING_MAX_GEAR),
        'The highest gear uses the same categorical cross-entropy derivative');
    close(analogBias(recordedAfter.model, standardTarget.step), [standardTarget.action.pedal, standardTarget.action.steering]
        .map((value) => config.learning.demonstrationRate * Math.atanh(value)), 'The standard objective remains distinct on the same GPU');
    close(analogBias(recordedAfter.model, 3), [0, 0], 'Recorded objective skips absent plan targets');
    recorded.dispose();

    const linkedRecorded = await create(recordedSeed.configuration, recordedSeed);
    const delayedGear = { ...recordedCases[1], action: { ...recordedCases[1].action, gear: 0 }, gearWeight: 1 };
    const linkedTargets = [recordedCases[0], delayedGear, recordedCases[2]];
    await linkedRecorded.trainBatchAsync(async () => linkedRecorded.gpu.learnTargets(pedalInput,
        [...linkedTargets, delayedGear], false, 'recorded'));
    const linkedWeights = (await linkedRecorded.checkpointAsync()).model;
    for (const target of linkedTargets) {
        const scale = target.step === 0 ? 1 : 0.25;
        close(analogBias(linkedWeights, target.step), expectedRecordedBias(target, scale),
            `Grouped recorded analog weight at slot ${target.step}`);
    }
    close(linkedWeights.outputWeights.slice(actionOutputs + DRIVING_ANALOG_AXES, 2 * actionOutputs).map((row) => row[biasColumn]),
        expectedGearGradient(0, 0.25),
        'Grouped recorded neutral uses the delayed target weight');
    linkedRecorded.dispose();

    if (targetGearsOnly) {
        assert(errors.length === 0, errors.join('\n'));
        return { adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture },
            checks: 'Target gear distributions and limits, twin-Q SAC gradients, recorded categorical targets, and delayed gear labels passed', results };
    }

    const clippedRecordedSeed = JSON.parse(JSON.stringify(recordedSeed));
    clippedRecordedSeed.configuration.learning.gradientNormLimit = 0.05;
    const clippedRecorded = await create(clippedRecordedSeed.configuration, clippedRecordedSeed);
    await clippedRecorded.trainBatchAsync(async () => clippedRecorded.gpu.learnTargets(pedalInput, [recordedCases[0]], true, 'recorded'));
    const recordedDiagnostics = await clippedRecorded.trainingDiagnosticsAsync();
    assert(recordedDiagnostics?.finite && recordedDiagnostics.clipScale > 0 && recordedDiagnostics.clipScale < 1,
        'Recorded objective uses global gradient clipping');
    close(analogBias((await clippedRecorded.checkpointAsync()).model, 0), expectedRecordedBias(recordedCases[0], recordedDiagnostics.clipScale),
        'Global clipping scales the recorded steering and brake update together');

    clippedRecorded.dispose();

    const adamRecorded = await create(clippedRecordedSeed.configuration, clippedRecordedSeed);
    const adamReference = await create(clippedRecordedSeed.configuration, clippedRecordedSeed);
    const runAdam = async (policy, target, commit = true, invalid = false) => {
        const gpu = policy.gpu;
        await policy.trainBatchAsync(async () => {
            gpu.beginRecordedSequence();
            gpu.learnTargets(pedalInput, [target], false, 'recorded');
            gpu.learnTargets(pedalInput, [target], false, 'recorded');
            if (invalid) gpu.device.queue.writeBuffer(gpu.parameters()[0].gradient, 0, new Float32Array([NaN]));
            gpu.endSequence(commit);
        });
    };
    await withoutReadbacks('Recorded Adam training batch', () => runAdam(adamRecorded, recordedCases[0]));
    await runAdam(adamReference, recordedCases[0]);
    const firstAdam = await adamRecorded.checkpointAsync(), firstAdamStats = await adamRecorded.trainingDiagnosticsAsync();
    close(analogBias(firstAdam.model, 0), [25, 0.4].map(value => value - clippedRecordedSeed.configuration.learning.demonstrationRate),
        'First Adam step debiases both moment estimates');
    assert(firstAdamStats.finite && firstAdamStats.samples === 2 && firstAdamStats.clipScale < 0.5 && firstAdamStats.updateNorm > 0,
        'Recorded Adam averages and clips its training gradient');
    close([firstAdamStats.learningRate], [clippedRecordedSeed.configuration.learning.demonstrationRate], 'Recorded Adam reports its configured rate');
    await runAdam(adamRecorded, delayedGear, false);
    compare(await adamRecorded.checkpointAsync(), firstAdam, 'Cancelled recorded batch commits no weights');
    await runAdam(adamRecorded, delayedGear, true, true);
    compare(await adamRecorded.checkpointAsync(), firstAdam, 'Nonfinite recorded batch commits no weights');
    const invalidAdam = await adamRecorded.trainingDiagnosticsAsync();
    assert(!invalidAdam.finite && invalidAdam.updateNorm === 0, 'Nonfinite recorded batch reports no update');
    await runAdam(adamRecorded, delayedGear);
    await runAdam(adamReference, delayedGear);
    compare(await adamRecorded.checkpointAsync(), await adamReference.checkpointAsync(),
        'Cancelled and nonfinite batches leave both Adam moments and bias correction unchanged');
    adamRecorded.dispose(); adamReference.dispose();

    // Train contradictory controls together so a constant policy cannot pass.
    const nativeConfig = unnormalizedConfig();
    nativeConfig.network.hiddenUnits = 3;
    nativeConfig.network.cameraLayers.forEach(layer => { layer.channels = 1; });
    nativeConfig.network.cameraProjectionUnits = 1;
    nativeConfig.observation.features = ['camera', 'localVelocity', 'centerline'];
    nativeConfig.memory.features = ['localVelocity'];
    nativeConfig.memory.windowMs = nativeConfig.memory.intervalMs = 100;
    nativeConfig.actions.planTimesSeconds = [0];
    nativeConfig.learning.demonstrationRate = 0.02;
    nativeConfig.learning.gradientNormLimit = 1;
    const nativeSeed = new DrivingPolicy({ game: 'acc', car: 'GT3', map: 'monza' },
        drivingObservationInputs(nativeConfig), () => 0.5, nativeConfig).checkpoint();
    const nativeHidden = nativeConfig.network.hiddenUnits;
    nativeSeed.model.gruWeights.forEach(row => row.fill(0));
    nativeSeed.model.outputWeights.forEach(row => row.fill(0));
    nativeSeed.model.outputWeights[gearRow(1)][nativeHidden] = 0.05;
    for (let j = 0; j < nativeHidden; j++) {
        nativeSeed.model.gruWeights[j][drivingPolicyInputs(nativeConfig) + nativeHidden] = -2;
        nativeSeed.model.gruWeights[2 * nativeHidden + j][2 + drivingCameraInputs(nativeConfig) + j] = 2;
    }
    const nativePolicy = await create(nativeConfig, nativeSeed);
    const nativeBatch = [
        { pedal: 0.7, steering: -0.65, gear: -1 },
        { pedal: -0.7, steering: 0.65, gear: 0 },
        { pedal: 0, steering: 0, gear: 1 },
    ].map((action, cue) => {
        const input = makeInput(nativeConfig, 0, false);
        input.features.fill(0);
        const offset = drivingMemoryInputs(nativeConfig);
        input.features.splice(offset, 3, ...Array.from({ length: 3 }, (_, j) => j === cue ? 1 : -1));
        input.recurrent = { episode: {}, capturedAt: 1100, historyTimes: [null] };
        input.recordedHistory = [{ capturedAt: 1000, features: input.features.slice(offset), camera: input.cameras.at(-1) }];
        return { input, targets: [{ step: 0, action }] };
    });
    const nativePredictions = () => Promise.all(nativeBatch.map(async ({ input }) => {
        const recurrent = { ...input.recurrent, episode: {} };
        await predict(nativePolicy, { ...input, recurrent: { ...recurrent, capturedAt: 1000 } });
        return (await predict(nativePolicy, { ...input, recurrent })).plan[0];
    }));
    const nativeBefore = await nativePredictions();
    const nativeBatches = 32;
    await withoutReadbacks('Recorded GRU learns distinct control contexts', async () => {
        for (let i = 0; i < nativeBatches; i++) {
            await nativePolicy.trainBatchAsync(async () => {
                await nativePolicy.learnRecordedBatchAsync(nativeBatch, () => true);
            });
        }
    });
    const nativeAfter = await nativePredictions(), nativeWeights = await nativePolicy.checkpointAsync();
    const recurrentWeights = model => model.gruWeights.map(row => row.slice(drivingPolicyInputs(nativeConfig), -1));
    assert(changed(recurrentWeights(nativeSeed.model), recurrentWeights(nativeWeights.model)), 'Recorded batch trains the GRU recurrent weights');
    assert(nativeWeights.model.steps === nativeBatches * nativeBatch.length, 'Recorded batch counts each committed origin');
    const totalAnalogError = predictions => predictions.reduce((sum, { action }, i) => {
        const target = nativeBatch[i].targets[0].action;
        return sum + Math.abs(action.pedal - target.pedal) + Math.abs(action.steering - target.steering);
    }, 0);
    const analogErrorBefore = totalAnalogError(nativeBefore), analogErrorAfter = totalAnalogError(nativeAfter);
    assert(analogErrorAfter < 0.7 * analogErrorBefore,
        `Recorded batch reduces native analog error across opposing commands (${analogErrorBefore} → ${analogErrorAfter})`);
    assert(nativeAfter[0].action.pedal > 0.2 && nativeAfter[0].action.steering < -0.2
        && nativeAfter[1].action.pedal < -0.2 && nativeAfter[1].action.steering > 0.2,
        'Native actions distinguish throttle, braking, and both steering directions');
    assert(nativeBefore.every(({ action }) => action.gear === 1)
        && nativeAfter.every(({ action }, i) => action.gear === nativeBatch[i].targets[0].action.gear),
        'Native gear logits learn distinct target gears on distinct inputs');
    nativePolicy.dispose();

    // Compare reused storage with a newly restored GPU for every update. Vary
    // history length and capture sharing to expose stale state/gradient contents.
    const reused = await create(config);
    for (let iteration = 0; iteration < 4; iteration++) {
        const sample = makeInput(config, iteration, iteration % 2 === 0, iteration === 2);
        const fresh = await create(config, await reused.checkpointAsync());
        const held = await reused.decideAsync(sample, false), expected = await fresh.decideAsync(sample, false);
        const replay = makeInput(config, iteration + 1, true, true);
        await reused.learnDemonstrationAsync(replay, action);
        await fresh.learnDemonstrationAsync(replay, action);
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
    const target = { pedal: Math.tanh(0.4), steering: Math.tanh(0.15), gear: 2 };
    const objective = async () => {
        const decision = await predict(gpu, input);
        const means = decision.plan[0].means;
        return -0.5 * means.reduce((sum, mean, i) => sum + (mean - [0.4, 0.15][i]) ** 2, 0)
            + Math.log(decision.plan[0].gearProbabilities[gearIndex(target.gear)]);
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
    for (let gate = 0; gate < 3; gate++) {
        const row = gate * h;
        for (const column of [0, inputs, inputs + 1, inputs + h]) coordinates.push({ key: 'gruWeights', row, column,
            buffer: gpu.gpu.gruWeights, index: row * (inputs + h + 1) + column, value: before.model.gruWeights[row][column] });
    }
    // Include all gear logits so the temperature's chain-rule factor is checked directly.
    for (let row = 0; row < actionOutputs; row++) for (const column of [0, h]) coordinates.push({ key: 'outputWeights', row, column,
        buffer: gpu.gpu.outputWeights, index: row * (h + 1) + column, value: before.model.outputWeights[row][column] });
    for (const coordinate of coordinates) {
        const epsilon = 0.01;
        gpu.gpu.device.queue.writeBuffer(coordinate.buffer, coordinate.index * 4, new Float32Array([coordinate.value + epsilon]));
        const plus = await objective();
        gpu.gpu.device.queue.writeBuffer(coordinate.buffer, coordinate.index * 4, new Float32Array([coordinate.value - epsilon]));
        const minus = await objective();
        gpu.gpu.device.queue.writeBuffer(coordinate.buffer, coordinate.index * 4, new Float32Array([coordinate.value]));
        coordinate.derivative = (plus - minus) / (2 * epsilon);
    }
    await withoutReadbacks('Finite-difference update', () => gpu.learnDemonstrationAsync(input, target));
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
    spatialSeed.model.gruWeights.forEach(row => row.fill(0));
    spatialSeed.model.gruWeights[2 * h][2 + drivingCameraInputs(config)] = 1;
    spatialSeed.model.outputWeights.forEach(row => row.fill(0));
    spatialSeed.model.outputWeights[0][0] = 1;
    spatial.gpu.device.queue.writeBuffer(spatial.gpu.gruWeights, 0, new Float32Array(spatialSeed.model.gruWeights.flat()));
    spatial.gpu.device.queue.writeBuffer(spatial.gpu.outputWeights, 0, new Float32Array(spatialSeed.model.outputWeights.flat()));
    const temporal = makeInput(config), slotSize = drivingMemorySlotInputs(config);
    temporal.features[2] = 0.8; temporal.features[slotSize + 2] = 0.1;
    temporal.features[drivingMemoryInputs(config)] = 0.4;
    const first = (await predict(spatial, temporal)).plan[0].means[0];
    temporal.features[2] = 0.1; temporal.features[slotSize + 2] = 0.8;
    assert(Math.abs((await predict(spatial, temporal)).plan[0].means[0] - first) > 0.001,
        'Prediction depends on historical order with identical current observations');
    temporal.features[2] = 0.8; temporal.features[slotSize + 2] = 0.1;
    close([first], [(await predict(spatial, temporal)).plan[0].means[0]], 'Predictions without timestamps are independent');

    // This fixture reads only speed, with update = 0.5 and candidate = tanh(speed).
    const advanceHidden = (previous, speed) => (previous + Math.tanh(speed)) / 2;
    const episode = {};
    const recurrentSample = (capturedAt, speed, history = [], sampleEpisode = episode) => {
        const sample = makeInput(config, 0, false);
        const historyTimes = new Array(drivingMemorySlots(config)).fill(null);
        sample.features.fill(0);
        sample.features[drivingMemoryInputs(config)] = speed;
        for (const [time, historicalSpeed] of history) {
            const age = capturedAt - time;
            const bin = Math.ceil(age / config.memory.intervalMs) - 1;
            const offset = bin * slotSize;
            historyTimes[bin] = time;
            sample.features[offset] = 1;
            sample.features[offset + 1] = age / config.memory.windowMs;
            sample.features[offset + 2] = historicalSpeed;
            sample.cameras[bin] = sample.cameras[historyTimes.length];
        }
        return { ...sample, recurrent: { episode: sampleEpisode, capturedAt, historyTimes } };
    };
    const liveOutput = async sample => {
        const { pass, outputs } = await spatial.gpu.forward(sample);
        spatial.gpu.release(pass);
        return outputs[0];
    };
    const checkReplay = async (sample, initial, speeds, label, recorded = false) => {
        const pass = recorded ? spatial.gpu.forwardPass(sample, false, 'recorded') : spatial.gpu.replay(sample);
        try {
            assert(pass.steps === speeds.length, `${label} processes the full replay sequence`);
            const [seed] = await spatial.gpu.download([pass.states]);
            const [output] = await spatial.gpu.download([pass.outputs]);
            close([seed, output], [initial, speeds.reduce(advanceHidden, initial)], label);
            if (recorded) {
                const features = await spatial.gpu.download([pass.features]);
                close(speeds.map((_speed, i) => features[i * drivingPolicyInputs(config) + 1]), speeds.map(() => 0),
                    'Dense recorded history uses each timestep as a current observation');
            }
        } finally { spatial.gpu.release(pass); }
    };
    const firstHidden = advanceHidden(0, 0.2);
    close([await liveOutput(recurrentSample(1000, 0.2))], [firstHidden], 'First live observation starts from zero');
    close([await liveOutput(recurrentSample(1000, -0.8))], [firstHidden], 'Duplicates preserve the existing live hidden state');
    const nextHidden = advanceHidden(firstHidden, 0.4);
    close([await liveOutput(recurrentSample(1300, 0.4, [[1000, 0.7]]))], [nextHidden],
        'Live observations carry state and skip history already represented by that state');
    await liveOutput(recurrentSample(1000, -0.6));
    await checkReplay(recurrentSample(1300, 0.3, [[1000, 0.6]]), 0, [0.6, 0.3],
        'Replay does not use a snapshot at the oldest sample');
    await checkReplay(recurrentSample(1900, 0.8, [[1600, 0.7]]), nextHidden, [0.7, 0.8],
        'Replay uses the nearest state before the oldest sample');
    const finalHidden = [-0.4, 0.9].reduce(advanceHidden, nextHidden);
    close([await liveOutput(recurrentSample(1900, 0.9, [[1600, -0.4]]))], [finalHidden],
        'Training replay and out-of-order observations leave live state intact and new history is processed');
    close([await liveOutput(recurrentSample(1900, 0.9, [], {}))], [advanceHidden(0, 0.9)],
        'A distinct episode clears the previous live state');
    const recordedEpisode = {};
    spatial.gpu.observeDemonstration(recurrentSample(1000, 0.2, [], recordedEpisode));
    spatial.gpu.observeDemonstration(recurrentSample(1300, 0.4, [[1000, 0.7]], recordedEpisode));
    spatial.gpu.observeDemonstration(recurrentSample(1300, -0.4, [[1000, 0.8]], recordedEpisode));
    close([await liveOutput(recurrentSample(1300, -0.8, [], recordedEpisode))], [nextHidden],
        'Recorded observations build the same recurrent state as sequential live inference');
    await checkReplay(recurrentSample(1900, 0.8, [[1600, 0.7]], recordedEpisode), nextHidden, [0.7, 0.8],
        'Delayed recorded learning backpropagates from an earlier recurrent seed');
    spatial.gpu.observeDemonstration(recurrentSample(1400, -0.9, [], recordedEpisode));
    const dense = recurrentSample(1600, 0.8, [[1300, 0.1]], recordedEpisode);
    dense.recordedHistory = [[1400, 0.65], [1500, -0.18]].map(([capturedAt, speed]) => {
        const sample = recurrentSample(capturedAt, speed, [], recordedEpisode);
        return { capturedAt, features: sample.features.slice(drivingMemoryInputs(config)), camera: sample.cameras.at(-1) };
    });
    await checkReplay(dense, nextHidden, [0.65, -0.18, 0.8],
        'Recorded replay restores the seed strictly before the first dense frame and ignores the sparse bins', true);
    const emptyDense = { ...dense, recordedHistory: [] };
    await checkReplay(emptyDense, advanceHidden(nextHidden, -0.9), [0.8],
        'An empty dense history still uses current observation and the previous recurrent seed', true);
    spatial.dispose();

    for (const cameraHistory of [true, false]) {
        const c = cloneConfig(config);
        if (!cameraHistory) c.memory.features = ['speed'];
        const policy = await create(c), sample = makeInput(c);
        sample.cameras[sample.cameras.length - 1] = { width: 384, height: 216, rgb: new Uint8Array(384 * 216 * 3) };
        const start = await policy.checkpointAsync();
        await policy.learnDemonstrationAsync(sample, action);
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
        const gear = 2;
        if (variant === 'large-logits') {
            document.model.outputWeights[gearRow(gear)].fill(0);
            document.model.outputWeights[gearRow(gear)][c.network.hiddenUnits] = 1000;
        }
        const policy = await create(c, document), sample = makeInput(c), start = await policy.checkpointAsync();
        if (variant === 'large-logits') assert((await predict(policy, sample)).plan[0].gearProbabilities[gearIndex(gear)] === 1, 'Large logits remain stable');
        await policy.learnDemonstrationAsync(sample, { pedal: -1, steering: -1, gear });
        if (variant === 'zero-rates') {
            const decision = await policy.decideAsync(sample, true);
            await policy.learnAsync(decision, 2, sample, 0.2, credits(decision)); policy.releaseDecision(decision);
            await policy.learnPlannedActionsAsync(sample, [{ step: 1, action }]);
        }
        const finish = await policy.checkpointAsync(); finite(finish);
        if (variant === 'zero-rates') {
            for (const key of ['cameraWeights', 'cameraProjectionWeights', 'gruWeights', 'outputWeights']) close(start.model[key].flat(2), finish.model[key].flat(2), 'Zero learning rates', 1e-12);
            assert(finish.model.steps === 2, 'Count completed transitions with zero rates');
        } else close(start.model.outputWeights.slice(DRIVING_ANALOG_AXES, actionOutputs).flat(), finish.model.outputWeights.slice(DRIVING_ANALOG_AXES, actionOutputs).flat(), 'Stable large logits');
        policy.dispose();
    }
    // Independent CPU forward oracle and numerical derivatives exercise both
    // normalizers through the production GPU path, including image history.
    const normalizedConfig = cloneConfig();
    normalizedConfig.observation.features = normalizedConfig.memory.features = ['camera', 'speed', 'centerline'];
    normalizedConfig.memory.windowMs = 600;
    normalizedConfig.network.cameraLayers.forEach(layer => { layer.channels = 2; });
    normalizedConfig.network.hiddenUnits = 4;
    normalizedConfig.network.cameraProjectionUnits = 16;
    normalizedConfig.actions.planTimesSeconds = [0, 0.1, 0.2];
    normalizedConfig.learning.demonstrationRate = 0.05;
    normalizedConfig.learning.gradientNormLimit = 1e6;
    const normH = normalizedConfig.network.hiddenUnits, normInputs = drivingPolicyInputs(normalizedConfig);
    const normalized = await create(normalizedConfig);
    const normalizedSeed = await normalized.checkpointAsync();
    normalizedSeed.model.outputWeights.forEach((row, r) => row.forEach((_value, c) => {
        if (c < normH) row[c] = Math.sin(1 + r * 0.7 + c) * 0.2;
    }));
    // Give the reset gate a visible effect and start both encoders at the same weights.
    for (let unit = 0; unit < normH; unit++) {
        const row = 2 * normH + unit, column = normInputs + (unit + 1) % normH;
        normalizedSeed.model.gruWeights[row][column] = 0.25;
        normalizedSeed.model.sac.targetEncoder.gruWeights[row][column] = 0.25;
    }
    normalized.dispose();
    const normalizedPolicy = await create(normalizedConfig, normalizedSeed);
    const normalizedInput = makeInput(normalizedConfig, 2, true, true);
    const norm = (values, gain, bias) => {
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
        return values.map((value, i) => (value - mean) / Math.sqrt(variance + normalizedConfig.network.layerNormEpsilon) * gain[i] + bias[i]);
    };
    const normalizedPass = await normalizedPolicy.gpu.forward(normalizedInput);
    const normalizedFeatures = await normalizedPolicy.gpu.download([normalizedPass.pass.features]);
    for (const camera of normalizedPass.pass.cameras) {
        const raw = await normalizedPolicy.gpu.download([camera.activations[normalizedConfig.network.cameraLayers.length + 2]]);
        const offset = new Uint32Array((await normalizedPolicy.gpu.download([camera.offset])).buffer)[0];
        close(Array.from(normalizedFeatures.slice(offset, offset + raw.length)),
            norm(Array.from(raw), ...normalizedSeed.model.cameraNormWeights), 'Camera LayerNorm CPU oracle', 0.00003);
    }
    const normGate = (sums, gate) => norm(sums,
        ...normalizedSeed.model.gruNormWeights.map(row => row.slice(gate * normH, (gate + 1) * normH)));
    const [updateRows, resetRows, candidateRows] = [0, 1, 2].map(gate =>
        normalizedSeed.model.gruWeights.slice(gate * normH, (gate + 1) * normH));
    const dot = (row, vector) => vector.reduce((sum, value, i) => sum + value * row[i], 0);
    const gpuStates = await normalizedPolicy.gpu.download([normalizedPass.pass.states]);
    let oracleHidden = new Array(normH).fill(0);
    for (let t = 0; t < normalizedPass.pass.steps; t++) {
        const features = normalizedFeatures.slice(t * normInputs, (t + 1) * normInputs);
        const vector = [...features, ...oracleHidden, 1];
        const update = normGate(updateRows.map(row => dot(row, vector)), 0).map(sigmoid);
        const reset = normGate(resetRows.map(row => dot(row, vector)), 1).map(sigmoid);
        const recurrence = candidateRows.map(row => dot(row.slice(normInputs, normInputs + normH), oracleHidden));
        const candidate = normGate(candidateRows.map((row, unit) =>
            dot(row, features) + row[normInputs + normH] + reset[unit] * recurrence[unit]), 2).map(Math.tanh);
        oracleHidden = oracleHidden.map((value, unit) => update[unit] * value + (1 - update[unit]) * candidate[unit]);
        const stateOffset = (t + 1) * 5 * normH;
        close(gpuStates.slice(stateOffset, stateOffset + 5 * normH),
            [...oracleHidden, ...update, ...reset, ...candidate, ...recurrence], `Normalized GRU CPU oracle at step ${t}`);
    }
    const expectedOutputs = normalizedSeed.model.outputWeights.map(row => dot(row, [...oracleHidden, 1]));
    close(normalizedPass.outputs, expectedOutputs, 'Normalized GRU output CPU oracle');
    normalizedPolicy.gpu.release(normalizedPass.pass);
    const normalizedTarget = { pedal: Math.tanh(0.4), steering: Math.tanh(-0.2), gear: 1 };
    const normalizedGearlessTarget = { step: 0, action: normalizedTarget, gearWeight: 0 };
    const normalizedObjective = async () => {
        const decision = await predict(normalizedPolicy, normalizedInput, reverseOnly);
        return -0.5 * decision.plan[0].means.reduce((sum, value, i) => sum + (value - [0.4, -0.2][i]) ** 2, 0);
    };
    const normBefore = await normalizedPolicy.checkpointAsync(), normCoordinates = [];
    const addCoordinates = (key, rows, buffer, rowIndices, columns, layer) => {
        for (const row of rowIndices) for (const column of columns) normCoordinates.push({ key, layer, row, column,
            buffer, index: row * rows[0].length + column, value: rows[row][column] });
    };
    normBefore.model.cameraWeights.forEach((rows, layer) => addCoordinates('cameraWeights', rows,
        normalizedPolicy.gpu.cameraWeights[layer], [0, 1], [0, rows[0].length - 1], layer));
    addCoordinates('cameraProjectionWeights', normBefore.model.cameraProjectionWeights, normalizedPolicy.gpu.cameraProjectionWeights,
        [0, 15], [0, 17, normBefore.model.cameraProjectionWeights[0].length - 1]);
    for (let gate = 0; gate < 3; gate++) addCoordinates('gruWeights', normBefore.model.gruWeights,
        normalizedPolicy.gpu.gruWeights, [gate * normH], [2, normInputs, normInputs + 1, normInputs + normH]);
    for (const key of ['cameraNormWeights', 'gruNormWeights']) addCoordinates(key, normBefore.model[key], normalizedPolicy.gpu[key],
        [0, 1], key === 'cameraNormWeights' ? [0, 7, 15] : [0, normH, 2 * normH]);
    addCoordinates('outputWeights', normBefore.model.outputWeights, normalizedPolicy.gpu.outputWeights, [0, 1, 2], [0, normH]);
    for (const coordinate of normCoordinates) {
        const epsilon = 0.002;
        normalizedPolicy.gpu.device.queue.writeBuffer(coordinate.buffer, coordinate.index * 4, new Float32Array([coordinate.value + epsilon]));
        const plus = await normalizedObjective();
        normalizedPolicy.gpu.device.queue.writeBuffer(coordinate.buffer, coordinate.index * 4, new Float32Array([coordinate.value - epsilon]));
        const minus = await normalizedObjective();
        normalizedPolicy.gpu.device.queue.writeBuffer(coordinate.buffer, coordinate.index * 4, new Float32Array([coordinate.value]));
        coordinate.derivative = (plus - minus) / (2 * epsilon);
    }
    await withoutReadbacks('Normalized finite-difference update', () => normalizedPolicy.learnPlannedActionsAsync(normalizedInput, [normalizedGearlessTarget]));
    const normAfter = await normalizedPolicy.checkpointAsync();
    let normalizedDerivativeError = 0;
    for (const coordinate of normCoordinates) {
        const rows = coordinate.key === 'cameraWeights' ? normAfter.model.cameraWeights[coordinate.layer] : normAfter.model[coordinate.key];
        const actual = (rows[coordinate.row][coordinate.column] - coordinate.value) / normalizedConfig.learning.demonstrationRate;
        const error = Math.abs(actual - coordinate.derivative);
        assert(error < 0.0002 + Math.abs(coordinate.derivative) * 0.03,
            `Normalized finite differences ${coordinate.key}/${coordinate.layer ?? ''}/${coordinate.row}/${coordinate.column}: ${actual} vs ${coordinate.derivative}`);
        normalizedDerivativeError = Math.max(normalizedDerivativeError, error);
    }
    for (const key of ['cameraWeights', 'cameraProjectionWeights', 'cameraNormWeights', 'gruWeights', 'gruNormWeights', 'outputWeights']) {
        assert(normCoordinates.some(coordinate => coordinate.key === key && Math.abs(coordinate.derivative) > 1e-5),
            `Normalized derivative fixture exercises ${key}`);
        assert(changed(normBefore.model[key], normAfter.model[key]), `Normalized update trains ${key}`);
    }
    assert(normCoordinates.some(({ key, column, derivative }) => key === 'gruNormWeights'
        && column === normH && Math.abs(derivative) > 1e-5), 'Normalized finite differences exercise the reset gate');
    assert(normCoordinates.some(({ key, row, column, derivative }) => key === 'gruWeights'
        && row === 2 * normH && column === normInputs + 1 && Math.abs(derivative) > 1e-5),
        'Normalized finite differences exercise cross-unit candidate recurrence');
    const normalizedReload = await create(normalizedConfig, JSON.parse(JSON.stringify(normAfter)));
    close((await predict(normalizedPolicy, normalizedInput)).plan.flatMap(point => point.means),
        (await predict(normalizedReload, normalizedInput)).plan.flatMap(point => point.means), 'Normalized checkpoint reload');
    normalizedReload.dispose(); normalizedPolicy.dispose();

    const allWeightKeys = ['cameraWeights', 'cameraProjectionWeights', 'cameraNormWeights', 'gruWeights', 'gruNormWeights', 'outputWeights'];
    const flattenWeights = checkpoint => [...allWeightKeys.flatMap(key => checkpoint.model[key]?.flat(2) ?? []),
        ...checkpoint.model.sac.criticWeights.flat(), ...checkpoint.model.sac.logStds];
    const normOf = values => Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));
    const deltaFrom = (before, after) => {
        const initial = flattenWeights(before);
        return flattenWeights(after).map((value, i) => value - initial[i]);
    };
    const clippingSeed = JSON.parse(JSON.stringify(normBefore));
    clippingSeed.configuration.learning.gradientNormLimit = 0.05;
    const clipped = await create(clippingSeed.configuration, clippingSeed);
    await withoutReadbacks('Global clipped update', () => clipped.learnPlannedActionsAsync(normalizedInput, [normalizedGearlessTarget]));
    const clippedAfter = await clipped.checkpointAsync(), clippedDiagnostics = await clipped.gpu.trainingDiagnostics();
    const unclippedDelta = deltaFrom(normBefore, normAfter), clippedDelta = deltaFrom(clippingSeed, clippedAfter);
    const clipScale = Math.min(1, clippingSeed.configuration.learning.gradientNormLimit * normalizedConfig.learning.demonstrationRate / normOf(unclippedDelta));
    assert(clipScale < 0.9, 'Global clipping fixture exceeds the threshold');
    close(clippedDelta, unclippedDelta.map(value => value * clipScale), 'Global clipping preserves full-parameter update direction', 0.0000003);
    assert(normOf(clippedDelta) <= normalizedConfig.learning.demonstrationRate * clippingSeed.configuration.learning.gradientNormLimit + 0.000002,
        'Global clipping bounds the full accumulated parameter update');
    close([clippedDiagnostics.gradientNorm, clippedDiagnostics.clipScale, clippedDiagnostics.updateNorm],
        [normOf(unclippedDelta) / normalizedConfig.learning.demonstrationRate, clipScale, normOf(clippedDelta)], 'Clipping diagnostics', 0.00003);
    clipped.dispose();

    // SAC replay batches average their contributions before one global clip.
    const sequenceClipped = await create(clippingSeed.configuration, clippingSeed), clipSequence = [];
    for (let i = 0; i < 4; i++) {
        const decision = await sequenceClipped.decideAsync(normalizedInput, true, reverseOnly);
        sequenceClipped.releaseDecision(decision);
        clipSequence.push({ startedAt: i * 100, endedAt: (i + 1) * 100, reward: 3,
            input: normalizedInput, next: normalizedInput, nextGears: reverseOnly,
            samples: [{ decision, credits: credits(decision), rewardShare: 1 }] });
    }
    await withoutReadbacks('Clipped sequence commit', () => sequenceClipped.trainBatchAsync(() => sequenceClipped.learnSequenceAsync(clipSequence, normalizedInput)));
    const clippedSequenceResult = await sequenceClipped.checkpointAsync(), sequenceDiagnostics = await sequenceClipped.gpu.trainingDiagnostics();
    const sequenceUnclipped = await create(normalizedConfig, normBefore);
    await sequenceUnclipped.trainBatchAsync(() => sequenceUnclipped.learnSequenceAsync(clipSequence, normalizedInput));
    const sequenceDelta = deltaFrom(normBefore, await sequenceUnclipped.checkpointAsync());
    const sequenceNorm = normOf(sequenceDelta) / normalizedConfig.learning.demonstrationRate;
    const sequenceClipScale = Math.min(1, clippingSeed.configuration.learning.gradientNormLimit / sequenceNorm);
    assert(sequenceClipScale < 0.9, 'Sequence clipping fixture exceeds the threshold');
    close(deltaFrom(clippingSeed, clippedSequenceResult), sequenceDelta.map(value => value * sequenceClipScale),
        'Sequence clipping preserves the average gradient direction', 0.0000003);
    close([sequenceDiagnostics.gradientNorm, sequenceDiagnostics.clipScale], [sequenceNorm, sequenceClipScale / sequenceDiagnostics.samples],
        'Sequence diagnostics report the average gradient norm', 0.00003);
    assert(normOf(deltaFrom(clippingSeed, clippedSequenceResult)) <= normalizedConfig.learning.demonstrationRate * clippingSeed.configuration.learning.gradientNormLimit + 0.000002,
        'Sequence global clip covers all accumulated replays');
    assert(sequenceDiagnostics.samples === 4, 'Sequence diagnostics count all accumulated replays');
    sequenceClipped.dispose(); sequenceUnclipped.dispose();

    // Equal plan targets add their full trunk gradients and retain each head's contribution.
    const planSumSeed = JSON.parse(JSON.stringify(normBefore));
    for (let step = 1; step < normalizedConfig.actions.planTimesSeconds.length; step++) {
        for (let row = 0; row < actionOutputs; row++) planSumSeed.model.outputWeights[step * actionOutputs + row] = planSumSeed.model.outputWeights[row].slice();
    }
    const oneHead = await create(normalizedConfig, planSumSeed), allHeads = await create(normalizedConfig, planSumSeed);
    const equalTargets = normalizedConfig.actions.planTimesSeconds.map((_time, step) => ({ ...normalizedGearlessTarget, step }));
    await oneHead.learnPlannedActionsAsync(normalizedInput, equalTargets.slice(0, 1));
    await allHeads.learnPlannedActionsAsync(normalizedInput, equalTargets);
    const oneHeadAfter = await oneHead.checkpointAsync(), allHeadsAfter = await allHeads.checkpointAsync();
    for (const key of allWeightKeys.filter(key => key !== 'outputWeights')) {
        const initial = planSumSeed.model[key].flat(2);
        close(allHeadsAfter.model[key].flat(2).map((value, i) => value - initial[i]),
            oneHeadAfter.model[key].flat(2).map((value, i) => (value - initial[i]) * equalTargets.length),
            `Plan targets sum trunk gradients/${key}`, 0.0000003);
    }
    for (const { step } of equalTargets) {
        close(allHeadsAfter.model.outputWeights.slice(step * actionOutputs, (step + 1) * actionOutputs).flat(),
            oneHeadAfter.model.outputWeights.slice(0, actionOutputs).flat(),
            `Plan loss preserves each active head contribution/${step}`, 0.0000003);
    }
    oneHead.dispose(); allHeads.dispose();

    const invalidGradientPolicy = await create(normalizedConfig, normBefore);
    const invalidGpu = invalidGradientPolicy.gpu, invalidParameters = invalidGpu.parameters();
    for (const invalid of [Infinity, -Infinity, NaN, 0]) {
        const clear = invalidGpu.device.createCommandEncoder();
        invalidParameters.forEach(({ gradient }) => clear.clearBuffer(gradient));
        invalidGpu.device.queue.submit([clear.finish()]);
        invalidGpu.device.queue.writeBuffer(invalidParameters[0].gradient, 0, new Float32Array([invalid]));
        if (invalid !== 0) invalidGpu.device.queue.writeBuffer(invalidParameters[1].gradient, 0, new Float32Array([1]));
        const apply = invalidGpu.device.createCommandEncoder();
        invalidGpu.applyGradients(apply, 1); invalidGpu.device.queue.submit([apply.finish()]);
        compare(await invalidGradientPolicy.checkpointAsync(), normBefore, 'Nonfinite or zero global gradient preserves every parameter');
        const diagnostic = await invalidGpu.trainingDiagnostics();
        assert(diagnostic.updateNorm === 0 && diagnostic.finite === (invalid === 0), 'Invalid and zero gradient diagnostics');
        assert(diagnostic.clipScale === (invalid === 0 ? 1 : 0), 'Nonfinite gradient rejects the entire commit');
    }
    invalidGradientPolicy.dispose();
    const normalization = { finiteDifferences: normCoordinates.length, maxDerivativeError: normalizedDerivativeError,
        clipping: clippedDiagnostics, sequenceClipping: sequenceDiagnostics };
    const constantSeed = JSON.parse(JSON.stringify(normBefore));
    constantSeed.configuration.learning.weightLimit = 2000;
    for (const key of ['cameraProjectionWeights', 'gruWeights']) constantSeed.model[key].forEach(row => {
        row.fill(0); row[row.length - 1] = 1000;
    });
    const constantPolicy = await create(constantSeed.configuration, constantSeed);
    const constantPass = await constantPolicy.gpu.forward(normalizedInput);
    const constantFeatures = await constantPolicy.gpu.download([constantPass.pass.features]);
    for (let t = 0; t < constantPass.pass.steps; t++) close(Array.from(constantFeatures.slice(t * normInputs + 2,
        t * normInputs + 2 + normalizedConfig.network.cameraProjectionUnits)), new Array(16).fill(0), 'Zero-variance camera normalization');
    close(constantPass.outputs, constantSeed.model.outputWeights.map(row => row[normH]), 'Large constant gate sums normalize before nonlinearities');
    constantPolicy.gpu.release(constantPass.pass);
    await withoutReadbacks('Zero-variance normalization backward', () => constantPolicy.learnPlannedActionsAsync(normalizedInput, [normalizedGearlessTarget]));
    const constantAfter = await constantPolicy.checkpointAsync(); finite(constantAfter);
    assert(changed(constantSeed.model.gruNormWeights, constantAfter.model.gruNormWeights), 'Zero-variance gates retain trainable bias gradients');
    constantPolicy.dispose();

    const productionConfig = cloneConfig(), productionPolicy = await create(productionConfig);
    const productionSample = makeInput(productionConfig), productionBefore = await productionPolicy.checkpointAsync();
    await withoutReadbacks('Default 256-feature normalized training', () => productionPolicy.learnDemonstrationAsync(productionSample, action));
    const productionAfter = await productionPolicy.checkpointAsync();
    assertTrained(productionBefore, productionAfter, 'Default normalized model');
    for (const key of ['cameraNormWeights', 'gruNormWeights']) assert(changed(productionBefore.model[key], productionAfter.model[key]), `Default normalized model trains ${key}`);
    normalization.defaultClipping = await productionPolicy.gpu.trainingDiagnostics();
    productionPolicy.dispose();
    const soak = [];
    if (soakIterations) for (const variant of ['default', 'saturated-long-history']) {
        const c = cloneConfig();
        if (variant === 'saturated-long-history') {
            c.network.cameraLayerNorm = c.network.gruLayerNorm = false;
            c.network.cameraLayers.forEach(layer => { layer.channels = 1; });
            c.network.cameraProjectionUnits = 32; c.network.hiddenUnits = 2;
            c.observation.features = ['camera', 'speed', 'centerline']; c.memory.features = ['camera', 'speed', 'centerline'];
            c.memory.windowMs = 30000;
        }
        let policy = await create(c);
        if (variant === 'saturated-long-history') {
            const saturated = await policy.checkpointAsync(); policy.dispose();
            // Amplify camera features beyond the tanh overflow threshold. With
            // update gates at 0.5, the long history drives hidden values to ±1.
            saturated.model.cameraWeights.forEach(rows => rows.forEach(row => row.fill(0.25)));
            saturated.model.cameraProjectionWeights.forEach(row => row.fill(1));
            saturated.model.gruWeights.forEach(row => row.fill(0));
            saturated.model.gruWeights[2][2] = -1;
            saturated.model.gruWeights[3][2] = 1;
            saturated.model.gruWeights[4][2] = 1;
            saturated.model.gruWeights[5][2] = -1;
            policy = await create(c, saturated);
        }
        const samples = Array.from({ length: 6 }, (_, i) => makeInput(c, i, true, i % 2 === 1 && variant === 'default'));
        const initial = await policy.checkpointAsync(), started = performance.now();
        let checkpointReloads = 0, checkedDecisions = 0, maxCameraFeature = 0, maxHidden = 0;
        const checkDecision = async (sample) => {
            const decision = await policy.decideAsync(sample, true);
            assert([decision.value, ...decision.plan.flatMap(step => [...step.means, ...step.latent, ...step.gearProbabilities,
                step.action.pedal, step.action.steering])].every(Number.isFinite), `${variant}: finite decision`);
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
                        maxHidden = Math.max(maxHidden, Math.abs(states[t * 5 * c.network.hiddenUnits + unit]));
                    }
                }
                assert(maxCameraFeature > 100 && maxHidden > 0.99 && maxHidden <= 1,
                    'Stress fixture reaches large camera features and bounded saturated GRU hidden states');
            }
            const target = { pedal: iteration % 3 ? 1 : -1,
                steering: Math.sin(iteration * 0.1), gear: drivingGearFromIndex(iteration % DRIVING_GEAR_COUNT) };
            try {
                await withoutReadbacks('Soak training', () => policy.trainBatchAsync(async () => {
                    await policy.learnAsync(decision, (iteration % 21) - 10, next, 0.2, credits(decision), iteration % 17 === 0);
                    await policy.learnDemonstrationAsync(sample, target);
                    await policy.learnPlannedActionsAsync(sample, [{ step: c.actions.planTimesSeconds.length - 1, action: target }]);
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
            maxCameraFeature, maxHidden, seconds: (performance.now() - started) / 1000 });
        policy.dispose();
    }
    assert(errors.length === 0, errors.join('\n'));
    return { adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture },
        checks: 'GPU-only prediction and learning, zero intermediate readbacks, CNN/projection/GRU/output finite differences, padded borders, spatial ordering and adaptive pooling gradients, all layers and gates train, historical camera gradients, masks, terminal/clipped transitions, delayed plans, checkpoint reload/synchronization, custom layouts/rates, allocation-free warmed ticks, reused versus fresh workspaces, bounded buffers and device loss passed',
        activationCases, sequenceChecks: 'Episode-scoped GRU state carry, duplicate and out-of-order observations, seeded training replay, adjacent-state replay, clipped double-Q Bellman targets, elapsed discount, terminal masks, Polyak targets, current-policy actor samples, learned entropy, and interrupted-batch rollback passed',
        nativeRecorded: { batches: nativeBatches, analogErrorBefore, analogErrorAfter,
            gears: nativeAfter.map(({ action }) => action.gear) },
        normalization, soak, finiteDifferences: { weightsChecked: coordinates.length, maxDerivativeError }, results };
}

app.whenReady().then(async () => {
    let window;
    try {
        await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
        window = new BrowserWindow({ show: false, webPreferences: { contextIsolation: true, nodeIntegration: false, backgroundThrottling: false } });
        window.webContents.on('console-message', (_event, _level, message) => process.stderr.write(message + '\n'));
        await window.loadURL(`http://127.0.0.1:${server.address().port}/`);
        const run = await window.webContents.executeJavaScript(`(async () => { ${bundle}
            try { return { result: await (${checks.toString()})(${process.argv.includes('--soak') ? 1000 : 0}, ${process.argv.includes('--target-gears')}) }; }
            catch (error) { return { error: error.stack || String(error) }; }
        })()`);
        if (run.error) process.stderr.write(run.error + '\n');
        else process.stdout.write(JSON.stringify(run.result, null, 2) + '\n');
        app.exit(run.error ? 1 : 0);
    } catch (error) { process.stderr.write(String(error.stack || error) + '\n'); app.exit(1); }
    finally { server.close(); window?.destroy(); }
});
