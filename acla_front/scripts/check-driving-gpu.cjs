// Run: npx electron scripts/check-driving-gpu.cjs
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

async function checks() {
    const { DrivingPolicy } = require('views/ai-driving-training/driving-policy');
    const { cloneConfig, drivingCameraInputs, drivingMemorySlots, drivingObservationInputs, drivingPolicyInputs } = require('views/ai-driving-training/driving-model-config');
    const { drivingMemoryInputs, drivingMemorySlotInputs } = require('views/ai-driving-training/driving-memory');
    const assert = (condition, label) => { if (!condition) throw new Error(label); };
    const errors = [];
    let readbacks = 0;
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
        for (const key of ['cameraWeights', 'lstmWeights', 'outputWeights']) {
            close(a.model[key].flat(2), b.model[key].flat(2), `${label}/${key}`);
        }
    };
    const changed = (a, b) => a.flat(2).some((value, i) => value !== b.flat(2)[i]);
    const finite = (checkpoint) => {
        for (const key of ['cameraWeights', 'lstmWeights', 'outputWeights']) {
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
        const rgb = Uint8Array.from({ length: 768 * 768 * 3 }, (_, i) => (i * 17 + frame * 31) % 256);
        const camera = { width: 768, height: 768, rgb };
        const cameras = Array.from({ length: bins + 1 }, (_, slot) =>
            slot < bins && (!history || !config.memory.features.includes('camera')) ? null : camera);
        if (distinct) cameras.forEach((image, slot) => {
            if (image && slot !== bins) cameras[slot] = { ...camera, rgb: Uint8Array.from(rgb, (value, i) => (value + slot * 19 + i % 7) % 256) };
        });
        for (let i = 0; i < bins; i++) { features[i * slots] = history ? 1 : 0; features[i * slots + 1] = (i + 1) / bins; }
        return { features, cameras };
    };
    const shifts = { up: true, down: false }, action = { throttle: 0.7, brake: 0.12, steering: -0.3, shiftUp: true };
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
        close(initial.model.outputWeights.slice(6).flat(), demonstration.model.outputWeights.slice(6).flat(), 'Only the demonstrated slot learns');
        await withoutReadbacks('distinct historical images', () => gpu.learnDemonstrationAsync(makeInput(config, 5, true, true), action, shifts));
        gpu.random = () => 0.99;
        for (const [reward, terminal, available] of [[0.8, false, shifts], [-20, false, { up: false, down: true }],
            [20, true, { up: false, down: false }], [-0.6, true, { up: true, down: true }]]) {
            const before = await gpu.checkpointAsync(), decision = await gpu.decideAsync(input, true, available);
            const following = terminal ? { features: [], cameras: [] } : next;
            await withoutReadbacks('actor-critic', () => gpu.learnAsync(decision, reward, following, 0.35, terminal));
            gpu.releaseDecision(decision);
            const after = await gpu.checkpointAsync();
            assertTrained(before, after, 'actor-critic');
            close(before.model.outputWeights.slice(6, -1).flat(), after.model.outputWeights.slice(6, -1).flat(), 'Unexecuted plans receive no reward');
            if (!available.up && !available.down) close(before.model.outputWeights.slice(3, 6).flat(), after.model.outputWeights.slice(3, 6).flat(), 'Forced hold receives no shift gradient');
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
        close(beforePlan.model.outputWeights.slice(0, finalSlot * 6).flat(), afterPlan.model.outputWeights.slice(0, finalSlot * 6).flat(), 'Earlier slots receive no delayed gradient');
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
            for (let iteration = 0; iteration < 2; iteration++) {
                const start = performance.now(), beforeReadbacks = readbacks;
                await withoutReadbacks('Full training tick', () => gpu.trainBatchAsync(async () => {
                    await gpu.learnDemonstrationAsync(samples[0], action, shifts);
                    for (const sample of samples.slice(1)) await gpu.learnPlannedActionsAsync(sample, targets);
                }));
                await predict(gpu, next);
                assert(readbacks === beforeReadbacks + 1, 'Only the control decision reads back during a training tick');
                timing.push(performance.now() - start);
                assert(gpu.gpu.buffers.size === bufferCount, 'GPU buffers remain bounded across ticks');
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
    config.learning.demonstrationRate = 0.05;
    const seedPolicy = new DrivingPolicy({ game: 'acc', car: 'GT3', map: 'monza' }, drivingObservationInputs(config), () => 0.6, config);
    const seed = seedPolicy.checkpoint();
    seed.model.cameraWeights.forEach((rows, l) => rows.forEach(row => { row.fill([0.005, 0.03, 0.05][l]); row[row.length - 1] = 0.05; }));
    seed.model.outputWeights.forEach(row => { row[0] = 0.1; row[1] = -0.05; });
    const gpu = await create(config, seed), input = makeInput(config, 1, true, true);
    const target = { throttle: 1 / (1 + Math.exp(1.5)), brake: 1 / (1 + Math.exp(2.5)), steering: Math.tanh(0.15), shiftUp: true };
    const objective = async () => {
        const decision = await predict(gpu, input);
        const means = decision.plan[0].means;
        return -0.5 * means.reduce((sum, mean, i) => sum + (mean - [-1.5, -2.5, 0.15][i]) ** 2, 0)
            + Math.log(decision.plan[0].shiftProbabilities[1]);
    };
    const before = await gpu.checkpointAsync(), coordinates = [];
    config.network.cameraLayers.forEach((_, l) => {
        const row = before.model.cameraWeights[l][0];
        for (const column of [0, row.length - 1]) coordinates.push({ key: 'cameraWeights', layer: l, row: 0, column,
            buffer: gpu.gpu.cameraWeights[l], index: column, value: row[column] });
    });
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
    spatialSeed.model.cameraWeights.forEach(rows => rows.forEach(row => { row.fill(0); row[0] = 1; }));
    const spatial = await create(config, spatialSeed);
    const cameraFeatures = async (pixel) => {
        const sample = makeInput(config, 0, false), rgb = new Uint8Array(768 * 768 * 3);
        rgb[pixel] = 255;
        sample.cameras[sample.cameras.length - 1] = { width: 768, height: 768, rgb };
        const { pass } = await spatial.gpu.forward(sample);
        try { return await spatial.gpu.download([pass.features]); }
        finally { spatial.gpu.release(pass); }
    };
    assert((await cameraFeatures(0))[2] === 1 && (await cameraFeatures(3))[2] === 0,
        'Convolutions distinguish equal-average images with different pixel positions');
    spatialSeed.model.cameraWeights.forEach((rows, l) => {
        rows.forEach(row => { row.fill(0); row[row.length - 2] = 1; });
        spatial.gpu.device.queue.writeBuffer(spatial.gpu.cameraWeights[l], 0, new Float32Array(rows.flat()));
    });
    assert((await cameraFeatures(768 * 768 * 3 - 1))[2 + drivingCameraInputs(config) - 1] === 1,
        'The bottom-right RGB pixel reaches the final CNN layer');
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
        sample.cameras[sample.cameras.length - 1] = { width: 768, height: 768, rgb: new Uint8Array(768 * 768 * 3) };
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
        if (variant === 'large-logits') { document.model.outputWeights[4].fill(0); document.model.outputWeights[4][c.network.hiddenUnits] = 1000; }
        const policy = await create(c, document), sample = makeInput(c), start = await policy.checkpointAsync();
        const available = { up: false, down: false };
        if (variant === 'large-logits') assert((await predict(policy, sample)).plan[0].shiftProbabilities[1] === 1, 'Large logits remain stable');
        await policy.learnDemonstrationAsync(sample, { throttle: 0, brake: 1, steering: -1 }, available);
        if (variant === 'zero-rates') {
            const decision = await policy.decideAsync(sample, true);
            await policy.learnAsync(decision, 2, sample, 0.2); policy.releaseDecision(decision);
            await policy.learnPlannedActionsAsync(sample, [{ step: 1, action, shifts }]);
        }
        const finish = await policy.checkpointAsync(); finite(finish);
        if (variant === 'zero-rates') {
            for (const key of ['cameraWeights', 'lstmWeights', 'outputWeights']) close(start.model[key].flat(2), finish.model[key].flat(2), 'Zero learning rates', 1e-12);
            assert(finish.model.steps === 2, 'Count completed transitions with zero rates');
        } else close(start.model.outputWeights.slice(3, 6).flat(), finish.model.outputWeights.slice(3, 6).flat(), 'Masked large logits');
        policy.dispose();
    }
    assert(errors.length === 0, errors.join('\n'));
    return { adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture },
        checks: 'GPU-only prediction and learning, zero intermediate readbacks, CNN/LSTM/output finite differences, all layers and gates train, historical camera gradients, masks, terminal/clipped transitions, delayed plans, checkpoint reload/synchronization, custom layouts/rates, bounded buffers and device loss passed',
        finiteDifferences: { weightsChecked: coordinates.length, maxDerivativeError }, results };
}

app.whenReady().then(async () => {
    let window;
    try {
        await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
        window = new BrowserWindow({ show: false, webPreferences: { contextIsolation: true, nodeIntegration: false, backgroundThrottling: false } });
        window.webContents.on('console-message', (_event, _level, message) => process.stderr.write(message + '\n'));
        await window.loadURL(`http://127.0.0.1:${server.address().port}/`);
        const result = await window.webContents.executeJavaScript(`(async () => { ${bundle}\nreturn await (${checks.toString()})(); })()`);
        process.stdout.write(JSON.stringify(result, null, 2) + '\n');
        app.exit(0);
    } catch (error) { process.stderr.write(String(error.stack || error) + '\n'); app.exit(1); }
    finally { server.close(); window?.destroy(); }
});
