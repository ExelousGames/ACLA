// Run with: npx electron scripts/check-vision-gpu.cjs
// Exercises the actual WGSL and ONNX IO binding in a hidden Chromium window.
const { app, BrowserWindow } = require('electron');
const fs = require('fs');
const path = require('path');
const http = require('http');
const ts = require('typescript');
const root = path.resolve(__dirname, '..');
const source = path.join(root, 'src/views/live-session/track-vision');
const files = ['vision-gpu-input', 'vision-gpu-depth', 'vision-gpu-segments', 'vision-gpu-queue', 'vision-decoding', 'vision-config', 'vision-colors', 'yolo-segmentation'];
const modules = files.map((name) => `${JSON.stringify('./' + name)}: function(require, exports) {\n${ts.transpileModule(
    fs.readFileSync(path.join(source, name + '.ts'), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.CommonJS } },
).outputText}\n}`).join(',');
const bundle = `const modules = {${modules}}, cache = {}; function require(id) {
    if (!cache[id]) { cache[id] = {}; modules[id](require, cache[id]); } return cache[id]; }`;
const server = http.createServer((req, res) => {
    const relative = decodeURIComponent(new URL(req.url, 'http://localhost').pathname).slice(1);
    const filename = path.resolve(root, relative);
    if (relative === '') { res.end('<!doctype html><title>Vision GPU checks</title>'); return; }
    if (!filename.startsWith(root + path.sep)) { res.writeHead(403).end(); return; }
    res.setHeader('Content-Type', filename.endsWith('.mjs') ? 'text/javascript' : filename.endsWith('.wasm') ? 'application/wasm' : 'application/octet-stream');
    const stream = fs.createReadStream(filename);
    stream.on('error', () => res.writeHead(404).end()); stream.pipe(res);
});
async function checks() {
    const { VisionGpuInput, imageGeometry } = require('./vision-gpu-input');
    const { GpuDepth, readDepthValues } = require('./vision-gpu-depth');
    const { GpuSegments } = require('./vision-gpu-segments');
    const { decodeSegments } = require('./vision-decoding');
    const assert = (condition, text) => { if (!condition) throw new Error(text); };
    const close = (actual, expected, label) => assert(Math.abs(actual - expected) < 0.00001, `${label}: ${actual} != ${expected}`);
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    assert(adapter, 'A WebGPU adapter is required');
    const device = await adapter.requestDevice();
    const errors = [];
    device.addEventListener('uncapturederror', (event) => errors.push(event.error.message));
    device.pushErrorScope('validation');
    const segmentChecks = [];
    const checkSegments = async (name, predictions, prototypes, threshold = 0.5, classCount = 2) => {
        const upload = (value) => {
            const buffer = device.createBuffer({ size: value.data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
            device.queue.writeBuffer(buffer, 0, value.data);
            return { dims: value.dims, type: 'float32', location: 'gpu-buffer', gpuBuffer: buffer };
        };
        const p = upload(predictions), m = upload(prototypes);
        const gpu = new GpuSegments(device, p, m, 64, threshold, classCount);
        try {
            const expected = decodeSegments(predictions, prototypes, 64, threshold, classCount);
            const actual = await gpu.read();
            const validation = await device.popErrorScope();
            assert(!validation, validation?.message);
            device.pushErrorScope('validation');
            assert(actual.length === expected.instances.length, `${name}: instance count ${actual.length} != ${expected.instances.length}`);
            actual.forEach((item, i) => {
                const reference = expected.instances[i];
                assert(item.classId === reference.classId, `${name}: class/order ${i}`);
                close(item.confidence, reference.confidence, `${name}: confidence ${i}`);
                item.box.forEach((value, c) => close(value, reference.box[c], `${name}: box ${i}/${c}`));
                assert(item.mask.every((value, pixel) => value === reference.mask[pixel]), `${name}: mask ${i}`);
            });
            segmentChecks.push({ name, instances: actual.length });
        } finally { await gpu.dispose(); p.gpuBuffer.destroy(); m.gpuBuffer.destroy(); }
    };
    const prototypes = { dims: [1, 2, 7, 9], data: Float32Array.from({ length: 126 }, (_, i) => (i % 7 - 3) / 4) };
    const rows = [
        [0, 0, 64, 64, 0.9, 0, 1, -1],
        [0, 0, 64, 64, 0.9, 0, -1, 1],
        [0, 0, 64, 64, 0.85, 1, 0.5, 1],
        [8, 16, 40, 48, 0.8, 0, 1, 0.5],
        [16, 8, 48, 40, 0.7, 1, -1, -0.5],
        [0, 0, 64, 64, 0.6, 0, 0, 0],
        [0, 0, 64, 64, 0.1, 0, 1, 1],
        [0, 0, 64, 64, 0.99, 1, NaN, 1],
        [0, 0, 64, 64, NaN, 0, 1, 1],
        [20, 20, 10, 10, 0.98, 0, 1, 1],
        [-8, -8, 80, 80, 1, 1, 1, 1],
    ];
    const e2e = { dims: [1, rows.length, 8], data: new Float32Array(rows.flat()) };
    await checkSegments('end-to-end, ties, crops, invalid values, odd mask dimensions', e2e, prototypes);
    await checkSegments('float32 confidence threshold boundary', e2e, prototypes, 0.9);
    await checkSegments('empty detections', { dims: e2e.dims, data: e2e.data.map((value, i) => i % 8 === 4 ? 0.1 : value) }, prototypes);
    const rawRows = rows.map(([x1, y1, x2, y2, score, label, ...coefficients]) =>
        [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, label === 0 ? score : 0, label === 1 ? score : 0, ...coefficients]);
    const rawPredictions = { dims: [1, 8, rawRows.length], data: Float32Array.from({ length: 8 * rawRows.length }, (_, i) => rawRows[i % rawRows.length][Math.floor(i / rawRows.length)]) };
    await checkSegments('raw class-aware NMS and invalid coefficients', rawPredictions, prototypes);
    const many = Array.from({ length: 350 }, (_, i) => [0, 0, 64, 64, i < 300 ? 0.9 : 0.8, 0, 1, -1]);
    await checkSegments('end-to-end 30-instance cap', { dims: [1, 350, 8], data: new Float32Array(many.flat()) }, prototypes);
    const rawMany = many.map(([x1, y1, x2, y2, score, , a, b], i) => [32, 32, 64, 64, i < 300 ? score : 0, i < 300 ? 0 : score, a, b]);
    await checkSegments('raw top-300 cap and complete suppression', { dims: [1, 8, 350],
        data: Float32Array.from({ length: 8 * 350 }, (_, i) => rawMany[i % 350][Math.floor(i / 350)]) }, prototypes);
    const download = async (buffer, count) => {
        const staging = device.createBuffer({ size: count * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        try {
            const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(buffer, 0, staging, 0, count * 4);
            device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ);
            return new Float32Array(staging.getMappedRange()).slice();
        } finally { staging.destroy(); }
    };
    const runtime = { Tensor: { fromGpuBuffer: (gpuBuffer, options) => ({ gpuBuffer, ...options }) } };
    const frame = document.createElement('canvas'); frame.width = 8; frame.height = 4;
    const paint = frame.getContext('2d'); paint.fillStyle = '#ff8040'; paint.fillRect(0, 0, 8, 4);
    const preprocessor = new VisionGpuInput(device, runtime, 8);
    const geometry = imageGeometry(frame, 8, true);
    preprocessor.prepare(frame, geometry, true);
    const input = await download(preprocessor.tensor.gpuBuffer, 192);
    const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
    [1, 128 / 255, 64 / 255].forEach((channel, i) => {
        close(input[i * 64 + 24], (channel - mean[i]) / std[i], 'RGB/CHW normalization');
        close(input[i * 64], (114 / 255 - mean[i]) / std[i], 'Letterbox padding');
    });
    const masked = { ...geometry, region: { width: 2, height: 2, mask: new Uint8Array([0, 1, 1, 1]) } };
    preprocessor.prepare(frame, masked, true);
    const excluded = await download(preprocessor.tensor.gpuBuffer, 192);
    close(excluded[16], (114 / 255 - mean[0]) / std[0], 'GPU input mask');
    close(excluded[20], (1 - mean[0]) / std[0], 'Retained input');
    let reads = 0, disposals = 0;
    const raw = new Float32Array([99, 99, 99, 99, 0, 1, 3, 7, 15, 31, -1, NaN, 99, 99, 99, 99]);
    const buffer = device.createBuffer({ size: raw.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(buffer, 0, raw);
    const output = { dims: [1, 4, 4], type: 'float32', location: 'gpu-buffer', gpuBuffer: buffer,
        getData: async () => { reads++; return download(buffer, 16); }, dispose: () => { disposals++; buffer.destroy(); } };
    const gpu = new GpuDepth(device, output, geometry);
    const canvas = document.createElement('canvas'); canvas.width = 4; canvas.height = 2;
    const context = canvas.getContext('2d');
    gpu.draw(context, { x: 0, y: 1, width: 4, height: 2 }, 4, 2);
    assert(reads === 0, 'Rendering must not download depth');
    const range = await gpu.readRange(); close(range.near, 1 / 32, 'GPU minimum'); close(range.far, 1, 'GPU maximum');
    const colors = context.getImageData(0, 0, 4, 2).data;
    assert([59, 130, 246, 255].every((value, i) => colors[i] === value), `Far color: ${Array.from(colors.slice(0, 4))}`);
    assert([239, 68, 68, 255].every((value, i) => colors[20 + i] === value), 'Near color');
    const result = { task: 'depth', width: 4, height: 4, gpu };
    const values = await Promise.all([readDepthValues(result), readDepthValues(result)]);
    assert(reads === 1 && values[0] === values[1], 'Readback must be cached');
    assert(values[0][0] === 0 && values[0][10] === 0 && values[0][11] === 0, 'Padding and invalid values');
    await gpu.dispose(); await gpu.dispose(); assert(disposals === 1, 'Dispose exactly once');
    preprocessor.dispose();
    await device.queue.onSubmittedWorkDone();
    const validation = await device.popErrorScope(); assert(!validation, validation?.message);
    assert(!errors.length, errors.join('\n'));
    // ONNX 1.22 creates its device during session initialization. Borrow that exact device.
    const ort = await import('/node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs');
    ort.env.wasm.wasmPaths = '/public/vision-runtime/'; ort.env.wasm.numThreads = 1;
    const session = await ort.InferenceSession.create('/public/vision-models/depth-anything-v2-small-252.onnx', {
        executionProviders: ['webgpu'], preferredOutputLocation: 'gpu-buffer',
    });
    const modelDevice = await ort.env.webgpu.device;
    modelDevice.addEventListener('uncapturederror', (event) => errors.push(event.error.message));
    const modelInput = new VisionGpuInput(modelDevice, ort, 252);
    const tensor = modelInput.prepare(frame, imageGeometry(frame, 252, true), true);
    const outputs = await session.run({ [session.inputNames[0]]: tensor });
    const modelDepth = new GpuDepth(modelDevice, outputs[session.outputNames[0]], geometry);
    modelDepth.draw(context, { x: 0, y: modelDepth.height / 4, width: modelDepth.width, height: modelDepth.height / 2 }, 4, 2);
    const modelRange = await modelDepth.readRange();
    assert(modelRange.near > 0 && modelRange.far >= modelRange.near, 'ONNX GPU depth range');
    const modelValues = await modelDepth.read();
    const validValues = modelValues.filter((value) => value > 0);
    close(Math.min(...validValues), modelRange.near, 'ONNX GPU/CPU minimum');
    close(Math.max(...validValues), modelRange.far, 'ONNX GPU/CPU maximum');
    modelDepth.draw(context, { x: 0, y: modelDepth.height / 4, width: modelDepth.width, height: modelDepth.height / 2 }, 4, 2);
    await modelDevice.queue.onSubmittedWorkDone();
    await modelDepth.dispose(); modelInput.dispose(); await session.release();
    assert(!errors.length, errors.join('\n'));
    return { adapter: adapter.info, checks: 'GPU segmentation parity, preprocessing, masking, reduction, coloring, lazy readback, disposal, and actual ONNX GPU IO binding passed', segmentChecks, modelRange };
}
app.whenReady().then(async () => {
    let window;
    try {
        await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
        window = new BrowserWindow({ show: false, webPreferences: { contextIsolation: true, nodeIntegration: false } });
        window.webContents.on('console-message', (_event, _level, message) => process.stderr.write(message + '\n'));
        await window.loadURL(`http://127.0.0.1:${server.address().port}/`);
        const result = await window.webContents.executeJavaScript(`(async () => { ${bundle}\nreturn await (${checks.toString()})(); })()`);
        process.stdout.write(JSON.stringify(result, null, 2) + '\n');
        app.exit(0);
    } catch (error) { process.stderr.write(String(error.stack || error) + '\n'); app.exit(1); }
    finally { server.close(); window?.destroy(); }
});
