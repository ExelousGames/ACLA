const fs = require('fs');
const path = require('path');

// Ship GPU and CPU runtime assets with the app so inference never needs a CDN.
const source = path.dirname(require.resolve('onnxruntime-web'));
// ONNX Runtime 1.22 hardcodes this limit instead of exposing a WebGPU option.
// Batch 64 dispatches per submission to reduce gaps between GPU submissions.
let runtimeChanged = false;
for (const name of ['ort.webgpu.bundle.min.mjs', 'ort.webgpu.min.mjs', 'ort.webgpu.min.js']) {
    const file = path.join(source, name);
    const runtime = fs.readFileSync(file, 'utf8');
    const assignments = runtime.match(/\bthis\.maxDispatchNumber=(?:16|64);/g);
    if (assignments?.length !== 1) {
        throw new Error(`Unexpected ONNX WebGPU dispatch limit in ${name}; review the runtime patch after upgrading ONNX.`);
    }
    const patched = runtime.replace('this.maxDispatchNumber=16;', 'this.maxDispatchNumber=64;');
    if (patched !== runtime) {
        fs.writeFileSync(file, patched);
        runtimeChanged = true;
    }
}
// Webpack assumes installed packages are unchanged at the same package version.
if (runtimeChanged) {
    for (const mode of ['development', 'production']) {
        for (const name of ['index.pack', 'index.pack.old']) {
            fs.rmSync(path.resolve(__dirname, `../node_modules/.cache/default-${mode}`, name), { force: true });
        }
    }
}
const target = path.resolve(__dirname, '../public/vision-runtime');
fs.mkdirSync(target, { recursive: true });
for (const name of [
    'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm',
    'ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm',
]) {
    fs.copyFileSync(path.join(source, name), path.join(target, name));
}
console.log('Prepared local track-vision runtime.');
