const fs = require('fs');
const path = require('path');

// Ship GPU and CPU runtime assets with the app so inference never needs a CDN.
const source = path.dirname(require.resolve('onnxruntime-web'));
const target = path.resolve(__dirname, '../public/vision-runtime');
fs.mkdirSync(target, { recursive: true });
for (const name of [
    'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm',
    'ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm',
]) {
    fs.copyFileSync(path.join(source, name), path.join(target, name));
}
console.log('Prepared local track-vision runtime.');
for (const model of ['yolo26n-sem', 'yolo26n-depth', 'yolo11n-seg']) {
    if (!fs.existsSync(path.resolve(__dirname, '../public/vision-models', `${model}.onnx`))) {
        console.warn(`${model} is not prepared. Run npm run setup:vision-models to enable its detector.`);
    }
}
