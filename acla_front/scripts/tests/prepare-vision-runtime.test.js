const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const scriptDirectory = path.resolve(__dirname, '..');
const script = fs.readFileSync(path.join(scriptDirectory, 'prepare-vision-runtime.js'), 'utf8');
const runtimeDirectory = path.resolve(scriptDirectory, '../node_modules/onnxruntime-web/dist');
const bundles = ['ort.webgpu.bundle.min.mjs', 'ort.webgpu.min.mjs', 'ort.webgpu.min.js'];
const assets = ['ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm',
    'ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm'];

function fixture(contents = 'this.maxDispatchNumber=16;this.pendingDispatchNumber=0;') {
    const files = new Map(bundles.map((name) => [path.join(runtimeDirectory, name), contents]));
    const writes = [], copies = [], removals = [];
    const mockRequire = Object.assign((name) => ({
        fs: {
            readFileSync: (file) => files.get(file),
            writeFileSync: (file, value) => { writes.push(file); files.set(file, value); },
            mkdirSync() {},
            rmSync: (file) => removals.push(file),
            copyFileSync: (source, target) => copies.push([source, target]),
        },
        path,
    }[name]), { resolve: () => path.join(runtimeDirectory, 'ort.node.min.js') });
    return { files, writes, copies, removals, run: () => vm.runInNewContext(script, {
        __dirname: scriptDirectory, require: mockRequire, console: { log() {} },
    }) };
}

test('preparation patches every exported WebGPU bundle and still ships the WASM assets', () => {
    const runtime = fixture();
    runtime.run();
    for (const name of bundles) {
        assert.equal(runtime.files.get(path.join(runtimeDirectory, name)),
            'this.maxDispatchNumber=64;this.pendingDispatchNumber=0;');
    }
    assert.deepEqual(runtime.copies, assets.map((name) => [path.join(runtimeDirectory, name),
        path.resolve(scriptDirectory, '../public/vision-runtime', name)]));
    assert.deepEqual(runtime.removals, ['development', 'production'].flatMap((mode) =>
        ['index.pack', 'index.pack.old'].map((name) =>
            path.resolve(scriptDirectory, `../node_modules/.cache/default-${mode}`, name))));
});

test('repeated preparation leaves already patched bundles untouched', () => {
    const runtime = fixture();
    runtime.run();
    runtime.run();
    assert.equal(runtime.writes.length, bundles.length);
    assert.equal(runtime.removals.length, 4);
});

for (const contents of ['this.maxDispatchNumber=32;', 'this.changedDispatchLimit=16;',
    'this.maxDispatchNumber=16;this.maxDispatchNumber=16;']) {
    test(`preparation rejects an unexpected runtime layout: ${contents}`, () => {
        const runtime = fixture(contents);
        assert.throws(runtime.run, /Unexpected ONNX WebGPU dispatch limit/);
        assert.equal(runtime.writes.length, 0);
        assert.equal(runtime.copies.length, 0);
    });
}
