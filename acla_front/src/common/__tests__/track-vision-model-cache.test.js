const fs = require('fs/promises');
const os = require('os');
const path = require('path');
const { createHash } = require('crypto');
const { createTrackVisionModelCache, registerTrackVisionModels } = require('../../../electron/track-vision-models');

const weights = Buffer.from([1, 2, 3, 4]);
const model = {
    id: 'a'.repeat(24), name: 'track-position', task: 'position', filename: 'best.onnx', classNames: ['curb', 'track'],
    sha256: createHash('sha256').update(weights).digest('hex'), sizeBytes: weights.length,
};
let directory;
const makeCache = () => createTrackVisionModelCache({ directory: () => directory });
const folder = () => path.join(directory, `${model.id}-${model.sha256}`);

beforeEach(async () => {
    directory = await fs.mkdtemp(path.join(os.tmpdir(), 'acla-vision-cache-test-'));
});

afterEach(async () => {
    if (path.dirname(directory) !== path.resolve(os.tmpdir()) || !path.basename(directory).startsWith('acla-vision-cache-test-')) {
        throw new Error('Unexpected cache test directory');
    }
    await fs.rm(directory, { recursive: true, force: true });
});

it('returns a cache miss without creating weights', async () => {
    await expect(makeCache().prepare(model)).resolves.toBeNull();
    expect(await fs.readdir(directory)).toEqual([]);
});

it.each(['segment', 'position'])('saves and reuses the verified %s ONNX download unchanged', async (task) => {
    const metadata = { ...model, task };
    await expect(makeCache().prepare(metadata, weights)).resolves.toEqual(weights);
    expect(await fs.readFile(path.join(folder(), 'weights.onnx'))).toEqual(weights);
    expect(await fs.readdir(folder())).toEqual(['weights.onnx']);
    await expect(makeCache().prepare(metadata)).resolves.toEqual(weights);
});

it.each([
    ['ArrayBuffer', () => Uint8Array.from(weights).buffer],
    ['typed array slice', () => Uint8Array.from([0, ...weights, 0]).subarray(1, 5)],
    ['DataView', () => new DataView(Uint8Array.from([0, ...weights, 0]).buffer, 1, 4)],
])('preserves the bytes of a %s download', async (_name, download) => {
    await expect(makeCache().prepare(model, download())).resolves.toEqual(weights);
});

it.each([Buffer.from([4, 3, 2, 1]), Buffer.from([1]), Buffer.alloc(0)])(
    'rejects corrupted or truncated downloads before saving', async (download) => {
        await expect(makeCache().prepare(model, download)).rejects.toThrow('checksum mismatch');
        expect(await fs.readdir(directory)).toEqual([]);
    });

it('rejects non-binary downloads', async () => {
    await expect(makeCache().prepare(model, 'not binary')).rejects.toThrow('Invalid Track Vision model download');
    expect(await fs.readdir(directory)).toEqual([]);
});

it.each([Buffer.from([4, 3, 2, 1]), Buffer.from([1]), Buffer.alloc(0)])(
    'requests a new download when cached ONNX fails checksum or size verification', async (corrupt) => {
        const cache = makeCache();
        await cache.prepare(model, weights);
        await fs.writeFile(path.join(folder(), 'weights.onnx'), corrupt);
        await expect(cache.prepare(model)).resolves.toBeNull();
        await expect(cache.prepare(model, weights)).resolves.toEqual(weights);
        await expect(makeCache().prepare(model)).resolves.toEqual(weights);
    });

it('ignores legacy checkpoints and locally converted ONNX caches', async () => {
    await fs.mkdir(folder(), { recursive: true });
    await fs.writeFile(path.join(folder(), 'weights.pt'), weights);
    await fs.writeFile(path.join(folder(), 'weights-512.onnx'), weights);
    await fs.writeFile(path.join(folder(), 'manifest-512.json'), JSON.stringify({ version: 4, model, inputSize: 512,
        onnxSha256: model.sha256 }));
    await expect(makeCache().prepare(model)).resolves.toBeNull();
    await expect(makeCache().prepare(model, weights)).resolves.toEqual(weights);
    expect(await fs.readFile(path.join(folder(), 'weights.onnx'))).toEqual(weights);
});

it('serializes simultaneous cache misses and reuses the completed download', async () => {
    const cache = makeCache();
    const write = jest.spyOn(fs, 'writeFile');
    try {
        await expect(Promise.all([cache.prepare(model, weights), cache.prepare(model, weights), cache.prepare(model)]))
            .resolves.toEqual([weights, weights, weights]);
        expect(write).toHaveBeenCalledTimes(1);
    } finally { write.mockRestore(); }
});

it('allows a queued valid download after a failed verification', async () => {
    const cache = makeCache();
    const failed = expect(cache.prepare(model, Buffer.from([9]))).rejects.toThrow('checksum mismatch');
    await expect(cache.prepare(model, weights)).resolves.toEqual(weights);
    await failed;
});

it('keeps backend versions separate by both model ID and checksum', async () => {
    const cache = makeCache();
    await cache.prepare(model, weights);
    await expect(cache.prepare({ ...model, id: 'b'.repeat(24) })).resolves.toBeNull();
    await expect(cache.prepare({ ...model, sha256: 'b'.repeat(64) })).resolves.toBeNull();
    await expect(cache.prepare(model)).resolves.toEqual(weights);
});

it.each([
    { filename: 'best.pt' }, { filename: 'best.onnx.pt' }, { filename: undefined }, { task: 'detect' },
    { id: '../outside' }, { sha256: 'invalid' }, { sizeBytes: 0 }, { sizeBytes: 512 * 1024 * 1024 + 1 },
    { classNames: [] }, { classNames: ['track', 'track'] },
])('rejects unsupported artifacts and invalid metadata (%j)', async (invalid) => {
    await expect(makeCache().prepare({ ...model, ...invalid }, weights)).rejects.toThrow('Invalid Track Vision model metadata');
    expect(await fs.readdir(directory)).toEqual([]);
});

it('accepts case-insensitive ONNX filenames', async () => {
    await expect(makeCache().prepare({ ...model, filename: 'BEST.ONNX' }, weights)).resolves.toEqual(weights);
});

it('only serves ONNX models to the main workspace frame', async () => {
    const ipcMain = { handle: jest.fn() };
    const mainWindow = { isDestroyed: () => false, webContents: { mainFrame: {} } };
    registerTrackVisionModels({ app: { getPath: () => directory }, ipcMain, getMainWindow: () => mainWindow });
    const handler = ipcMain.handle.mock.calls[0][1];
    expect(() => handler({}, model, weights)).toThrow('main workspace');
    expect(() => handler({ sender: mainWindow.webContents, senderFrame: {} }, model, weights)).toThrow('main workspace');
    await expect(handler({ sender: mainWindow.webContents, senderFrame: mainWindow.webContents.mainFrame }, model, weights))
        .resolves.toEqual(weights);
});
