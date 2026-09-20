const fs = require('fs/promises');
const os = require('os');
const path = require('path');
const { createHash } = require('crypto');
const { createTrackVisionModelCache, registerTrackVisionModels } = require('../../../electron/track-vision-models');

const weights = Buffer.from([1, 2, 3, 4]);
const model = {
    id: 'a'.repeat(24), name: 'track', task: 'segment', classNames: ['curb', 'track'],
    sha256: createHash('sha256').update(weights).digest('hex'), sizeBytes: weights.length,
};
let directory;
let exporter;
const makeCache = () => createTrackVisionModelCache({ directory: () => directory, exportModel: exporter });
const folder = () => path.join(directory, `${model.id}-${model.sha256}`);

beforeEach(async () => {
    directory = await fs.mkdtemp(path.join(os.tmpdir(), 'acla-vision-cache-test-'));
    exporter = jest.fn(async (filename) => fs.writeFile(filename.replace('.pt', '.onnx'), Buffer.from([5, 6])));
});
afterEach(async () => {
    if (path.dirname(directory) !== path.resolve(os.tmpdir()) || !path.basename(directory).startsWith('acla-vision-cache-test-')) {
        throw new Error('Unexpected cache test directory');
    }
    await fs.rm(directory, { recursive: true, force: true });
});

it('returns a cache miss without exporting or creating weights', async () => {
    await expect(makeCache().prepare(model)).resolves.toBeNull();
    expect(exporter).not.toHaveBeenCalled();
});

it('saves verified weights and labels and reuses ONNX across cache instances', async () => {
    await expect(makeCache().prepare(model, weights)).resolves.toEqual(Buffer.from([5, 6]));
    expect(await fs.readFile(path.join(folder(), 'weights.pt'))).toEqual(weights);
    expect(exporter).toHaveBeenCalledWith(path.join(folder(), 'weights.pt'), ['curb', 'track']);
    await expect(makeCache().prepare(model)).resolves.toEqual(Buffer.from([5, 6]));
    expect(exporter).toHaveBeenCalledTimes(1);
});

it('rejects truncated or mismatched downloads before exporting', async () => {
    await expect(makeCache().prepare(model, Buffer.from([4, 3, 2, 1]))).rejects.toThrow('checksum mismatch');
    await expect(makeCache().prepare(model, Buffer.from([1]))).rejects.toThrow('checksum mismatch');
    expect(exporter).not.toHaveBeenCalled();
    expect(await fs.readdir(directory)).toEqual([]);
});

it('rebuilds damaged ONNX from cached backend weights', async () => {
    const cache = makeCache();
    await cache.prepare(model, weights);
    await fs.writeFile(path.join(folder(), 'weights.onnx'), 'corrupt');
    await expect(cache.prepare(model)).resolves.toEqual(Buffer.from([5, 6]));
    expect(exporter).toHaveBeenCalledTimes(2);
});

it('retains downloaded weights after a failed export so retry does not download again', async () => {
    exporter.mockRejectedValueOnce(new Error('Export failed'));
    const cache = makeCache();
    await expect(cache.prepare(model, weights)).rejects.toThrow('Export failed');
    await expect(cache.prepare(model)).resolves.toEqual(Buffer.from([5, 6]));
    expect(exporter).toHaveBeenCalledTimes(2);
});

it('serializes concurrent prepares and exports one copy', async () => {
    const cache = makeCache();
    await Promise.all([cache.prepare(model, weights), cache.prepare(model, weights)]);
    expect(exporter).toHaveBeenCalledTimes(1);
});

it('treats a new backend version as a cache miss and rebuilds changed label metadata', async () => {
    const cache = makeCache();
    await cache.prepare(model, weights);
    await expect(cache.prepare({ ...model, sha256: 'b'.repeat(64) })).resolves.toBeNull();
    await cache.prepare({ ...model, classNames: ['track', 'curb'] });
    expect(exporter).toHaveBeenLastCalledWith(path.join(folder(), 'weights.pt'), ['track', 'curb']);
    expect(exporter).toHaveBeenCalledTimes(2);
});

it('rejects path traversal and non-main renderer requests', async () => {
    await expect(makeCache().prepare({ ...model, id: '../outside' }, weights)).rejects.toThrow('Invalid Track Vision');
    const ipcMain = { handle: jest.fn() };
    registerTrackVisionModels({ app: {}, ipcMain, getMainWindow: () => null });
    const handler = ipcMain.handle.mock.calls[0][1];
    expect(() => handler({}, model, weights)).toThrow('main workspace');
    expect(exporter).not.toHaveBeenCalled();
});
