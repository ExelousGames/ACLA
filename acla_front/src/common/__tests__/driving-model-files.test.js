const fs = require('fs/promises');
const os = require('os');
const path = require('path');
const { registerDrivingModelFiles } = require('../../../electron/driving-model-files');

const document = (steps = 0) => JSON.stringify({ format: 'kestrel-self-driving-model', algorithm: 'online-actor-critic',
    configuration: { aggression: 50 }, model: { steps, lstmWeights: [[0.1]], outputWeights: [[0.2]] } });
let directory;
beforeEach(async () => { directory = await fs.mkdtemp(path.join(os.tmpdir(), 'acla-driving-files-test-')); });
afterEach(async () => {
    jest.restoreAllMocks();
    if (path.dirname(directory) !== path.resolve(os.tmpdir()) || !path.basename(directory).startsWith('acla-driving-files-test-')) {
        throw new Error('Unexpected model test directory');
    }
    await fs.rm(directory, { recursive: true, force: true });
});

function setup() {
    const handlers = {};
    const window = { isDestroyed: () => false, webContents: { mainFrame: {} } };
    const event = { sender: window.webContents, senderFrame: window.webContents.mainFrame };
    const filePath = path.join(directory, 'model.json');
    const dialog = { showSaveDialog: jest.fn(async () => ({ canceled: false, filePath })),
        showOpenDialog: jest.fn(async () => ({ canceled: false, filePaths: [filePath] })) };
    registerDrivingModelFiles({ app: { getPath: () => directory }, dialog,
        ipcMain: { handle: (key, handler) => { handlers[key] = handler; } }, getMainWindow: () => window });
    return { handlers, event, dialog, filePath, call: (name, ...args) => handlers[`driving-model-${name}`](event, ...args) };
}

it('creates a complete model at the selected location, replaces it on save and reopens it', async () => {
    const run = setup();
    expect(await run.call('current')).toBeNull();
    expect(await run.call('choose', 'new')).toBe(run.filePath);
    expect(run.dialog.showSaveDialog).toHaveBeenCalledWith(expect.anything(), expect.objectContaining({
        title: 'Save new Kestrel self driving model',
        filters: [{ name: 'Kestrel self driving model', extensions: ['json'] }],
        defaultPath: path.join(directory, 'kestrel-self-driving-model.json'),
    }));
    await run.call('write', run.filePath, document());
    await run.call('select', run.filePath);
    expect(await run.call('current')).toBe(run.filePath);
    await run.call('write', run.filePath, document(50));
    expect(await fs.readFile(run.filePath, 'utf8')).toBe(document(50));
    expect(await fs.readdir(directory)).toEqual(['model.json']);
    const restarted = setup();
    expect(await restarted.call('current')).toBeNull();
    expect(await restarted.call('choose', 'open')).toBe(run.filePath);
    expect(restarted.dialog.showOpenDialog).toHaveBeenCalledWith(expect.anything(), expect.objectContaining({
        title: 'Open Kestrel self driving model',
        filters: [{ name: 'Kestrel self driving model', extensions: ['json'] }],
    }));
    expect(await restarted.call('read', run.filePath)).toBe(document(50));
});

it.each(['new', 'open'])('cancels %s without changing the current file or writing anything', async (mode) => {
    const run = setup();
    await run.call('choose', 'new');
    await run.call('write', run.filePath, document(10));
    await run.call('select', run.filePath);
    run.dialog.showSaveDialog.mockResolvedValue({ canceled: true });
    run.dialog.showOpenDialog.mockResolvedValue({ canceled: true, filePaths: [] });
    expect(await run.call('choose', mode)).toBeNull();
    expect(await run.call('current')).toBe(run.filePath);
    expect(await fs.readFile(run.filePath, 'utf8')).toBe(document(10));
});

it('restricts every operation to the main window frame and picker-authorized files', async () => {
    const run = setup();
    for (const name of ['current', 'choose', 'select', 'read', 'write']) {
        for (const event of [{ sender: {} }, { ...run.event, senderFrame: {} }]) {
            await expect(run.handlers[`driving-model-${name}`](event, run.filePath, document())).rejects.toThrow(/main workspace/);
        }
    }
    for (const name of ['read', 'write', 'select']) {
        await expect(run.call(name, run.filePath, document())).rejects.toThrow(/Choose a model file/);
    }
    expect(await fs.readdir(directory)).toEqual([]);
});

it('serializes overlapping saves and reads so the last checkpoint wins', async () => {
    const run = setup();
    await run.call('choose', 'new');
    const writes = [10, 50, 100].map((steps) => run.call('write', run.filePath, document(steps)));
    const read = run.call('read', run.filePath);
    await Promise.all(writes);
    expect(await read).toBe(document(100));
    expect(await fs.readdir(directory)).toEqual(['model.json']);
});

it('leaves the last complete file intact after a failed replacement and permits retry', async () => {
    const run = setup();
    await run.call('choose', 'new');
    await run.call('write', run.filePath, document(10));
    jest.spyOn(fs, 'rename').mockRejectedValueOnce(new Error('Disk unavailable'));
    await expect(run.call('write', run.filePath, document(50))).rejects.toThrow('Disk unavailable');
    expect(await fs.readFile(run.filePath, 'utf8')).toBe(document(10));
    expect(await fs.readdir(directory)).toEqual(['model.json']);
    await run.call('write', run.filePath, document(100));
    expect(await run.call('read', run.filePath)).toBe(document(100));
});

it.each(['{broken', '{}', JSON.stringify({ version: 13, weights: [1] }), ' '.repeat(32 * 1024 * 1024 + 1)])(
    'rejects invalid or oversized writes without damaging a saved model %#', async (json) => {
        const run = setup();
        await run.call('choose', 'new');
        await run.call('write', run.filePath, document(10));
        await expect(run.call('write', run.filePath, json)).rejects.toThrow();
        expect(await fs.readFile(run.filePath, 'utf8')).toBe(document(10));
    });

it('reports missing, corrupt and oversized files without selecting them as the active model', async () => {
    const run = setup();
    await run.call('choose', 'open');
    await expect(run.call('read', run.filePath)).rejects.toThrow();
    await fs.writeFile(run.filePath, '{broken');
    await expect(run.call('read', run.filePath)).rejects.toThrow();
    await fs.truncate(run.filePath, 32 * 1024 * 1024 + 1);
    await expect(run.call('read', run.filePath)).rejects.toThrow(/too large/);
    expect(await run.call('current')).toBeNull();
});
