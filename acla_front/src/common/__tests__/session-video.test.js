/** @jest-environment node */
const fs = require('fs/promises');
const os = require('os');
const path = require('path');
const { EventEmitter } = require('events');
const { registerSessionVideo } = require('../../../electron/session-video');

let directory;
let handlers;
let event;
let telemetry;

beforeEach(async () => {
    directory = await fs.mkdtemp(path.join(os.tmpdir(), 'acla-video-test-'));
    telemetry = path.join(directory, 'acc.jsonl');
    await fs.writeFile(telemetry, '{}\n');
    handlers = {};
    const contents = new EventEmitter();
    contents.mainFrame = {};
    event = { sender: contents, senderFrame: contents.mainFrame };
    registerSessionVideo({ ipcMain: { handle: (name, callback) => { handlers[name] = callback; } },
        getMainWindow: () => ({ isDestroyed: () => false, webContents: contents }),
        getRecordingDirectory: () => directory, shell: { showItemInFolder: jest.fn() } });
});
afterEach(async () => { await fs.rm(directory, { recursive: true, force: true }); });
const call = (name, ...args) => handlers[`session-video-${name}`](event, ...args);

it('rejects a second video while recording and after saving without changing the first', async () => {
    const { filePath } = await call('create', telemetry);
    await expect(call('create', telemetry)).rejects.toThrow('Only one video is allowed per session');
    await call('append', filePath, Uint8Array.from([1, 2]));
    await call('finish', filePath);
    await expect(call('create', telemetry)).rejects.toThrow('Only one video is allowed per session');
    expect(await call('read', filePath, 0)).toEqual({ data: 'AQI=', bytesRead: 2 });
    expect(await call('list', telemetry)).toHaveLength(1);
});

it('rejects concurrent video creation for the same session', async () => {
    const results = await Promise.allSettled([call('create', telemetry), call('create', telemetry)]);
    expect(results.filter(result => result.status === 'fulfilled')).toHaveLength(1);
    expect(results.find(result => result.status === 'rejected').reason.message).toContain('Only one video');
    expect((await fs.readdir(directory)).filter(name => name.endsWith('.mp4'))).toHaveLength(1);
});

it('recognizes an existing video on disk and allows a separate session to record', async () => {
    await fs.writeFile(`${telemetry}.video-1-abc.mp4`, Uint8Array.from([1]));
    await expect(call('create', telemetry)).rejects.toThrow('Only one video is allowed per session');
    const otherTelemetry = path.join(directory, 'other.jsonl');
    await fs.writeFile(otherTelemetry, '{}\n');
    await expect(call('create', otherTelemetry)).resolves.toHaveProperty('filePath');
});

it('persists ordered bytes and discovers clips again for the same telemetry file', async () => {
    const { filePath } = await call('create', telemetry);
    expect(filePath).toMatch(/\.mp4$/);
    await call('append', filePath, Uint8Array.from([1, 2]));
    await call('append', filePath, Uint8Array.from([3, 4]));
    await expect(call('read', filePath, 0)).rejects.toThrow('not ready');
    await call('finish', filePath);
    expect(await call('list', telemetry)).toEqual([expect.objectContaining({ filePath, size: 4, mimeType: 'video/mp4' })]);
    expect(await call('read', filePath, 2)).toEqual({ data: 'AwQ=', bytesRead: 2 });
    await fs.unlink(telemetry);
    expect(await call('list', telemetry)).toHaveLength(1);
    await call('delete', filePath);
    expect(await call('list', telemetry)).toEqual([]);
});

it('only lists MP4 recordings and rejects WebM file operations', async () => {
    const legacyPath = `${telemetry}.video-1-abc.webm`;
    await fs.writeFile(legacyPath, Uint8Array.from([1, 2]));
    const { filePath } = await call('create', telemetry);
    await call('append', filePath, Uint8Array.from([3, 4]));
    await call('finish', filePath);
    expect(await call('list', telemetry)).toEqual([
        expect.objectContaining({ filePath, mimeType: 'video/mp4', size: 2 }),
    ]);
    await expect(call('read', legacyPath, 0)).rejects.toThrow('Invalid video file');
    await expect(call('reveal', legacyPath)).rejects.toThrow('Invalid video file');
    await expect(call('delete', legacyPath)).rejects.toThrow('Invalid video file');
});

it('allows recovery after a renderer crash and rejects stale writes', async () => {
    const { filePath } = await call('create', telemetry);
    await call('append', filePath, Uint8Array.from([1]));
    event.sender.emit('render-process-gone');
    expect(await call('read', filePath, 0)).toEqual({ data: 'AQ==', bytesRead: 1 });
    await expect(call('append', filePath, Uint8Array.from([2]))).rejects.toThrow('Invalid video write');
});

it('rejects foreign frames, unrelated files, and oversized chunks', async () => {
    await expect(handlers['session-video-list']({ ...event, senderFrame: {} }, telemetry)).rejects.toThrow('main workspace');
    await expect(call('read', telemetry, 0)).rejects.toThrow('Invalid video file');
    await expect(call('delete', __filename)).rejects.toThrow('outside');
    const { filePath } = await call('create', telemetry);
    await expect(call('append', filePath, new Uint8Array(512 * 1024 + 1))).rejects.toThrow('Invalid video write');
    await expect(call('read', filePath, -1)).rejects.toThrow('not ready');
});
