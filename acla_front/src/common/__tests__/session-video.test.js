/** @jest-environment node */
const fs = require('fs/promises');
const os = require('os');
const path = require('path');
const { EventEmitter } = require('events');
const { runInThisContext } = require('vm');
const { registerSessionVideo } = require('../../../electron/session-video');
const { VIDEO_TIMING_BYTES, readVideoTelemetryStart } = require('../../data/session-video-timing');

let directory;
let handlers;
let event;
let telemetry;
let serveTrainingMedia;

beforeEach(async () => {
    directory = await fs.mkdtemp(path.join(os.tmpdir(), 'acla-video-test-'));
    telemetry = path.join(directory, 'acc.jsonl');
    await fs.writeFile(telemetry, '{}\n');
    handlers = {};
    const contents = new EventEmitter();
    contents.mainFrame = {};
    event = { sender: contents, senderFrame: contents.mainFrame };
    serveTrainingMedia = registerSessionVideo({ ipcMain: { handle: (name, callback) => { handlers[name] = callback; } },
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

it('persists video timing inside the uploaded MP4 bytes and exposes it when opening an annotation', async () => {
    const { filePath } = await call('create', telemetry);
    await call('append', filePath, Uint8Array.from([1, 2, 3, 4]));
    await expect(call('finish', filePath, NaN)).rejects.toThrow('Invalid video telemetry start time');
    const finishing = call('finish', filePath, 0.37125);
    await expect(call('read', filePath, 0)).rejects.toThrow('not ready');
    await finishing;
    const { data, bytesRead } = await call('read', filePath, 4);
    expect(bytesRead).toBe(VIDEO_TIMING_BYTES);
    expect(readVideoTelemetryStart(Buffer.from(data, 'base64'))).toBe(0.37125);
    expect((await call('list', telemetry))[0].size).toBe(4 + VIDEO_TIMING_BYTES);
    await expect(call('training-write', filePath, telemetry, { ranges: null })).resolves.toMatchObject({
        videoTelemetryStartSeconds: 0.37125,
    });
    await expect(call('training-open', `${filePath}.training-ranges.json`)).resolves.toMatchObject({
        videoTelemetryStartSeconds: 0.37125,
    });
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

it('saves and reloads independent annotations with both filenames', async () => {
    const first = path.join(directory, 'first.webm');
    const second = path.join(directory, 'second.mp4');
    await Promise.all([fs.writeFile(first, 'video one'), fs.writeFile(second, 'video two')]);
    const selected = { version: 2, videoFileName: 'first.webm', telemetryFileName: 'acc.jsonl', game: 'acc', offsetSeconds: -0.25,
        ranges: [{ start: 1, end: 3, label: 'Shift down' }, { start: 1, end: 3 },
            { start: 2, end: 3.5, label: 'Switch lane left' }, { start: 3.5, end: 5, label: 'Switch lane right' }] };
    const other = { ...selected, videoFileName: 'second.mp4', game: 'iracing', ranges: null, fullRecordingLabel: 'Shift up' };
    await expect(call('training-read', first)).resolves.toBeNull();
    const draft = { ...selected,
        ranges: [{ ...selected.ranges[0], label: '  Shift down ' },
            { ...selected.ranges[1], label: ' ' }, ...selected.ranges.slice(2)] };
    await expect(call('training-write', first, telemetry, draft)).resolves.toMatchObject({ selection: selected });
    await call('training-write', second, telemetry, other);
    await expect(call('training-read', first)).resolves.toEqual(selected);
    await expect(call('training-read', second)).resolves.toEqual(other);
    await call('training-write', second, telemetry, { ranges: [], game: null });
    expect(await call('training-read', second)).not.toHaveProperty('game');
    await expect(call('training-write', first, telemetry, { ...selected, ranges: [{ start: 2, end: 4 }, { start: 1, end: 3 }] }))
        .rejects.toThrow('sorted by start time');
    await expect(call('training-write', first, telemetry, { ranges: [{ start: 2, end: 2 }] }))
        .rejects.toThrow('an end after the start');
    await expect(call('training-write', first, telemetry, { ranges: [{ start: 1, end: 2, label: 123 }] }))
        .rejects.toThrow('Training range labels must be text');
    await expect(call('training-write', first, telemetry, { ranges: [{ start: 1, end: 2, label: 'Shifting' }] }))
        .rejects.toThrow('Choose a training label from the list');
    await expect(call('training-write', first, telemetry, { ...selected, game: 'unknown' }))
        .rejects.toThrow('Choose a recognized simulator');
    await expect(call('training-write', first, telemetry, { ...selected, offsetSeconds: NaN }))
        .rejects.toThrow('Invalid training range file');
    await expect(call('training-write', first, telemetry, { ranges: [{ start: 1, end: 2, label: 'é'.repeat(128 * 1024) }] }))
        .rejects.toThrow('Training annotation is too large');
    expect(JSON.parse(await fs.readFile(`${first}.training-ranges.json`, 'utf8'))).toEqual(selected);
    expect(await fs.readFile(first, 'utf8')).toBe('video one');
    expect(await fs.readFile(second, 'utf8')).toBe('video two');
    await expect(call('training-write', telemetry, telemetry, selected)).rejects.toThrow('Choose a local video');
    await expect(handlers['session-video-training-write']({ ...event, senderFrame: {} }, first, telemetry, selected)).rejects.toThrow('main workspace');
});

it('edits renamed annotations in place and streams byte ranges for video seeking', async () => {
    global.Headers = runInThisContext('Headers');
    global.Response = runInThisContext('Response');
    const video = path.join(directory, 'camera.webm');
    const annotation = path.join(directory, 'turns.json');
    await fs.writeFile(video, '0123456789');
    await fs.writeFile(annotation, JSON.stringify({ version: 2, videoFileName: 'camera.webm', telemetryFileName: 'acc.jsonl',
        game: 'iracing', offsetSeconds: 0.2, ranges: [{ start: 1, end: 2, label: 'Clean corner' }] }));
    const { id, videoUrl, telemetryUrl, selection, name } = await call('training-open', annotation);
    expect(name).toBe('turns.json');
    expect(selection).toMatchObject({ videoFileName: 'camera.webm', telemetryFileName: 'acc.jsonl', game: 'iracing', offsetSeconds: 0.2 });
    const ranges = [{ start: 2, end: 3, label: 'Clean corner' }, { start: 2.5, end: 3.5 }];
    const updated = { ...selection, ranges, offsetSeconds: 0.4 };
    await expect(call('training-update', id, { ranges, offsetSeconds: 0.4 })).resolves.toMatchObject({ id, name, selection: updated });
    await expect(call('training-update', id, { ranges: [...ranges, { start: 4, end: 5, label: 'Clean corner' }] }))
        .rejects.toThrow('Choose a training label from the list');
    expect(JSON.parse(await fs.readFile(annotation, 'utf8'))).toEqual(updated);
    await expect(call('training-open', annotation)).resolves.toMatchObject({ selection: updated });
    await call('training-update', id, { ranges, game: null });
    const withoutGame = (await call('training-open', annotation)).selection;
    expect(withoutGame).not.toHaveProperty('game');
    expect(withoutGame.offsetSeconds).toBe(0.4);
    await expect(call('training-update', id, { ranges, offsetSeconds: 0 })).resolves.toMatchObject({ selection: { offsetSeconds: 0 } });
    expect(await fs.readdir(directory)).toHaveLength(3);
    const request = (url, range) => ({ url, method: 'GET', headers: new Headers(range ? { range } : {}) });
    try {
        const part = await serveTrainingMedia(request(videoUrl, 'bytes=3-5'));
        expect(part.status).toBe(206);
        expect(part.headers.get('content-range')).toBe('bytes 3-5/10');
        expect(await part.text()).toBe('345');
        expect(await (await serveTrainingMedia(request(telemetryUrl))).text()).toBe('{}\n');
        expect((await serveTrainingMedia(request(videoUrl, 'bytes=12-'))).status).toBe(416);
        event.sender.emit('render-process-gone');
        expect((await serveTrainingMedia(request(videoUrl))).status).toBe(404);
        await expect(call('training-update', id, selection)).rejects.toThrow('Open the training annotation before editing it');
    } finally {
        delete global.Headers;
        delete global.Response;
    }
});

it('persists a full recording label and clears it when clips are selected', async () => {
    const video = path.join(directory, 'aligned.mp4');
    await fs.writeFile(video, 'video');
    await expect(call('training-write', video, telemetry, { ranges: null, fullRecordingLabel: 'Cornering' }))
        .rejects.toThrow('Choose a training label from the list');
    const { id, selection } = await call('training-write', video, telemetry, { ranges: null, fullRecordingLabel: '  Warmup ' });
    expect(selection.fullRecordingLabel).toBe('Warmup');
    await expect(call('training-read', video)).resolves.toEqual(selection);
    await expect(call('training-update', id, { ranges: null })).resolves.toMatchObject({ selection: { ranges: null } });
    expect(await call('training-read', video)).not.toHaveProperty('fullRecordingLabel');
    await call('training-update', id, { ranges: null, fullRecordingLabel: selection.fullRecordingLabel });
    await call('training-update', id, { ranges: [{ start: 0, end: 1 }], fullRecordingLabel: selection.fullRecordingLabel });
    expect(await call('training-read', video)).not.toHaveProperty('fullRecordingLabel');
});

it('retains legacy sidecars and rejects missing or out-of-folder media in imported JSON', async () => {
    const video = `${telemetry}.video-1-abc.mp4`;
    const annotation = `${video}.training-ranges.json`;
    const legacy = { version: 1, offsetSeconds: 0.2, ranges: null, fullRecordingLabel: 'Cornering' };
    await fs.writeFile(video, 'video');
    await fs.writeFile(annotation, JSON.stringify(legacy));
    await expect(call('training-read', video)).resolves.toEqual(legacy);
    const opened = await call('training-open', annotation);
    expect(opened).toMatchObject({ selection: {
        ...legacy, version: 2, videoFileName: path.basename(video), telemetryFileName: 'acc.jsonl',
    } });
    const updated = { ...opened.selection, game: 'acc' };
    await expect(call('training-write', video, telemetry, { ranges: null, fullRecordingLabel: legacy.fullRecordingLabel, game: 'acc' }))
        .resolves.toMatchObject({ selection: updated });
    await expect(call('training-read', video)).resolves.toEqual(updated);
    const correctedTelemetry = path.join(directory, 'corrected.jsonl');
    await fs.writeFile(correctedTelemetry, '{}\n');
    await expect(call('training-write', video, correctedTelemetry, { ranges: null, fullRecordingLabel: legacy.fullRecordingLabel }))
        .resolves.toHaveProperty('selection', { version: 2, videoFileName: path.basename(video), telemetryFileName: 'corrected.jsonl',
            ranges: null, fullRecordingLabel: legacy.fullRecordingLabel });
    const changed = { ...legacy, version: 2, videoFileName: path.basename(video), telemetryFileName: 'missing.jsonl' };
    await fs.writeFile(annotation, JSON.stringify(changed));
    await expect(call('training-open', annotation)).rejects.toThrow('Put missing.jsonl in the same folder');
    await fs.writeFile(annotation, JSON.stringify({ ...changed, telemetryFileName: '../acc.jsonl' }));
    await expect(call('training-open', annotation)).rejects.toThrow('Invalid training range file');
});

it('finds legacy and renamed annotations in a folder selection and reports broken pairs', async () => {
    const nested = path.join(directory, 'session');
    await fs.mkdir(nested);
    const legacyVideo = `${telemetry}.video-1-abc.mp4`;
    const sidecar = `${legacyVideo}.training-ranges.json`;
    const renamed = path.join(nested, 'corners.json');
    const missing = path.join(nested, 'missing.training-ranges.json');
    const malformed = path.join(nested, 'bad.training-ranges.json');
    const unrelated = path.join(directory, 'settings.json');
    const legacy = { version: 1, offsetSeconds: 0, ranges: null };
    const selection = { version: 2, videoFileName: 'camera.mp4', telemetryFileName: 'inputs.jsonl',
        offsetSeconds: 0.25, ranges: [{ start: 1, end: 2 }] };
    await Promise.all([
        fs.writeFile(legacyVideo, 'video'),
        fs.writeFile(sidecar, JSON.stringify(legacy)),
        fs.writeFile(path.join(nested, 'camera.mp4'), 'video'),
        fs.writeFile(path.join(nested, 'inputs.jsonl'), '{}\n'),
        fs.writeFile(renamed, JSON.stringify(selection)),
        fs.writeFile(missing, JSON.stringify({ ...selection, telemetryFileName: path.basename(telemetry) })),
        fs.writeFile(malformed, '{'),
        fs.writeFile(unrelated, JSON.stringify({ version: 2, enabled: true })),
    ]);

    const files = [sidecar, unrelated, renamed, missing, malformed];
    const found = await call('training-scan', files);
    expect(found).toEqual([
        expect.objectContaining({ name: path.basename(sidecar), selection: { ...legacy, version: 2,
            videoFileName: path.basename(legacyVideo), telemetryFileName: path.basename(telemetry) } }),
        null,
        expect.objectContaining({ name: 'corners.json', selection }),
        { error: 'Put acc.jsonl in the same folder as this annotation.' },
        { error: expect.any(String) },
    ]);
    await expect(call('training-update', found[2].id, { ranges: [] })).resolves.toMatchObject({
        selection: { ...selection, ranges: [] },
    });
    await expect(handlers['session-video-training-scan']({ ...event, senderFrame: {} }, files)).rejects.toThrow('main workspace');
});
