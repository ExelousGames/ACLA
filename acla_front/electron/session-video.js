const fs = require('fs/promises');
const path = require('path');
const crypto = require('crypto');
const { Readable } = require('stream');
const trainingLabels = require('../src/data/recorded-training-labels.json');
const { VIDEO_TIMING_BYTES, encodeVideoTelemetryStart, readVideoTelemetryStart } = require('../src/data/session-video-timing');
const { DESKTOP_GAME_SET } = require('./recording/recording-protocol');

const CHUNK_BYTES = 512 * 1024;
const MAX_TRAINING_RANGES = 2000;
const MAX_SIDECAR_BYTES = 256 * 1024;
const TRAINING_LABELS = new Set(trainingLabels);
const TRAINING_MEDIA_SCHEME = 'acla-training-media';
const RANGE_SUFFIX = '.training-ranges.json';
const VIDEO_TYPES = {
  mp4: 'video/mp4', webm: 'video/webm', mkv: 'video/x-matroska', mov: 'video/quicktime',
  m4v: 'video/mp4', avi: 'video/x-msvideo', ogv: 'video/ogg', mpg: 'video/mpeg',
  mpeg: 'video/mpeg', '3gp': 'video/3gpp',
};
const extension = (name) => path.extname(name).slice(1).toLowerCase();
const videoType = (name) => Object.hasOwn(VIDEO_TYPES, extension(name)) && VIDEO_TYPES[extension(name)];
const fileName = (value) => typeof value === 'string' && value.length > 0 && !/[/\\:\x00-\x1f]/.test(value);
const inferredTelemetry = (video) => video.replace(/\.jsonl\.video-\d+-[a-f0-9-]+\.(mp4|webm)$|\.[^.]+$/i, '.jsonl');

function trainingLabel(label) {
  if (label !== undefined && typeof label !== 'string') throw new Error('Training range labels must be text.');
  return label?.trim();
}

function trainingSelection(document) {
  const { version, offsetSeconds, ranges, videoFileName, telemetryFileName, game } = document ?? {};
  if ((version !== 1 && version !== 2) || (offsetSeconds !== undefined && !Number.isFinite(offsetSeconds))
    || (version === 2 && (!fileName(videoFileName) || !videoType(videoFileName)
      || !fileName(telemetryFileName) || extension(telemetryFileName) !== 'jsonl'))
    || (ranges !== null && (!Array.isArray(ranges) || ranges.length > MAX_TRAINING_RANGES))) {
    throw new Error('Invalid training range file.');
  }
  if (game !== undefined && !DESKTOP_GAME_SET.has(game)) throw new Error('Choose a recognized simulator for the training annotation.');
  let previousEnd = -Infinity;
  const validated = ranges?.map((range) => {
    const { start, end, label } = range ?? {};
    if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end <= start || start <= previousEnd) {
      throw new Error('Training ranges must be sorted and separated.');
    }
    previousEnd = end;
    const trimmedLabel = trainingLabel(label);
    return trimmedLabel ? { start, end, label: trimmedLabel } : { start, end };
  }) ?? null;
  const fullRecordingLabel = ranges === null ? trainingLabel(document.fullRecordingLabel) : undefined;
  const fullRecording = fullRecordingLabel ? { fullRecordingLabel } : {};
  const metadata = { ...(offsetSeconds === undefined ? {} : { offsetSeconds }), ...(game === undefined ? {} : { game }) };
  return version === 1 ? { version, ...metadata, ranges: validated, ...fullRecording }
    : { version, videoFileName, telemetryFileName, ...metadata, ranges: validated, ...fullRecording };
}

async function readTrainingSelection(filePath, discoverRenamed = false) {
  const info = await fs.lstat(filePath);
  if (!info.isFile() || info.size > MAX_SIDECAR_BYTES) {
    if (discoverRenamed) return null;
    throw new Error('Invalid training range file.');
  }
  const file = await fs.open(filePath, 'r');
  try {
    const bytes = Buffer.alloc(MAX_SIDECAR_BYTES + 1);
    const { bytesRead } = await file.read(bytes);
    let document;
    try {
      if (bytesRead > MAX_SIDECAR_BYTES) throw new Error('Invalid training range file.');
      document = JSON.parse(bytes.toString('utf8', 0, bytesRead));
    } catch (error) {
      if (discoverRenamed) return null;
      throw error;
    }
    if (discoverRenamed && !(document?.version === 2 && (document.videoFileName || document.telemetryFileName))) return null;
    return trainingSelection(document);
  } finally { await file.close(); }
}

function validateTrainingLabels({ ranges, fullRecordingLabel }, previous) {
  const error = 'Choose a training label from the list.';
  if (fullRecordingLabel && fullRecordingLabel !== previous?.fullRecordingLabel && !TRAINING_LABELS.has(fullRecordingLabel)) {
    throw new Error(error);
  }
  // Editing the time bounds keeps a legacy label on the same clip. Count old
  // occurrences because the ranges have no stable ID to compare across edits.
  const legacyLabels = new Map();
  for (const { label } of previous?.ranges ?? []) {
    if (label && !TRAINING_LABELS.has(label)) legacyLabels.set(label, (legacyLabels.get(label) ?? 0) + 1);
  }
  for (const { label } of ranges ?? []) {
    if (!label || TRAINING_LABELS.has(label)) continue;
    const remaining = legacyLabels.get(label) ?? 0;
    if (!remaining) throw new Error(error);
    legacyLabels.set(label, remaining - 1);
  }
}

async function writeTrainingSelection(filePath, selection, previous) {
  const contents = `${JSON.stringify(selection, null, 2)}\n`;
  if (Buffer.byteLength(contents) > MAX_SIDECAR_BYTES) throw new Error('Training annotation is too large.');
  validateTrainingLabels(selection, previous);
  const temporary = `${filePath}.${crypto.randomUUID()}.tmp`;
  const file = await fs.open(temporary, 'wx');
  try {
    try { await file.writeFile(contents); }
    finally { await file.close(); }
    await fs.rename(temporary, filePath);
  }
  finally { await fs.rm(temporary, { force: true }); }
}

async function regularFile(filePath) {
  if (typeof filePath !== 'string' || !path.isAbsolute(filePath) || !(await fs.lstat(filePath)).isFile()) {
    throw new Error('Choose a regular local file.');
  }
  return fs.realpath(filePath);
}

async function videoTelemetryStart(filePath) {
  if (extension(filePath) !== 'mp4') return null;
  const file = await fs.open(filePath, 'r');
  try {
    const { size } = await file.stat();
    if (size < VIDEO_TIMING_BYTES) return null;
    const tail = Buffer.alloc(VIDEO_TIMING_BYTES);
    const { bytesRead } = await file.read(tail, 0, tail.length, size - tail.length);
    return bytesRead === tail.length ? readVideoTelemetryStart(tail) : null;
  } finally { await file.close(); }
}

async function streamTrainingMedia(request, filePath, type) {
  if (request.method !== 'GET' && request.method !== 'HEAD') return new Response(null, { status: 405 });
  const headers = new Headers({ 'Content-Type': type, 'X-Content-Type-Options': 'nosniff',
    'Access-Control-Allow-Origin': '*', 'Accept-Ranges': 'bytes', 'Cache-Control': 'no-store' });
  let file;
  try {
    if (await regularFile(filePath) !== filePath) return new Response(null, { status: 404, headers });
    file = await fs.open(filePath, 'r');
    const { size } = await file.stat();
    let start = 0, end = size - 1;
    const range = request.headers.get('range');
    if (range) {
      const bytes = /^bytes=(\d*)-(\d*)$/.exec(range);
      if (bytes?.[1]) {
        start = Number(bytes[1]);
        if (bytes[2]) end = Math.min(end, Number(bytes[2]));
      } else if (bytes?.[2]) start = Math.max(0, size - Number(bytes[2]));
      else start = size;
      if (start > end) {
        headers.set('Content-Range', `bytes */${size}`);
        return new Response(null, { status: 416, headers });
      }
      headers.set('Content-Range', `bytes ${start}-${end}/${size}`);
    }
    headers.set('Content-Length', String(end - start + 1));
    const status = range ? 206 : 200;
    if (request.method === 'HEAD' || size === 0) return new Response(null, { status, headers });
    const response = new Response(Readable.toWeb(file.createReadStream({ start, end })), { status, headers });
    file = null; // The response stream owns the file handle.
    return response;
  } catch { return new Response(null, { status: 404, headers }); }
  finally { await file?.close(); }
}

function registerSessionVideo({ ipcMain, getMainWindow, getRecordingDirectory, shell }) {
  const writers = new Map();
  const creating = new Set();
  const watched = new WeakSet();
  const media = new Map();
  const annotations = new Map();
  const mediaKey = crypto.randomBytes(32);
  const keyFor = (filePath) => crypto.createHmac('sha256', mediaKey).update(filePath).digest('hex');
  const belongsToSession = (name, source) => name.startsWith(`${path.basename(source)}.video-`)
    && /\.video-\d+-[a-f0-9-]+\.mp4$/.test(name);
  const owner = (event) => {
    const window = getMainWindow();
    if (!window || window.isDestroyed() || event.sender !== window.webContents
      || event.senderFrame !== window.webContents.mainFrame) throw new Error('Video recording is available only in the main workspace.');
  };
  const watchOwner = (sender) => {
    if (watched.has(sender)) return;
    watched.add(sender);
    const clear = (inPlace = false) => {
      for (const [file, writer] of writers) if (writer.sender === sender) writers.delete(file);
      if (!inPlace) {
        for (const [key, source] of media) if (source.sender === sender) media.delete(key);
        for (const [key, source] of annotations) if (source.sender === sender) annotations.delete(key);
      }
    };
    sender.on('destroyed', () => clear());
    sender.on('render-process-gone', () => clear());
    sender.on('did-start-navigation', (_event, _url, inPlace, isMainFrame) => { if (isMainFrame) clear(inPlace); });
  };
  const authorized = async (filePath) => {
    if (typeof filePath !== 'string' || !path.isAbsolute(filePath)) throw new Error('Invalid recording path.');
    const directory = await fs.realpath(getRecordingDirectory());
    const resolved = await fs.realpath(filePath);
    if (path.dirname(resolved) !== directory || !(await fs.stat(resolved)).isFile()) throw new Error('File is outside the recording directory.');
    return resolved;
  };
  const videoPath = async (filePath) => {
    const resolved = await authorized(filePath);
    if (!/\.video-\d+-[a-f0-9-]+\.mp4$/.test(resolved)) throw new Error('Invalid video file.');
    return resolved;
  };
  // Preload supplies paths from the user's file pickers. Only resolved media
  // named in a selected annotation can be streamed to the renderer.
  const trainingVideo = async (filePath) => {
    if (typeof filePath !== 'string' || !videoType(filePath)) throw new Error('Choose a local video file.');
    return regularFile(filePath);
  };
  const describeAnnotation = async (event, filePath, selection, videoPath, telemetryPath) => {
    owner(event);
    const videoTelemetryStartSeconds = await videoTelemetryStart(videoPath);
    owner(event);
    watchOwner(event.sender);
    const id = keyFor(filePath);
    annotations.set(id, { filePath, sender: event.sender });
    const grant = (source, type) => {
      const key = keyFor(source);
      media.set(key, { filePath: source, type, sender: event.sender });
      return `${TRAINING_MEDIA_SCHEME}://recording/${key}`;
    };
    return {
      id, name: path.basename(filePath), selection,
      videoUrl: grant(videoPath, videoType(videoPath)),
      telemetryUrl: grant(telemetryPath, 'text/plain; charset=utf-8'),
      ...(videoTelemetryStartSeconds === null ? {} : { videoTelemetryStartSeconds }),
    };
  };
  ipcMain.handle('session-video-training-read', async (event, filePath) => {
    owner(event);
    const video = await trainingVideo(filePath);
    let selection;
    try { selection = await readTrainingSelection(video + RANGE_SUFFIX); }
    catch (error) { if (error.code === 'ENOENT') return null; throw error; }
    if (selection.version === 2 && selection.videoFileName !== path.basename(video)) {
      throw new Error('This annotation refers to a different video.');
    }
    return selection;
  });
  const readAnnotation = async (annotation, discover = false) => {
    if (extension(annotation) !== 'json') throw new Error('Choose a training annotation JSON file.');
    const sidecar = annotation.toLowerCase().endsWith(RANGE_SUFFIX);
    const saved = await readTrainingSelection(annotation, discover && !sidecar);
    if (!saved) return null;
    const videoFileName = saved.version === 2 ? saved.videoFileName : path.basename(annotation).slice(0, -RANGE_SUFFIX.length);
    if (saved.version === 1 && (!sidecar || !videoType(videoFileName))) {
      throw new Error('Open the original video and save this older annotation to add its filenames.');
    }
    const selection = trainingSelection({ ...saved, version: 2, videoFileName,
      telemetryFileName: saved.telemetryFileName ?? inferredTelemetry(videoFileName) });
    const directory = path.dirname(annotation);
    const sibling = async (name) => {
      const requested = path.join(directory, name);
      const resolved = await regularFile(requested).catch(() => null);
      if (resolved !== requested) throw new Error(`Put ${name} in the same folder as this annotation.`);
      return resolved;
    };
    const [video, telemetry] = await Promise.all([sibling(selection.videoFileName), sibling(selection.telemetryFileName)]);
    return { selection, video, telemetry };
  };
  ipcMain.handle('session-video-training-open', async (event, filePath) => {
    owner(event);
    const annotation = await regularFile(filePath);
    const { selection, video, telemetry } = await readAnnotation(annotation);
    return describeAnnotation(event, annotation, selection, video, telemetry);
  });
  ipcMain.handle('session-video-training-scan', async (event, filePaths) => {
    owner(event);
    const results = [];
    for (const filePath of filePaths) {
      try {
        const annotation = await regularFile(filePath);
        const pair = await readAnnotation(annotation, true);
        results.push(pair ? await describeAnnotation(event, annotation, pair.selection, pair.video, pair.telemetry) : null);
      } catch (error) {
        results.push({ error: error.message });
      }
    }
    return results;
  });
  ipcMain.handle('session-video-training-update', async (event, id, draft) => {
    owner(event);
    const source = annotations.get(id);
    if (source?.sender !== event.sender) throw new Error('Open the training annotation before editing it.');
    const { selection, video, telemetry } = await readAnnotation(source.filePath);
    const { ranges, fullRecordingLabel, game = selection.game } = draft ?? {};
    const updated = trainingSelection({ ...selection, ranges, fullRecordingLabel, game: game ?? undefined });
    await writeTrainingSelection(source.filePath, updated, selection);
    return describeAnnotation(event, source.filePath, updated, video, telemetry);
  });
  ipcMain.handle('session-video-training-write', async (event, filePath, telemetryPath, draft) => {
    owner(event);
    const video = await trainingVideo(filePath);
    const telemetry = await regularFile(telemetryPath);
    if (path.dirname(video) !== path.dirname(telemetry)) {
      throw new Error('Choose video and telemetry files from the same folder.');
    }
    const sidecar = video + RANGE_SUFFIX;
    const videoFileName = path.basename(video);
    const telemetryFileName = path.basename(telemetry);
    let previous;
    try { previous = await readTrainingSelection(sidecar); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    const sameVideo = previous?.version === 1 || previous?.videoFileName === videoFileName;
    const samePair = sameVideo && (previous.version === 1 ? inferredTelemetry(videoFileName) === telemetryFileName
      : previous.telemetryFileName === telemetryFileName);
    const { ranges, fullRecordingLabel, game = samePair ? previous.game : undefined } = draft ?? {};
    const selection = trainingSelection({ version: 2, videoFileName, telemetryFileName, ranges, fullRecordingLabel, game: game ?? undefined,
      offsetSeconds: samePair ? previous.offsetSeconds : undefined });
    await writeTrainingSelection(sidecar, selection, sameVideo ? previous : undefined);
    return describeAnnotation(event, sidecar, selection, video, telemetry);
  });
  ipcMain.handle('session-video-create', async (event, telemetryPath) => {
    owner(event);
    watchOwner(event.sender);
    const source = await authorized(telemetryPath);
    if (!source.endsWith('.jsonl')) throw new Error('Invalid telemetry file.');
    if (creating.has(source)) throw new Error('Only one video is allowed per session.');
    creating.add(source);
    try {
      const files = await fs.readdir(path.dirname(source));
      if (files.some(name => belongsToSession(name, source))) throw new Error('Only one video is allowed per session.');
      const startedAt = Date.now();
      const filePath = `${source}.video-${startedAt}-${crypto.randomUUID()}.mp4`;
      await fs.writeFile(filePath, Buffer.alloc(0), { flag: 'wx' });
      writers.set(filePath, { sender: event.sender, finishing: false });
      return { filePath, startedAt };
    } finally { creating.delete(source); }
  });
  ipcMain.handle('session-video-append', async (event, filePath, bytes) => {
    owner(event);
    const writer = writers.get(filePath);
    if (writer?.sender !== event.sender || writer.finishing || !(bytes instanceof Uint8Array) || bytes.byteLength > CHUNK_BYTES) {
      throw new Error('Invalid video write.');
    }
    await fs.appendFile(await videoPath(filePath), bytes);
  });
  ipcMain.handle('session-video-finish', async (event, filePath, videoTelemetryStartSeconds) => {
    owner(event);
    const writer = writers.get(filePath);
    if (writer?.sender !== event.sender || writer.finishing) throw new Error('Unknown video recording.');
    const timing = videoTelemetryStartSeconds === undefined ? null : encodeVideoTelemetryStart(videoTelemetryStartSeconds);
    writer.finishing = true;
    try {
      const resolved = await videoPath(filePath);
      if (!(await fs.stat(resolved)).size) await fs.unlink(resolved);
      else if (timing) await fs.appendFile(resolved, timing);
    } finally { writers.delete(filePath); }
  });
  ipcMain.handle('session-video-list', async (event, telemetryPath) => {
    owner(event);
    if (typeof telemetryPath !== 'string' || !path.isAbsolute(telemetryPath) || !telemetryPath.endsWith('.jsonl')) {
      throw new Error('Invalid telemetry file.');
    }
    let directory;
    try {
      directory = await fs.realpath(path.dirname(telemetryPath));
      if (directory !== await fs.realpath(getRecordingDirectory())) throw new Error('File is outside the recording directory.');
    } catch (error) {
      if (error.code === 'ENOENT') return [];
      throw error;
    }
    const source = path.join(directory, path.basename(telemetryPath));
    const prefix = `${path.basename(source)}.video-`;
    const clips = [];
    for (const name of (await fs.readdir(directory)).sort()) {
      if (!belongsToSession(name, source)) continue;
      const filePath = await videoPath(path.join(directory, name));
      const { size } = await fs.stat(filePath);
      if (size) clips.push({ filePath, name, size, mimeType: 'video/mp4', startedAt: Number(name.slice(prefix.length).split('-')[0]) });
    }
    return clips;
  });
  ipcMain.handle('session-video-read', async (event, filePath, offset) => {
    owner(event);
    if (!Number.isSafeInteger(offset) || offset < 0 || writers.has(filePath)) throw new Error('Video is not ready to upload.');
    const file = await fs.open(await videoPath(filePath), 'r');
    try {
      const bytes = Buffer.alloc(CHUNK_BYTES);
      const { bytesRead } = await file.read(bytes, 0, bytes.length, offset);
      return { data: bytes.subarray(0, bytesRead).toString('base64'), bytesRead };
    } finally { await file.close(); }
  });
  ipcMain.handle('session-video-delete', async (event, filePath) => {
    owner(event);
    if (writers.has(filePath)) throw new Error('Stop recording before deleting the video.');
    await fs.unlink(await videoPath(filePath));
  });
  ipcMain.handle('session-video-reveal', async (event, filePath) => {
    owner(event);
    shell.showItemInFolder(await videoPath(filePath));
  });
  return (request) => {
    const key = new URL(request.url).pathname.slice(1);
    const source = media.get(key);
    if (!source || getMainWindow()?.webContents !== source.sender) return new Response(null, { status: 404 });
    return streamTrainingMedia(request, source.filePath, source.type);
  };
}

module.exports = { registerSessionVideo, TRAINING_MEDIA_SCHEME };
