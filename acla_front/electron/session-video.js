const fs = require('fs/promises');
const path = require('path');
const crypto = require('crypto');

const CHUNK_BYTES = 512 * 1024;

function registerSessionVideo({ ipcMain, getMainWindow, getRecordingDirectory, shell }) {
  const writers = new Map();
  const creating = new Set();
  const watched = new WeakSet();
  const belongsToSession = (name, source) => name.startsWith(`${path.basename(source)}.video-`)
    && /\.video-\d+-[a-f0-9-]+\.mp4$/.test(name);
  const owner = (event) => {
    const window = getMainWindow();
    if (!window || window.isDestroyed() || event.sender !== window.webContents
      || event.senderFrame !== window.webContents.mainFrame) throw new Error('Video recording is available only in the main workspace.');
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
  ipcMain.handle('session-video-create', async (event, telemetryPath) => {
    owner(event);
    if (!watched.has(event.sender)) {
      watched.add(event.sender);
      const clear = () => { for (const [file, sender] of writers) if (sender === event.sender) writers.delete(file); };
      event.sender.on('destroyed', clear);
      event.sender.on('render-process-gone', clear);
      event.sender.on('did-start-navigation', (_event, _url, _inPlace, isMainFrame) => { if (isMainFrame) clear(); });
    }
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
      writers.set(filePath, event.sender);
      return { filePath, startedAt };
    } finally { creating.delete(source); }
  });
  ipcMain.handle('session-video-append', async (event, filePath, bytes) => {
    owner(event);
    if (writers.get(filePath) !== event.sender || !(bytes instanceof Uint8Array) || bytes.byteLength > CHUNK_BYTES) {
      throw new Error('Invalid video write.');
    }
    await fs.appendFile(await videoPath(filePath), bytes);
  });
  ipcMain.handle('session-video-finish', async (event, filePath) => {
    owner(event);
    if (writers.get(filePath) !== event.sender) throw new Error('Unknown video recording.');
    writers.delete(filePath);
    const resolved = await videoPath(filePath);
    if (!(await fs.stat(resolved)).size) await fs.unlink(resolved);
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
}

module.exports = { registerSessionVideo };
