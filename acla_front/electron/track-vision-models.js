const fs = require('fs/promises');
const path = require('path');
const { createHash } = require('crypto');
const { execFile } = require('child_process');

const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const CACHE_VERSION = 1;

function validateModel(model) {
  if (!model || typeof model.id !== 'string' || typeof model.sha256 !== 'string'
    || !/^[a-f\d]{24}$/i.test(model.id) || !/^[a-f\d]{64}$/i.test(model.sha256)
    || model.task !== 'segment' || !Number.isSafeInteger(model.sizeBytes) || model.sizeBytes <= 0
    || model.sizeBytes > 512 * 1024 * 1024 || !Array.isArray(model.classNames) || !model.classNames.length
    || !model.classNames.every((label) => typeof label === 'string' && label.trim())
    || new Set(model.classNames).size !== model.classNames.length) {
    throw new Error('Invalid Track Vision model metadata.');
  }
}

async function readFile(filename) {
  try { return await fs.readFile(filename); }
  catch (error) { if (error.code === 'ENOENT') return null; throw error; }
}

async function writeFile(filename, bytes) {
  const temporary = `${filename}.tmp`;
  await fs.writeFile(temporary, bytes);
  await fs.rename(temporary, filename);
}

function createTrackVisionModelCache({ directory, exportModel }) {
  const pending = new Map();
  async function prepare(model, download) {
    const folder = path.join(directory(), `${model.id}-${model.sha256}`);
    const manifestPath = path.join(folder, 'manifest.json');
    const weightsPath = path.join(folder, 'weights.pt');
    const onnxPath = path.join(folder, 'weights.onnx');
    const manifestBytes = await readFile(manifestPath);
    let manifest;
    try { manifest = manifestBytes ? JSON.parse(manifestBytes.toString()) : null; }
    catch (error) { if (!(error instanceof SyntaxError)) throw error; }
    if (manifest?.version === CACHE_VERSION && manifest.model?.sha256 === model.sha256
      && JSON.stringify(manifest.model.classNames) === JSON.stringify(model.classNames)) {
      const onnx = await readFile(onnxPath);
      if (onnx?.length && hash(onnx) === manifest.onnxSha256) return onnx;
    }

    let weights = await readFile(weightsPath);
    if (!weights || weights.length !== model.sizeBytes || hash(weights) !== model.sha256) {
      if (download === undefined) return null;
      if (!(download instanceof ArrayBuffer) && !ArrayBuffer.isView(download)) {
        throw new Error('Invalid Track Vision model download.');
      }
      weights = Buffer.from(download instanceof ArrayBuffer ? new Uint8Array(download) : download);
      if (weights.length !== model.sizeBytes || hash(weights) !== model.sha256) {
        throw new Error('Track Vision model checksum mismatch. Retry the download.');
      }
      await fs.mkdir(folder, { recursive: true });
      await writeFile(weightsPath, weights);
    }
    await exportModel(weightsPath, model.classNames);
    const onnx = await readFile(onnxPath);
    if (!onnx?.length) throw new Error('Track Vision model export produced no ONNX weights.');
    await writeFile(manifestPath, JSON.stringify({ version: CACHE_VERSION, model, onnxSha256: hash(onnx) }));
    return onnx;
  }
  return {
    async prepare(model, download) {
      validateModel(model);
      const key = `${model.id}-${model.sha256}`;
      // Multiple panels share the cache; only one export may write a model at a time.
      const operation = (pending.get(key) || Promise.resolve()).catch(() => undefined)
        .then(() => prepare(model, download));
      pending.set(key, operation);
      try { return await operation; }
      finally { if (pending.get(key) === operation) pending.delete(key); }
    },
  };
}

function registerTrackVisionModels({ app, ipcMain, getMainWindow, getPythonExecutable }) {
  const cache = createTrackVisionModelCache({
    directory: () => path.join(app.getPath('userData'), 'track-vision', 'models'),
    exportModel: (weights, labels) => new Promise((resolve, reject) => {
      const script = app.isPackaged
        ? path.join(process.resourcesPath, 'py-scripts', 'export_track_vision_model.py')
        : path.join(app.getAppPath(), 'src', 'py-scripts', 'export_track_vision_model.py');
      execFile(getPythonExecutable(), [script, '--weights', weights, '--labels', JSON.stringify(labels)], {
        windowsHide: true, timeout: 600000, maxBuffer: 8 * 1024 * 1024,
        env: { ...process.env, YOLO_OFFLINE: 'true', YOLO_AUTOINSTALL: 'false',
          YOLO_CONFIG_DIR: path.join(app.getPath('userData'), 'track-vision', 'config') },
      }, (error, _stdout, stderr) => {
        if (error) reject(new Error(`Track Vision model preparation failed. ${stderr.trim().slice(-1500) || error.message}`));
        else resolve();
      });
    }),
  });
  ipcMain.handle('track-vision-model-prepare', (event, model, download) => {
    const window = getMainWindow();
    if (!window || window.isDestroyed() || event.sender !== window.webContents
      || event.senderFrame !== window.webContents.mainFrame) {
      throw new Error('Track Vision models are available only in the main workspace.');
    }
    return cache.prepare(model, download);
  });
}

module.exports = { createTrackVisionModelCache, registerTrackVisionModels };
