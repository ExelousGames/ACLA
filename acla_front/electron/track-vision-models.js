const fs = require('fs/promises');
const path = require('path');
const { createHash } = require('crypto');

const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');

function validateModel(model) {
  if (!model || typeof model.id !== 'string' || typeof model.sha256 !== 'string'
    || !/^[a-f\d]{24}$/i.test(model.id) || !/^[a-f\d]{64}$/i.test(model.sha256)
    || !['segment', 'position'].includes(model.task) || typeof model.filename !== 'string' || !/\.onnx$/i.test(model.filename)
    || !Number.isSafeInteger(model.sizeBytes) || model.sizeBytes <= 0
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

function createTrackVisionModelCache({ directory }) {
  const pending = new Map();
  async function prepare(model, download) {
    const folder = path.join(directory(), `${model.id}-${model.sha256}`);
    const weightsPath = path.join(folder, 'weights.onnx');
    let weights = await readFile(weightsPath);
    if (!weights || weights.length !== model.sizeBytes || hash(weights) !== model.sha256) {
      if (download === undefined) return null;
      if (!(download instanceof ArrayBuffer) && !ArrayBuffer.isView(download)) {
        throw new Error('Invalid Track Vision model download.');
      }
      weights = Buffer.from(download instanceof ArrayBuffer
        ? new Uint8Array(download) : new Uint8Array(download.buffer, download.byteOffset, download.byteLength));
      if (weights.length !== model.sizeBytes || hash(weights) !== model.sha256) {
        throw new Error('Track Vision model checksum mismatch. Retry the download.');
      }
      await fs.mkdir(folder, { recursive: true });
      await writeFile(weightsPath, weights);
    }
    return weights;
  }
  return {
    async prepare(model, download) {
      validateModel(model);
      const key = `${model.id}-${model.sha256}`;
      // Multiple panels share the cache; only one download may write a model at a time.
      const operation = (pending.get(key) || Promise.resolve()).catch(() => undefined)
        .then(() => prepare(model, download));
      pending.set(key, operation);
      try { return await operation; }
      finally { if (pending.get(key) === operation) pending.delete(key); }
    },
  };
}

function registerTrackVisionModels({ app, ipcMain, getMainWindow }) {
  const cache = createTrackVisionModelCache({
    directory: () => path.join(app.getPath('userData'), 'track-vision', 'models'),
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
