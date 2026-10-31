const fs = require('fs/promises');
const path = require('path');
const { randomUUID } = require('crypto');

const MAX_MODEL_BYTES = 32 * 1024 * 1024;

function validateDocument(json) {
  if (typeof json !== 'string' || Buffer.byteLength(json, 'utf8') > MAX_MODEL_BYTES) {
    throw new Error('The model file is too large.');
  }
  const document = JSON.parse(json);
  if (!document || document.format !== 'kestrel-self-driving-model' || document.algorithm !== 'online-actor-critic'
    || !document.configuration || typeof document.configuration !== 'object'
    || !(document.model === null || (document.model && typeof document.model === 'object'))) {
    throw new Error('Invalid self-driving model file.');
  }
}

function registerDrivingModelFiles({ app, ipcMain, dialog, getMainWindow }) {
  const owners = new WeakMap();
  const pending = new Map();
  const queue = (filePath, operation) => {
    const task = (pending.get(filePath) || Promise.resolve()).catch(() => undefined).then(operation);
    pending.set(filePath, task);
    return task.finally(() => { if (pending.get(filePath) === task) pending.delete(filePath); });
  };
  const handle = (name, operation) => ipcMain.handle(`driving-model-${name}`, async (event, ...args) => {
    const window = getMainWindow();
    if (!window || window.isDestroyed() || event.sender !== window.webContents
      || event.senderFrame !== window.webContents.mainFrame) {
      throw new Error('Model files are available only in the main workspace.');
    }
    let owner = owners.get(event.sender);
    if (!owner) { owner = { paths: new Set(), current: null }; owners.set(event.sender, owner); }
    return operation(window, owner, ...args);
  });
  const requireSelected = (owner, filePath) => {
    if (!owner.paths.has(filePath)) throw new Error('Choose a model file first.');
  };

  handle('current', (_window, owner) => owner.current);
  handle('choose', async (window, owner, mode) => {
    const filters = [{ name: 'Kestrel self driving model', extensions: ['json'] }];
    let filePath;
    if (mode === 'new') {
      const result = await dialog.showSaveDialog(window, {
        title: 'Save new Kestrel self driving model', buttonLabel: 'Create model', filters,
        defaultPath: path.join(app.getPath('documents'), 'kestrel-self-driving-model.json'),
      });
      if (!result.canceled) filePath = result.filePath;
    } else if (mode === 'open') {
      const result = await dialog.showOpenDialog(window, {
        title: 'Open Kestrel self driving model', filters, properties: ['openFile'],
        defaultPath: owner.current || app.getPath('documents'),
      });
      if (!result.canceled) filePath = result.filePaths[0];
    } else throw new Error('Invalid model file operation.');
    if (!filePath) return null;
    if (!path.isAbsolute(filePath) || path.extname(filePath).toLowerCase() !== '.json') {
      throw new Error('Choose a .json model file.');
    }
    owner.paths.add(filePath);
    return filePath;
  });
  handle('select', (_window, owner, filePath) => {
    requireSelected(owner, filePath);
    // Only remember the path for this desktop session, after renderer validation.
    owner.current = filePath;
  });
  handle('read', (_window, owner, filePath) => {
    requireSelected(owner, filePath);
    return queue(filePath, async () => {
      const info = await fs.stat(filePath);
      if (!info.isFile() || info.size > MAX_MODEL_BYTES) throw new Error('The model file is too large or is not a file.');
      const json = await fs.readFile(filePath, 'utf8');
      validateDocument(json);
      return json;
    });
  });
  handle('write', (_window, owner, filePath, json) => {
    requireSelected(owner, filePath);
    validateDocument(json);
    return queue(filePath, async () => {
      const temporary = path.join(path.dirname(filePath), `.${path.basename(filePath)}.${randomUUID()}.tmp`);
      try {
        const file = await fs.open(temporary, 'wx');
        try { await file.writeFile(json, 'utf8'); await file.sync(); }
        finally { await file.close(); }
        await fs.rename(temporary, filePath);
      } finally { await fs.unlink(temporary).catch((error) => { if (error.code !== 'ENOENT') throw error; }); }
    });
  });
}

module.exports = { registerDrivingModelFiles };
