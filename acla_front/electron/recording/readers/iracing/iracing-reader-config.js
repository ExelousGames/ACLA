'use strict';

const path = require('path');
const { isRegularFile } = require('../acc/acc-reader-config');
const { recordingStartFailure } = require('../../recording-protocol');

function getReaderLaunchConfig({ app, readerEntryPath, env = process.env, platform = process.platform, resourcesPath = process.resourcesPath } = {}) {
  const unavailable = (message) => recordingStartFailure('unsupported-recording-game', message);
  if (platform !== 'win32') return unavailable('iRacing live telemetry requires Windows.');
  if (!app || typeof app.getAppPath !== 'function' || typeof app.isPackaged !== 'boolean') {
    return unavailable('The iRacing recording runtime could not be resolved.');
  }
  if (!isRegularFile(readerEntryPath)) return unavailable('The iRacing telemetry reader is not installed.');
  const root = app.isPackaged ? resourcesPath : app.getAppPath();
  const scriptDirectory = path.join(root, ...(app.isPackaged ? ['py-scripts'] : ['src', 'py-scripts']));
  const runtime = path.join(root, ...(app.isPackaged ? ['python-env'] : ['.venv', 'py-scripts']));
  const pythonExecutable = [env.ACLA_PYTHON_PATH, path.join(runtime, 'Scripts', 'python.exe'), path.join(runtime, 'Scripts', 'python3.exe')]
    .filter((candidate) => typeof candidate === 'string').find(isRegularFile);
  if (!pythonExecutable) return unavailable('The managed Python runtime required by the iRacing capture process is unavailable.');
  const scriptName = 'iracing_sdk.py';
  if (!isRegularFile(path.join(scriptDirectory, scriptName))) return unavailable('The iRacing SDK capture script is unavailable.');
  return { ok: true, config: {
    game: 'iracing', readerEntryPath,
    readerOptions: { runtime: 'python', pythonExecutable, scriptDirectory, scriptName },
  } };
}

module.exports = { getReaderLaunchConfig };
