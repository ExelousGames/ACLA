'use strict';

const fs = require('fs');
const path = require('path');
const { recordingStartFailure } = require('../../recording-protocol');

function isRegularFile(filePath) {
  try {
    return path.isAbsolute(filePath) && fs.statSync(filePath).isFile();
  } catch {
    return false;
  }
}

function unsupported(message) {
  return recordingStartFailure('unsupported-recording-game', message);
}

function getReaderLaunchConfig({ app, readerEntryPath, env = process.env } = {}) {
  if (!app || typeof app.getAppPath !== 'function' || typeof app.isPackaged !== 'boolean') {
    return unsupported('The ACC recording runtime could not be resolved.');
  }
  if (!isRegularFile(readerEntryPath)) {
    return unsupported('The ACC telemetry reader is not installed.');
  }

  const appRoot = app.getAppPath();
  const resourcesPath = process.resourcesPath;
  const scriptDirectory = app.isPackaged
    ? path.join(resourcesPath, 'py-scripts')
    : path.join(appRoot, 'src', 'py-scripts');
  const scriptName = 'ACCMemoryExtractor.py';
  const scriptPath = path.join(scriptDirectory, scriptName);

  const managedEnvironment = app.isPackaged
    ? path.join(resourcesPath, 'python-env')
    : path.join(appRoot, '.venv', 'py-scripts');
  const candidates = [
    env.ACLA_PYTHON_PATH,
    ...(process.platform === 'win32'
      ? [
        path.join(managedEnvironment, 'Scripts', 'python.exe'),
        path.join(managedEnvironment, 'Scripts', 'python3.exe'),
      ]
      : [
        path.join(managedEnvironment, 'bin', 'python3'),
        path.join(managedEnvironment, 'bin', 'python'),
      ]),
  ].filter((candidate) => typeof candidate === 'string' && candidate.length > 0);

  const pythonExecutable = candidates.find(isRegularFile);
  if (!pythonExecutable) {
    return unsupported('The managed Python runtime required by the ACC reader is unavailable.');
  }
  if (!isRegularFile(scriptPath)) {
    return unsupported('The ACC shared-memory extractor is unavailable.');
  }

  return {
    ok: true,
    config: {
      game: 'acc',
      readerEntryPath,
      readerOptions: {
        runtime: 'python',
        pythonExecutable,
        scriptDirectory,
        scriptName,
      },
    },
  };
}

module.exports = { getReaderLaunchConfig, isRegularFile };
