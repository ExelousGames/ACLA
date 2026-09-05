const { app, BrowserWindow, ipcMain, utilityProcess, MessageChannelMain } = require('electron');
const { PythonShell } = require('python-shell');
const path = require('path');
const isDev = require('electron-is-dev');
const { execFile } = require('child_process');
const fs = require('fs');
const os = require('os');
const crypto = require('crypto');
const { detectSupportedDesktopGame } = require('./desktop-game-detection');
const { RecordingSessionManager } = require('../electron/recording/recording-session-manager');
const {
  DESKTOP_GAME_SET,
  recordingStartFailure,
  validateRecordingStartConfig,
} = require('../electron/recording/recording-protocol');

function getPythonExecutable() {
  const manualOverride = process.env.ACLA_PYTHON_PATH;
  if (manualOverride && fs.existsSync(manualOverride)) {
    return manualOverride;
  }

  const envDir = app.isPackaged
    ? path.join(process.resourcesPath, 'python-env')
    : path.join(__dirname, '../.venv/py-scripts');

  const candidates = process.platform === 'win32'
    ? [
      path.join(envDir, 'Scripts', 'python.exe'),
      path.join(envDir, 'Scripts', 'python3.exe')
    ]
    : [
      path.join(envDir, 'bin', 'python3'),
      path.join(envDir, 'bin', 'python')
    ];

  for (const candidate of candidates) {
    if (candidate && fs.existsSync(candidate)) {
      return candidate;
    }
  }

  // Fall back to system Python if the managed environments are not available.
  return process.platform === 'win32' ? 'python' : 'python3';
}

const devMode = app.isPackaged ? false : isDev;
let mainWindow;
let isAppQuitting = false;
let recordingManager = null;
let latestDetectedGame = null;
let recordingIpcRegistered = false;
let quitAfterRecordingShutdown = false;
const recordedFileReads = new Map();

function getRecordingDirectory() {
  return path.join(app.getPath('userData'), 'acla-temp');
}

function isCurrentMainRenderer(sender) {
  return Boolean(
    sender
    && mainWindow
    && !mainWindow.isDestroyed()
    && mainWindow.webContents === sender
    && !sender.isDestroyed?.(),
  );
}

function isPathInside(parentDirectory, candidatePath) {
  const relative = path.relative(path.resolve(parentDirectory), path.resolve(candidatePath));
  return relative !== '' && !relative.startsWith('..') && !path.isAbsolute(relative);
}

function isRegularFileSync(filePath) {
  try {
    return path.isAbsolute(filePath) && fs.statSync(filePath).isFile();
  } catch {
    return false;
  }
}

async function getReaderLaunchConfig(game) {
  const readerDirectory = path.join(app.getAppPath(), 'electron', 'recording', 'readers', game);
  const readerEntryPath = path.join(readerDirectory, `${game}-reader-worker.js`);
  if (!isRegularFileSync(readerEntryPath)) {
    return recordingStartFailure(
      'unsupported-recording-game',
      `Live recording for ${game} is not installed yet.`,
    );
  }
  const configPath = path.join(readerDirectory, `${game}-reader-config.js`);
  if (!isRegularFileSync(configPath)) {
    return recordingStartFailure(
      'unsupported-recording-game',
      `The launch configuration for ${game} is unavailable.`,
    );
  }
  try {
    const producer = require(configPath);
    if (typeof producer.getReaderLaunchConfig !== 'function') {
      return recordingStartFailure('unsupported-recording-game', `The ${game} reader cannot be launched.`);
    }
    return await producer.getReaderLaunchConfig({ app, readerEntryPath });
  } catch (error) {
    return recordingStartFailure(
      'unsupported-recording-game',
      `The ${game} reader cannot be launched: ${error?.message || String(error)}`,
    );
  }
}

function getOrCreateRecordingManager() {
  if (!recordingManager) {
    recordingManager = new RecordingSessionManager({
      utilityProcess,
      MessageChannelMain,
      getMainWindow: () => mainWindow,
      getReaderLaunchConfig,
      recordingDirectory: getRecordingDirectory(),
    });
  }
  return recordingManager;
}

function getWindowsTasklist() {
  return new Promise((resolve, reject) => {
    execFile('tasklist', ['/FO', 'CSV', '/NH'], { encoding: 'utf8', windowsHide: true }, (error, stdout) => {
      if (error) {
        reject(error);
        return;
      }

      resolve(stdout);
    });
  });
}

ipcMain.handle('detect-desktop-game', async () => {
  if (process.platform !== 'win32') {
    return { supported: false, detectedGame: null };
  }

  const tasklistOutput = await getWindowsTasklist();
  latestDetectedGame = detectSupportedDesktopGame(tasklistOutput);
  return {
    supported: true,
    detectedGame: latestDetectedGame,
  };
});

function cleanupRecordedFileRead(readId, { kill = true } = {}) {
  const entry = recordedFileReads.get(readId);
  if (!entry) return;
  recordedFileReads.delete(readId);
  entry.utility.off?.('message', entry.onMessage);
  entry.utility.off?.('exit', entry.onExit);
  entry.utility.off?.('error', entry.onError);
  if (kill) {
    try { entry.utility.kill(); } catch { /* already exited */ }
  }
}

function cancelRecordedFileReadsForOwner(ownerWebContentsId) {
  for (const [readId, entry] of Array.from(recordedFileReads.entries())) {
    if (entry.ownerWebContentsId !== ownerWebContentsId) continue;
    try { entry.utility.postMessage({ type: 'cancel', readId }); } catch { /* utility failed */ }
    cleanupRecordedFileRead(readId);
  }
}

function waitForRecordedReadSignal(entry, key, timeoutMs = 5000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      entry.waiters.delete(key);
      reject(new Error(`Recorded-file reader ${key} handshake timed out.`));
    }, timeoutMs);
    entry.waiters.set(key, {
      resolve: (value) => {
        clearTimeout(timer);
        entry.waiters.delete(key);
        resolve(value);
      },
      reject: (error) => {
        clearTimeout(timer);
        entry.waiters.delete(key);
        reject(error);
      },
    });
  });
}

function registerRecordingIpc() {
  if (recordingIpcRegistered) return;
  recordingIpcRegistered = true;

  ipcMain.handle('recording-session-start', async (event, config) => {
    if (!isCurrentMainRenderer(event.sender)) throw new Error('Recording requests are allowed only from the active workspace.');
    const failure = validateRecordingStartConfig(config);
    if (failure) return failure;

    const tasklistOutput = process.platform === 'win32' ? await getWindowsTasklist() : '';
    latestDetectedGame = process.platform === 'win32'
      ? detectSupportedDesktopGame(tasklistOutput)
      : latestDetectedGame;
    if (latestDetectedGame !== config.game) {
      throw new Error('The requested recording game does not match the active simulator.');
    }
    return getOrCreateRecordingManager().startSession({
      game: config.game,
      ownerWebContentsId: event.sender.id,
    });
  });

  ipcMain.handle('recording-session-stop', async (event) => {
    if (!isCurrentMainRenderer(event.sender)) throw new Error('Recording stop is allowed only from the active workspace.');
    if (!recordingManager) throw new Error('No recording session is active.');
    return recordingManager.stopSession(event.sender.id);
  });

  ipcMain.handle('recorded-file-read-start', async (event, request) => {
    if (!isCurrentMainRenderer(event.sender)) throw new Error('Recorded-file reads are allowed only from the active workspace.');
    if (!request || typeof request !== 'object' || Array.isArray(request)
      || Object.keys(request).length !== 3
      || typeof request.filePath !== 'string' || !path.isAbsolute(request.filePath)
      || !DESKTOP_GAME_SET.has(request.game)
      || (request.purpose !== 'validate' && request.purpose !== 'consume')) {
      throw new TypeError('Recorded-file read request is invalid.');
    }
    const recordingDirectory = getRecordingDirectory();
    let resolvedPath;
    let resolvedRecordingDirectory;
    try {
      resolvedPath = fs.realpathSync(path.resolve(request.filePath));
      resolvedRecordingDirectory = fs.realpathSync(recordingDirectory);
    } catch {
      throw new Error('Recorded file is outside the authorized recording directory or is not a regular file.');
    }
    if (!isPathInside(resolvedRecordingDirectory, resolvedPath) || !isRegularFileSync(resolvedPath)) {
      throw new Error('Recorded file is outside the authorized recording directory or is not a regular file.');
    }
    const committedRowLimit = recordingManager?.hasActiveSession()
      ? recordingManager.getActiveRecordedFileReadLimit({
        ownerWebContentsId: event.sender.id,
        game: request.game,
        filePath: resolvedPath,
      })
      : null;

    const readId = crypto.randomUUID();
    const workerPath = path.join(app.getAppPath(), 'electron', 'recording', 'workers', 'recorded-file-reader-worker.js');
    if (!isRegularFileSync(workerPath)) throw new Error('Recorded-file reader is not installed.');
    const utility = utilityProcess.fork(workerPath, [], {
      serviceName: 'ACLA Recorded File Reader',
      stdio: 'pipe',
    });
    const channel = new MessageChannelMain();
    const entry = {
      readId,
      utility,
      ownerWebContentsId: event.sender.id,
      waiters: new Map(),
      onMessage: null,
      onExit: null,
      onError: null,
    };
    entry.onMessage = (rawMessage) => {
      const message = rawMessage?.data ?? rawMessage;
      if (message?.readId !== readId) return;
      entry.waiters.get(message.type)?.resolve(message);
      if (message.type === 'failed') {
        for (const waiter of entry.waiters.values()) {
          waiter.reject(new Error(message.error || 'Recorded-file reader failed.'));
        }
      }
      if (message.type === 'complete' || message.type === 'failed') cleanupRecordedFileRead(readId, { kill: false });
    };
    entry.onExit = () => {
      for (const waiter of entry.waiters.values()) waiter.reject(new Error('Recorded-file reader exited unexpectedly.'));
      cleanupRecordedFileRead(readId, { kill: false });
    };
    entry.onError = (error) => {
      for (const waiter of entry.waiters.values()) waiter.reject(error);
      cleanupRecordedFileRead(readId);
    };
    utility.on?.('message', entry.onMessage);
    utility.on?.('exit', entry.onExit);
    utility.on?.('error', entry.onError);
    recordedFileReads.set(readId, entry);

    try {
      const readyPromise = waitForRecordedReadSignal(entry, 'ready');
      const preloadReadyPromise = waitForRecordedReadSignal(entry, 'preload-ready');
      utility.postMessage({
        type: 'initialize',
        readId,
        filePath: resolvedPath,
        game: request.game,
        purpose: request.purpose,
        recordingDirectory: resolvedRecordingDirectory,
        ...(committedRowLimit !== null ? { rowLimit: committedRowLimit } : {}),
        portRoles: ['eventsToPreload'],
      }, [channel.port1]);
      mainWindow.webContents.postMessage(
        'recorded-file-read-port',
        { readId, game: request.game },
        [channel.port2],
      );
      await readyPromise;
      await preloadReadyPromise;
      setImmediate(() => {
        if (recordedFileReads.get(readId) === entry) {
          try {
            utility.postMessage({ type: 'start', readId });
          } catch (error) {
            entry.onError(error);
          }
        }
      });
      return { readId };
    } catch (error) {
      cleanupRecordedFileRead(readId);
      throw error;
    }
  });

  ipcMain.handle('recorded-file-read-cancel', async (event, readId) => {
    if (!isCurrentMainRenderer(event.sender)) throw new Error('Recorded-file cancellation is allowed only from the active workspace.');
    if (typeof readId !== 'string' || !readId) throw new TypeError('Recorded-file read id is invalid.');
    const entry = recordedFileReads.get(readId);
    if (!entry || entry.ownerWebContentsId !== event.sender.id) throw new Error('Recorded-file read was not found.');
    try { entry.utility.postMessage({ type: 'cancel', readId }); } finally { cleanupRecordedFileRead(readId); }
  });
}

// Store active Python shells, Multiple concurrent Python processes
const activeShells = new Map();
let nextShellId = 0;

function getShellEntry(shellId) {
  return activeShells.get(shellId) || null;
}

function emitPythonStart(shellId, entry) {

  const payload = {
    script: entry.script,
    args: entry.args,
    keepAlive: entry.keepAlive,
    pythonPath: entry.pythonPath,
    startedAt: entry.startedAt,
  };

  mainWindow.webContents.send('python-start', shellId, payload);
}

function finalizeShell(shellId, extra = {}) {
  const entry = activeShells.get(shellId);
  if (!entry || entry.finalized) {
    return;
  }

  entry.finalized = true;
  activeShells.delete(shellId);

  if (entry.pyshell && typeof entry.pyshell.removeAllListeners === 'function') {
    entry.pyshell.removeAllListeners();
  }
  entry.pyshell = null;

  const finishedAt = Date.now();
  const payload = {
    script: entry.script,
    args: entry.args,
    keepAlive: entry.keepAlive,
    pythonPath: entry.pythonPath,
    startedAt: entry.startedAt,
    finishedAt,
    durationMs: finishedAt - entry.startedAt,
    lastMessageAt: entry.lastMessageAt,
    messageCount: entry.messageCount,
    stopRequestedBy: entry.stopRequestedBy || null,
    reason: extra.reason || 'unknown',
    exitCode: extra.exitCode ?? null,
    signal: extra.signal ?? null,
    error: extra.error,
  };

  const logMessage = `Python shell ${shellId} (${entry.script}) ended [reason=${payload.reason}]`;
  if (payload.error) {
    console.error(logMessage, payload);
  } else if (devMode) {
    console.info(logMessage, payload);
  }

  if (mainWindow) {
    mainWindow.webContents.send('python-end', shellId, payload);
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    title: 'Kestrel Motorsport Analyst',
    width: 900, height: 680,
    minWidth: 820,
    minHeight: 600,
    frame: false,
    thickFrame: true,
    backgroundColor: '#0a0a0f',
    show: false,
    webPreferences: {
      //To send messages to the listener created above, you can use the ipcRenderer.send API. 
      // By default, the renderer process has no Node.js or Electron module access. 
      // As an app developer, you need to choose which APIs to expose from your preload script using the contextBridge API.
      // !!!!!! We don't directly expose the whole ipcRenderer.send API for security reasons. Make sure to limit the renderer's access to Electron APIs as much as possible.
      preload: path.join(__dirname, '../src/common/preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    }
  });

  mainWindow.loadURL(devMode ? 'http://localhost:3000' : `file://${path.join(__dirname, '../build/index.html')}`);
  const ownerWebContentsId = mainWindow.webContents.id;
  mainWindow.webContents.on('destroyed', () => {
    cancelRecordedFileReadsForOwner(ownerWebContentsId);
    if (recordingManager?.isOwnedBy(ownerWebContentsId)) {
      void recordingManager.shutdownAll();
    }
  });

  const sendWindowState = () => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('window-maximized-changed', mainWindow.isMaximized());
    }
  };

  mainWindow.once('ready-to-show', () => {
    mainWindow?.show();
    sendWindowState();
  });
  mainWindow.on('maximize', sendWindowState);
  mainWindow.on('unmaximize', sendWindowState);
  mainWindow.on('closed', () => {
    mainWindow = null;
    if (process.platform !== 'darwin' && !isAppQuitting) app.quit();
  });
}

// start running Python scripts
function resolveScriptDirectory(customPath) {
  if (app.isPackaged) {
    const packagedBase = path.join(process.resourcesPath, 'py-scripts');
    if (customPath && path.isAbsolute(customPath) && fs.existsSync(customPath)) {
      return customPath;
    }
    return packagedBase;
  }

  if (customPath) {
    if (path.isAbsolute(customPath) && fs.existsSync(customPath)) {
      return customPath;
    }

    const viaApp = path.join(app.getAppPath(), customPath);
    if (fs.existsSync(viaApp)) {
      return viaApp;
    }

    const viaCwd = path.resolve(process.cwd(), customPath);
    if (fs.existsSync(viaCwd)) {
      return viaCwd;
    }
  }

  return app.isPackaged
    ? path.join(process.resourcesPath, 'py-scripts')
    : path.join(__dirname, '../src/py-scripts');
}

ipcMain.handle('run-python-script', (event, script, options = {}) => {
  try {
    const shellId = nextShellId++;

    const pythonPath = options.pythonPath || getPythonExecutable();
    const scriptDirectory = resolveScriptDirectory(options.scriptPath);
    const shellOptions = {
      ...options,
      pythonPath,
      scriptPath: scriptDirectory,
    };

    const shellArgs = Array.isArray(options.args) ? [...options.args] : [];
    shellOptions.args = shellArgs;
    const keepAlive = shellArgs.includes('--stream');

    const pyshell = new PythonShell(script, shellOptions);

    const entry = {
      shellId,
      pyshell,
      script,
      args: shellArgs,
      keepAlive,
      pythonPath,
      startedAt: Date.now(),
      lastMessageAt: null,
      messageCount: 0,
      stopRequestedBy: null,
      finalized: false,
    };

    activeShells.set(shellId, entry);
    emitPythonStart(shellId, entry);

    pyshell.on('message', (message) => {
      const shellEntry = getShellEntry(shellId);
      if (shellEntry) {
        shellEntry.lastMessageAt = Date.now();
        shellEntry.messageCount += 1;
      }
      if (mainWindow) {
        mainWindow.webContents.send('python-message', shellId, message);
      }
    });

    if (!keepAlive) {
      pyshell.end((err) => {
        if (err) {
          console.error(`Python shell ${shellId} ended with error:`, err);
        }
      });
    }

    pyshell.on('close', (code, signal) => {
      const shellEntry = getShellEntry(shellId);
      const reason = shellEntry && shellEntry.stopRequestedBy ? 'terminated' : 'close';
      finalizeShell(shellId, {
        reason,
        exitCode: code ?? null,
        signal: signal ?? null,
      });
    });

    pyshell.on('error', (error) => {
      console.error(`Python shell ${shellId} error:`, error);
      finalizeShell(shellId, {
        reason: 'error',
        error: error && error.message ? error.message : String(error),
      });
    });

    return {
      shellId,
      metadata: {
        script,
        args: shellArgs,
        keepAlive,
        pythonPath,
        startedAt: entry.startedAt,
      },
    };
  } catch (error) {
    console.error('Failed to start python script', {
      script,
      options,
      error,
    });
    throw error;
  }
});

ipcMain.handle('write-temp-file', async (event, options = {}) => {
  const {
    content,
    directory,
    prefix = 'acla_temp',
    extension = '.json'
  } = options;

  if (typeof content !== 'string') {
    throw new Error('write-temp-file: content must be a string');
  }

  const baseDir = directory
    ? path.resolve(directory)
    : path.join(app.getPath('userData'), 'acla-temp');

  await fs.promises.mkdir(baseDir, { recursive: true });

  const safeExtension = extension.startsWith('.') ? extension : `.${extension}`;
  const fileName = `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2)}${safeExtension}`;
  const fullPath = path.join(baseDir, fileName);

  await fs.promises.writeFile(fullPath, content, 'utf8');

  return {
    success: true,
    path: fullPath
  };
});

ipcMain.handle('delete-temp-file', async (event, filePath) => {
  if (!filePath) {
    return { success: false, error: 'No file path provided' };
  }

  try {
    await fs.promises.unlink(filePath);
    return { success: true };
  } catch (error) {
    if (error.code === 'ENOENT') {
      return { success: true, skipped: true };
    }
    return { success: false, error: error.message };
  }
});

ipcMain.handle('validate-telemetry-file', async (event, filePath) => {
  if (typeof filePath !== 'string' || !filePath || !path.isAbsolute(filePath)) {
    return {
      exists: false,
      readable: false,
      hasData: false,
      size: 0,
      error: 'Telemetry file path must be absolute',
    };
  }

  try {
    const stat = await fs.promises.stat(filePath);
    if (!stat.isFile()) {
      return {
        exists: true,
        readable: false,
        hasData: false,
        size: 0,
        error: 'Telemetry path is not a file',
      };
    }

    const handle = await fs.promises.open(filePath, 'r');
    await handle.close();
    return {
      exists: true,
      readable: true,
      hasData: stat.size > 0,
      size: stat.size,
    };
  } catch (error) {
    const missing = error && error.code === 'ENOENT';
    return {
      exists: !missing,
      readable: false,
      hasData: false,
      size: 0,
      error: error && error.message ? error.message : String(error),
    };
  }
});

//renderer process send message to a python shell
ipcMain.handle('send-message-to-python', async (event, shellId, message) => {
  const shellEntry = getShellEntry(shellId);
  const pyshell = shellEntry && shellEntry.pyshell;
  if (!pyshell) {
    console.warn(`Attempted to send message to missing python shell ${shellId}`);
    return { success: false, error: `Python shell ${shellId} not found` };
  }
  try {
    pyshell.send(message);
    return { success: true };
  } catch (error) {
    console.error(`Failed to send message to python shell ${shellId}:`, error);
    return { success: false, error: error?.message || 'Unknown error sending message' };
  }
});

ipcMain.handle('stop-python-script', async (event, shellId, initiator = 'renderer') => {
  const shellEntry = getShellEntry(shellId);
  const pyshell = shellEntry && shellEntry.pyshell;
  if (!pyshell) {
    return { success: false, error: `Python shell ${shellId} not found` };
  }

  try {
    if (shellEntry) {
      const resolvedInitiator = initiator || (event?.senderFrame?.url ?? 'renderer');
      shellEntry.stopRequestedBy = resolvedInitiator;
    }

    if (typeof pyshell.terminate === 'function') {
      pyshell.terminate();
    } else if (pyshell.childProcess && typeof pyshell.childProcess.kill === 'function') {
      pyshell.childProcess.kill();
    } else {
      return { success: false, error: 'Unable to terminate python shell' };
    }

    return { success: true };
  } catch (error) {
    console.error(`Failed to stop python shell ${shellId}:`, error);
    return { success: false, error: error?.message || 'Unknown error stopping python shell' };
  }
});

// ── Floating AI-chat window ─────────────────────────────────────────────
// A frameless, always-on-top window that loads the same React bundle under
// #/floating-chat. Typed display snapshots arrive through the IPC broker.
//
// "Always on top" uses the highest Windows level ('screen-saver'). Note: a
// true exclusive-fullscreen game will still cover this — users need to run
// the game in borderless windowed mode for the overlay to show through.
let floatingChatWindow = null;
let overlayEnabled = false;
let overlayRendererReady = false;
let currentOverlayPresentation = null;
let overlayPresentationSequence = 0;
let lastOverlayPresentationSnapshot = null;
const overlayPendingPresentations = new Map();
const overlayPresentationQueue = [];
const OVERLAY_ACK_TIMEOUT_MS = 15000;

function isOverlayRendererSender(event) {
  return Boolean(
    floatingChatWindow
    && !floatingChatWindow.isDestroyed()
    && event.sender.id === floatingChatWindow.webContents.id
  );
}

function isSerializableOverlayValue(value) {
  try {
    const seen = new Set();
    const visit = (candidate) => {
      if (candidate === null || typeof candidate === 'string' || typeof candidate === 'boolean') return true;
      if (typeof candidate === 'number') return Number.isFinite(candidate);
      if (Array.isArray(candidate)) {
        if (seen.has(candidate)) return false;
        seen.add(candidate);
        const valid = candidate.every(visit);
        seen.delete(candidate);
        return valid;
      }
      if (typeof candidate !== 'object') return false;
      const prototype = Object.getPrototypeOf(candidate);
      if (prototype !== Object.prototype && prototype !== null) return false;
      if (seen.has(candidate)) return false;
      seen.add(candidate);
      const valid = Object.values(candidate).every(visit);
      seen.delete(candidate);
      return valid;
    };
    if (!visit(value)) return false;
    const json = JSON.stringify(value);
    return typeof json === 'string' && json.length <= 5_000_000;
  } catch {
    return false;
  }
}

function overlayPresentationKey(presentation) {
  return `${presentation.presentationId}\u0000${presentation.presentationRevision}`;
}

function isMainWindowSender(event) {
  return Boolean(
    mainWindow
    && !mainWindow.isDestroyed()
    && event.sender === mainWindow.webContents
  );
}

ipcMain.handle('window-control', (event, action) => {
  if (!isMainWindowSender(event)) return { success: false, isMaximized: false };

  switch (action) {
    case 'minimize':
      mainWindow.minimize();
      break;
    case 'toggle-maximize':
      if (mainWindow.isMaximized()) mainWindow.unmaximize();
      else mainWindow.maximize();
      break;
    case 'close':
      mainWindow.close();
      break;
    case 'is-maximized':
      return mainWindow.isMaximized();
    default:
      return { success: false, isMaximized: mainWindow.isMaximized() };
  }

  return { success: true, isMaximized: mainWindow?.isMaximized() || false };
});

function validateOverlayPresentationSnapshot(presentation) {
  if (!presentation || typeof presentation !== 'object' || Array.isArray(presentation)) return 'Malformed overlay presentation.';
  if (typeof presentation.presentationId !== 'string' || !presentation.presentationId.trim()) return 'Overlay presentationId is required.';
  if (!Number.isInteger(presentation.presentationRevision) || presentation.presentationRevision < 1) return 'Overlay presentationRevision must be a positive integer.';
  if (!presentation.session || presentation.session.presentationId !== presentation.presentationId) return 'Overlay session identity does not match.';
  if (!Array.isArray(presentation.cards)) return 'Overlay cards must be an array.';
  const names = new Set();
  for (const card of presentation.cards) {
    if (!card || typeof card !== 'object' || Array.isArray(card)) return 'Malformed overlay card.';
    if (typeof card.componentName !== 'string' || !card.componentName.trim()) return 'Overlay card componentName is required.';
    if (typeof card.componentType !== 'string' || !card.componentType.trim()) return 'Overlay card componentType is required.';
    if (names.has(card.componentName)) return `Duplicate overlay componentName '${card.componentName}'.`;
    names.add(card.componentName);
    if (!Number.isInteger(card.revision) || card.revision < 1) return 'Overlay card revision must be a positive integer.';
    if (!['expanded', 'folded', 'focus'].includes(card.status)) return 'Unknown overlay display status.';
    if (!['pinned', 'flow'].includes(card.placement)) return 'Unknown overlay placement.';
  }
  if (!isSerializableOverlayValue(presentation)) return 'Overlay presentation must be JSON-safe.';
  return null;
}

function validateOverlaySessionDescriptor(descriptor) {
  if (!descriptor || typeof descriptor !== 'object' || Array.isArray(descriptor)) return 'Overlay session descriptor is required.';
  if (typeof descriptor.aiSessionId !== 'string' || !descriptor.aiSessionId.trim()) return 'Overlay AI session ID is required.';
  if (!['front_desk', 'live', 'recorded', 'user_summary', 'agent'].includes(descriptor.mode)) return 'Unknown overlay session mode.';
  if (!descriptor.displayIdentity || typeof descriptor.displayIdentity !== 'object' || Array.isArray(descriptor.displayIdentity)) {
    return 'Overlay display identity is required.';
  }
  if (!isSerializableOverlayValue(descriptor)) return 'Overlay session descriptor must be JSON-safe.';
  return null;
}

function settleOverlayPresentation(key, acknowledgement) {
  const pending = overlayPendingPresentations.get(key);
  if (!pending) return;
  overlayPendingPresentations.delete(key);
  if (pending.timer) clearTimeout(pending.timer);
  if (acknowledgement?.accepted
    && pending.presentation.presentationId === currentOverlayPresentation?.presentationId
    && (
      lastOverlayPresentationSnapshot?.presentationId !== pending.presentation.presentationId
      || pending.presentation.presentationRevision >= lastOverlayPresentationSnapshot.presentationRevision
    )) {
    lastOverlayPresentationSnapshot = pending.presentation;
  }
  pending.resolve(acknowledgement);
}

function rejectOverlayPresentations(error, presentationId = null) {
  const retainedPresentations = presentationId
    ? overlayPresentationQueue.filter((presentation) => presentation.presentationId !== presentationId)
    : [];
  overlayPresentationQueue.splice(0, overlayPresentationQueue.length, ...retainedPresentations);
  Array.from(overlayPendingPresentations.entries()).forEach(([key, pending]) => {
    if (presentationId && pending.presentationId !== presentationId) return;
    if (pending.timer) clearTimeout(pending.timer);
    pending.resolve({
      presentationId: pending.presentationId,
      presentationRevision: pending.presentationRevision,
      accepted: false,
      error,
    });
    overlayPendingPresentations.delete(key);
  });
}

function forwardOverlayPresentation(presentation) {
  const key = overlayPresentationKey(presentation);
  const pending = overlayPendingPresentations.get(key);
  if (!pending || !floatingChatWindow || floatingChatWindow.isDestroyed()) return;
  if (presentation.presentationId !== currentOverlayPresentation?.presentationId) {
    settleOverlayPresentation(key, {
      presentationId: presentation.presentationId,
      presentationRevision: presentation.presentationRevision,
      accepted: false,
      error: `Overlay presentation '${presentation.presentationId}' is no longer active.`,
    });
    return;
  }
  if (!pending.timer) {
    pending.timer = setTimeout(() => {
      settleOverlayPresentation(key, {
        presentationId: presentation.presentationId,
        presentationRevision: presentation.presentationRevision,
        accepted: false,
        error: 'Overlay acknowledgement timed out.',
      });
    }, OVERLAY_ACK_TIMEOUT_MS);
  }
  floatingChatWindow.webContents.send('overlay-presentation-snapshot', presentation);
}

function syncOverlayVisibility() {
  if (!floatingChatWindow || floatingChatWindow.isDestroyed()) return;
  if (overlayEnabled && overlayRendererReady) {
    floatingChatWindow.showInactive();
  } else {
    floatingChatWindow.hide();
  }
}

function flushOverlayPresentationQueue() {
  if (!overlayRendererReady || !floatingChatWindow || floatingChatWindow.isDestroyed()) return;
  while (overlayPresentationQueue.length > 0) {
    forwardOverlayPresentation(overlayPresentationQueue.shift());
  }
}

function createFloatingChatWindow() {
  if (floatingChatWindow && !floatingChatWindow.isDestroyed()) {
    return floatingChatWindow;
  }

  overlayRendererReady = false;

  // Start hidden; once ready, the renderer reports the unified shell dimensions.
  floatingChatWindow = new BrowserWindow({
    title: 'Kestrel Motorsport Analyst',
    width: 300,
    height: 64,
    frame: false,
    // Without thickFrame:false, Windows still attaches the WS_THICKFRAME
    // resize-handle chrome to frameless+transparent windows. When the
    // window loses focus that chrome renders as a visible white border
    // (the "inactive window" frame). Disabling it kills the white frame
    // on defocus — also disables programmatic OS resize, which we don't
    // want anyway since the pill is a fixed-size overlay.
    thickFrame: false,
    resizable: false,
    transparent: true,
    backgroundColor: '#00000000',
    hasShadow: false,
    roundedCorners: false,
    alwaysOnTop: true,
    skipTaskbar: true,
    show: false,
    useContentSize: true,
    webPreferences: {
      preload: path.join(__dirname, '../src/common/preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  // Force a fully transparent background post-creation — on Windows the
  // constructor `backgroundColor` is occasionally ignored, leaving the
  // window's surface painted with the system theme color (white in light
  // mode), which reads as a frame around the pill.
  floatingChatWindow.setBackgroundColor('#00000000');

  // 'screen-saver' is the highest level on Windows — required to float over
  // borderless-windowed games. macOS uses the same enum.
  floatingChatWindow.setAlwaysOnTop(true, 'screen-saver');
  floatingChatWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });

  const base = devMode
    ? 'http://localhost:3000'
    : `file://${path.join(__dirname, '../build/index.html')}`;
  floatingChatWindow.loadURL(`${base}#/floating-chat`);

  floatingChatWindow.on('close', (event) => {
    if (!isAppQuitting) event.preventDefault();
  });

  floatingChatWindow.on('closed', () => {
    floatingChatWindow = null;
    overlayRendererReady = false;
    rejectOverlayPresentations('Overlay renderer shut down before acknowledging the presentation.');
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('floating-chat-closed');
    }
  });

  return floatingChatWindow;
}

ipcMain.handle('overlay-session-create', (_event, descriptor) => {
  const validationError = validateOverlaySessionDescriptor(descriptor);
  if (validationError) return { success: false, error: validationError };
  try {
    createFloatingChatWindow();
    const previousPresentationId = currentOverlayPresentation?.presentationId;
    if (previousPresentationId) {
      rejectOverlayPresentations('Overlay presentation was replaced.', previousPresentationId);
    }
    lastOverlayPresentationSnapshot = null;
    currentOverlayPresentation = {
      ...JSON.parse(JSON.stringify(descriptor)),
      presentationId: `overlay-presentation-${Date.now().toString(36)}-${(++overlayPresentationSequence).toString(36)}`,
    };
    syncOverlayVisibility();
    return { success: true, presentation: currentOverlayPresentation };
  } catch (error) {
    console.error('Failed to create overlay session:', error);
    return { success: false, error: error?.message || 'Unknown overlay session error' };
  }
});

ipcMain.handle('overlay-session-destroy', (_event, presentationId) => {
  if (typeof presentationId !== 'string' || !presentationId.trim()) {
    return { success: false, error: 'Overlay presentationId is required.' };
  }
  if (currentOverlayPresentation?.presentationId !== presentationId) {
    return { success: true, ended: false };
  }
  rejectOverlayPresentations('Overlay presentation ended before acknowledging the presentation.', presentationId);
  currentOverlayPresentation = null;
  lastOverlayPresentationSnapshot = null;
  return { success: true, ended: true };
});

ipcMain.handle('overlay-session-set-enabled', (_event, enabled) => {
  overlayEnabled = Boolean(enabled);
  if (overlayEnabled) createFloatingChatWindow();
  syncOverlayVisibility();
  return { success: true, enabled: overlayEnabled };
});

ipcMain.handle('overlay-session-is-enabled', () => overlayEnabled);

ipcMain.handle('overlay-presentation-submit', (event, presentation) => {
  const validationError = validateOverlayPresentationSnapshot(presentation);
  if (validationError) {
    return {
      presentationId: presentation?.presentationId || 'unknown',
      presentationRevision: presentation?.presentationRevision || 0,
      accepted: false,
      error: validationError,
    };
  }
  if (presentation.presentationId !== currentOverlayPresentation?.presentationId) {
    return {
      presentationId: presentation.presentationId,
      presentationRevision: presentation.presentationRevision,
      accepted: false,
      error: `Overlay presentation '${presentation.presentationId}' is no longer active.`,
    };
  }
  if (!floatingChatWindow || floatingChatWindow.isDestroyed()) {
    return { presentationId: presentation.presentationId, presentationRevision: presentation.presentationRevision, accepted: false, error: 'Overlay window is unavailable.' };
  }
  if (isOverlayRendererSender(event)) {
    return { presentationId: presentation.presentationId, presentationRevision: presentation.presentationRevision, accepted: false, error: 'Overlay renderer cannot submit presentations.' };
  }
  const key = overlayPresentationKey(presentation);
  if (overlayPendingPresentations.has(key)) {
    return { presentationId: presentation.presentationId, presentationRevision: presentation.presentationRevision, accepted: false, error: 'Duplicate overlay presentation revision.' };
  }
  return new Promise((resolve) => {
    overlayPendingPresentations.set(key, {
      presentationId: presentation.presentationId,
      presentationRevision: presentation.presentationRevision,
      presentation: JSON.parse(JSON.stringify(presentation)),
      resolve,
      timer: null,
    });
    if (overlayRendererReady) {
      forwardOverlayPresentation(presentation);
    } else {
      overlayPresentationQueue.push(presentation);
    }
  });
});

ipcMain.on('overlay-renderer-ready', (event) => {
  if (!isOverlayRendererSender(event)) return;
  overlayRendererReady = true;
  if (overlayPresentationQueue.length === 0 && lastOverlayPresentationSnapshot) {
    floatingChatWindow.webContents.send('overlay-presentation-snapshot', lastOverlayPresentationSnapshot);
  }
  flushOverlayPresentationQueue();
  syncOverlayVisibility();
});

ipcMain.on('overlay-presentation-acknowledgement', (event, acknowledgement) => {
  if (!isOverlayRendererSender(event)) return;
  if (!acknowledgement || typeof acknowledgement.presentationId !== 'string'
    || !Number.isInteger(acknowledgement.presentationRevision)) return;
  const key = overlayPresentationKey(acknowledgement);
  const pending = overlayPendingPresentations.get(key);
  if (!pending || acknowledgement.presentationId !== pending.presentationId) return;
  settleOverlayPresentation(key, acknowledgement);
});

ipcMain.on('overlay-renderer-event', (event, rendererEvent) => {
  if (!isOverlayRendererSender(event) || !isSerializableOverlayValue(rendererEvent)) return;
  if (typeof rendererEvent?.presentationId !== 'string'
    || rendererEvent.presentationId !== currentOverlayPresentation?.presentationId
    || typeof rendererEvent.componentName !== 'string'
    || !rendererEvent.componentName.trim()
    || !Number.isInteger(rendererEvent.revision)
    || typeof rendererEvent.event !== 'string'
    || !rendererEvent.event.trim()) return;
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('overlay-renderer-event', rendererEvent);
  }
});

// Track the widest card and total stack height. Horizontal resizing keeps
// the visual center; vertical resizing keeps the dragged top edge fixed.
ipcMain.handle('resize-floating-chat', (event, payload) => {
  if (!isOverlayRendererSender(event) || !floatingChatWindow || floatingChatWindow.isDestroyed()) {
    return { success: false };
  }
  const width = Math.max(280, Math.round(Number(payload?.width) || 280));
  const height = Math.max(58, Math.round(Number(payload?.height) || 58));
  const bounds = floatingChatWindow.getBounds();
  const newX = bounds.x - Math.round((width - bounds.width) / 2);
  floatingChatWindow.setBounds({ x: newX, y: bounds.y, width, height });
  return { success: true };
});

app.on('ready', () => {
  registerRecordingIpc();
  createWindow();
});

app.on('before-quit', (event) => {
  if (!quitAfterRecordingShutdown && recordingManager?.hasActiveSession()) {
    event.preventDefault();
    isAppQuitting = true;
    for (const readId of Array.from(recordedFileReads.keys())) cleanupRecordedFileRead(readId);
    void recordingManager.shutdownAll().finally(() => {
      quitAfterRecordingShutdown = true;
      app.quit();
    });
    return;
  }
  isAppQuitting = true;
  for (const readId of Array.from(recordedFileReads.keys())) cleanupRecordedFileRead(readId);
  rejectOverlayPresentations('Application is shutting down.');
  if (floatingChatWindow && !floatingChatWindow.isDestroyed()) {
    floatingChatWindow.close();
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});
