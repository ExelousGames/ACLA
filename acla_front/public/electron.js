const { app, BrowserWindow, ipcMain } = require('electron');
const { PythonShell } = require('python-shell');
const path = require('path');
const isDev = require('electron-is-dev');
const { spawn, execFile, execSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const { detectSupportedDesktopGame } = require('./desktop-game-detection');

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
  return {
    supported: true,
    detectedGame: detectSupportedDesktopGame(tasklistOutput),
  };
});

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

// Speech recognition variables
let speechRecognitionProcess = null;
let isSpeechRecognitionAvailable = false;

// Function to check speech recognition availability
async function checkSpeechRecognitionAvailability() {
  try {
    execSync('python -c "import speech_recognition; print(\'OK\')"', { stdio: 'ignore' });
    isSpeechRecognitionAvailable = true;
    console.log('Speech recognition is available');
    return true;
  } catch (error) {
    isSpeechRecognitionAvailable = false;
    console.log('Speech recognition is not available:', error.message);
    return false;
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    title: 'Kestrel Motorsport Analyst',
    width: 900, height: 680,
    webPreferences: {
      //To send messages to the listener created above, you can use the ipcRenderer.send API. 
      // By default, the renderer process has no Node.js or Electron module access. 
      // As an app developer, you need to choose which APIs to expose from your preload script using the contextBridge API.
      // !!!!!! We don't directly expose the whole ipcRenderer.send API for security reasons. Make sure to limit the renderer's access to Electron APIs as much as possible.
      preload: path.join(__dirname, '../src/common/preload.js'),
    }
  });

  mainWindow.loadURL(devMode ? 'http://localhost:3000' : `file://${path.join(__dirname, '../build/index.html')}`);
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

// Speech Recognition IPC Handlers
ipcMain.handle('check-speech-recognition-availability', async () => {
  return await checkSpeechRecognitionAvailability();
});

ipcMain.handle('start-speech-recognition', async (event) => {
  try {
    if (!isSpeechRecognitionAvailable) {
      return { success: false, error: 'Speech recognition not available' };
    }

    if (speechRecognitionProcess) {
      return { success: false, error: 'Speech recognition already in progress' };
    }

    // Use the enhanced speech recognition script
    const enhancedScriptPath = path.join(resolveScriptDirectory(), 'enhanced_speech_recognition.py');

    if (!fs.existsSync(enhancedScriptPath)) {
      return { success: false, error: 'Enhanced speech recognition script not found' };
    }

    // Start the enhanced Python speech recognition process with 30-second timeout
    const pythonExec = getPythonExecutable();
    speechRecognitionProcess = spawn(pythonExec, [enhancedScriptPath, '30'], {
      stdio: 'pipe',
      shell: false,
      cwd: path.dirname(enhancedScriptPath)
    });

    let recognitionResult = null;
    let recognitionMethod = null;
    let recognitionConfidence = null;

    speechRecognitionProcess.stdout.on('data', (data) => {
      try {
        const lines = data.toString().split('\n').filter(line => line.trim());
        for (const line of lines) {
          const result = JSON.parse(line);

          if (result.status === 'success') {
            recognitionResult = result.transcript;
            recognitionMethod = result.method || 'unknown';
            recognitionConfidence = result.confidence || 0.5;
          } else if (result.status === 'error') {
            console.error('Enhanced speech recognition error:', result.error);
          } else if (result.status === 'warning') {
            console.warn('Enhanced speech recognition warning:', result.message);
          }

          // Send status updates to renderer with enhanced information
          if (mainWindow) {
            mainWindow.webContents.send('speech-recognition-status', {
              ...result,
              enhanced: true,
              timestamp: Date.now()
            });
          }
        }
      } catch (parseError) {
        console.error('Error parsing enhanced speech recognition output:', parseError, 'Raw data:', data.toString());
      }
    });

    speechRecognitionProcess.stderr.on('data', (data) => {
      const errorMessage = data.toString();
      console.error('Enhanced speech recognition stderr:', errorMessage);

      // Send error status to renderer
      if (mainWindow && errorMessage.trim()) {
        mainWindow.webContents.send('speech-recognition-status', {
          status: 'error',
          error: errorMessage.trim(),
          enhanced: true
        });
      }
    });

    speechRecognitionProcess.on('close', (code) => {
      const process = speechRecognitionProcess;
      speechRecognitionProcess = null;

      // Send result to renderer with enhanced information
      if (mainWindow) {
        mainWindow.webContents.send('speech-recognition-complete', {
          success: code === 0,
          transcript: recognitionResult,
          method: recognitionMethod,
          confidence: recognitionConfidence,
          enhanced: true,
          error: code !== 0 ? `Enhanced speech recognition process exited with code ${code}` : null
        });
      }
    });

    return { success: true, recordingId: Date.now().toString(), enhanced: true };

  } catch (error) {
    console.error('Error starting enhanced speech recognition:', error);
    return { success: false, error: error.message };
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
    if (!['expanded', 'folded', 'full_size'].includes(card.status)) return 'Unknown overlay display status.';
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

ipcMain.handle('stop-speech-recognition', async (event) => {
  try {
    if (!speechRecognitionProcess) {
      return { success: false, error: 'No speech recognition in progress' };
    }

    // Kill the process
    speechRecognitionProcess.kill('SIGTERM');

    // Wait a bit for graceful termination
    await new Promise(resolve => setTimeout(resolve, 1000));

    if (speechRecognitionProcess && !speechRecognitionProcess.killed) {
      speechRecognitionProcess.kill('SIGKILL');
    }

    return { success: true };

  } catch (error) {
    console.error('Error stopping speech recognition:', error);
    return { success: false, error: error.message };
  }
});

app.on('ready', () => {
  createWindow();
  // Check speech recognition availability on startup
  checkSpeechRecognitionAvailability();
});

app.on('before-quit', () => {
  isAppQuitting = true;
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
