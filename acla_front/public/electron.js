const { app, BrowserWindow, ipcMain } = require('electron');
const { PythonShell } = require('python-shell');
const path = require('path');
const isDev = require('electron-is-dev');
const { spawn, execSync } = require('child_process');
const fs = require('fs');
const os = require('os');

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
let speechRecognitionTempFile = null;
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
  mainWindow.on('closed', () => mainWindow = null);
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

    // Check if enhanced script exists, otherwise create it
    if (!fs.existsSync(enhancedScriptPath)) {
      console.log('Enhanced speech recognition script not found, falling back to basic recognition');
      // Fall back to basic implementation if enhanced script is missing
      return await startBasicSpeechRecognition();
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

// Fallback basic speech recognition function
async function startBasicSpeechRecognition() {
  try {
    // Create a temporary file for the audio recording
    const tempDir = os.tmpdir();
    speechRecognitionTempFile = path.join(tempDir, `speech_${Date.now()}.wav`);

    // Create the basic Python script for speech recognition
    const speechScript = `
import speech_recognition as sr
import sys
import json

def recognize_speech():
    r = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            print(json.dumps({"status": "listening"}))
            sys.stdout.flush()
            
            # Better ambient noise adjustment
            r.adjust_for_ambient_noise(source, duration=1.0)
            
            # Improved recognition settings
            r.energy_threshold = 300
            r.dynamic_energy_threshold = True
            r.pause_threshold = 0.8
            r.phrase_threshold = 0.3
            r.non_speaking_duration = 0.8
            
            # Listen for audio input with longer timeout
            audio = r.listen(source, timeout=15, phrase_time_limit=15)
            
            print(json.dumps({"status": "processing"}))
            sys.stdout.flush()
            
            # Try Google first for better accuracy, then fall back to Sphinx
            try:
                text = r.recognize_google(audio, language='en-US')
                print(json.dumps({"status": "success", "transcript": text, "method": "google", "confidence": 0.8}))
            except (sr.UnknownValueError, sr.RequestError):
                try:
                    text = r.recognize_sphinx(audio)
                    print(json.dumps({"status": "success", "transcript": text, "method": "sphinx", "confidence": 0.6}))
                except sr.UnknownValueError:
                    print(json.dumps({"status": "error", "error": "Could not understand audio"}))
                except sr.RequestError as e:
                    print(json.dumps({"status": "error", "error": f"Recognition error: {e}"}))
                    
    except sr.WaitTimeoutError:
        print(json.dumps({"status": "error", "error": "No speech detected within timeout"}))
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}))

if __name__ == "__main__":
    recognize_speech()
`;

    // Write the Python script to a temporary file
    const scriptPath = path.join(tempDir, `speech_recognition_${Date.now()}.py`);
    fs.writeFileSync(scriptPath, speechScript);

    // Start the Python speech recognition process
    const pythonExec = getPythonExecutable();
    speechRecognitionProcess = spawn(pythonExec, [scriptPath], {
      stdio: 'pipe',
      shell: false
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
            recognitionMethod = result.method || 'basic';
            recognitionConfidence = result.confidence || 0.5;
          } else if (result.status === 'error') {
            console.error('Speech recognition error:', result.error);
          }

          // Send status updates to renderer
          if (mainWindow) {
            mainWindow.webContents.send('speech-recognition-status', result);
          }
        }
      } catch (parseError) {
        console.error('Error parsing speech recognition output:', parseError);
      }
    });

    speechRecognitionProcess.stderr.on('data', (data) => {
      console.error('Speech recognition stderr:', data.toString());
    });

    speechRecognitionProcess.on('close', (code) => {
      // Clean up temporary files
      try {
        if (fs.existsSync(scriptPath)) fs.unlinkSync(scriptPath);
        if (speechRecognitionTempFile && fs.existsSync(speechRecognitionTempFile)) {
          fs.unlinkSync(speechRecognitionTempFile);
        }
      } catch (cleanupError) {
        console.error('Error cleaning up temporary files:', cleanupError);
      }

      const process = speechRecognitionProcess;
      speechRecognitionProcess = null;
      speechRecognitionTempFile = null;

      // Send result to renderer
      if (mainWindow) {
        mainWindow.webContents.send('speech-recognition-complete', {
          success: code === 0,
          transcript: recognitionResult,
          method: recognitionMethod,
          confidence: recognitionConfidence,
          enhanced: false,
          error: code !== 0 ? `Process exited with code ${code}` : null
        });
      }
    });

    return { success: true, recordingId: Date.now().toString(), enhanced: false };

  } catch (error) {
    console.error('Error starting basic speech recognition:', error);
    return { success: false, error: error.message };
  }
}

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