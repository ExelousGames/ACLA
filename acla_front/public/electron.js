const { app, BrowserWindow, ipcMain } = require('electron');
const { PythonShell } = require('python-shell');
const path = require('path');
const isDev = require('electron-is-dev');
const { spawn, execSync } = require('child_process');
const fs = require('fs');
const os = require('os');

const devMode = app.isPackaged ? false : isDev;
let mainWindow;

// Store active Python shells, Multiple concurrent Python processes
const activeShells = new Map();
let nextShellId = 0;

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
ipcMain.handle('run-python-script', (event, script, options) => {
  const shellId = nextShellId++;
  const newShell = new Promise((resolve, reject) => {

    const pyshell = new PythonShell(script, options);

    // Receive messages from Python script -  line of text from stdout
    pyshell.on('message', (message) => {
      if (mainWindow) {
        mainWindow.webContents.send('python-message', shellId, message);
      }
    });

    //Indicates that the stream has no more data to read, Used when you're reading from process output streams, Doesn't necessarily mean the process has terminated
    pyshell.end(function (err, code, signal) {
      if (err) throw err;
    });

    //The process has definitely terminated, callback is invoked when the process is terminated.
    pyshell.on('close', () => {

      activeShells.delete(shellId);
      if (mainWindow) {
        mainWindow.webContents.send('python-end', shellId);
      }
      resolve({ shellId }); // Resolve with the shellId
    });


    pyshell.on('error', (error) => {
      activeShells.delete(shellId);
      reject(error);
    });


  });

  activeShells.set(shellId, newShell);
  return {
    shellId: shellId
  }
});

//renderer process send message to a python shell
ipcMain.handle('send-message-to-python', async (event, shellId, message) => {
  const pyshell = activeShells.get(shellId);
  if (!pyshell) {
    throw new Error(`Python shell ${shellId} not found`);
  }
  pyshell.send(message);
  return { success: true };
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
    const enhancedScriptPath = path.join(__dirname, '../src/py-scripts/enhanced_speech_recognition.py');

    // Check if enhanced script exists, otherwise create it
    if (!fs.existsSync(enhancedScriptPath)) {
      console.log('Enhanced speech recognition script not found, falling back to basic recognition');
      // Fall back to basic implementation if enhanced script is missing
      return await startBasicSpeechRecognition();
    }

    // Start the enhanced Python speech recognition process with 30-second timeout
    speechRecognitionProcess = spawn('python', [enhancedScriptPath, '30'], {
      stdio: 'pipe',
      shell: true,
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
    speechRecognitionProcess = spawn('python', [scriptPath], {
      stdio: 'pipe',
      shell: true
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