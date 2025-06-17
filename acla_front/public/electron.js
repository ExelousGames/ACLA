import { app, BrowserWindow, ipcMain } from 'electron';
import { PythonShell } from 'python-shell';
import path from 'path';
import { fileURLToPath } from 'url';
import isDev from 'electron-is-dev';

const devMode = app.isPackaged ? false : isDev;
let mainWindow;

// Store active Python shells, Multiple concurrent Python processes
const activeShells = new Map();
let nextShellId = 0;

function createWindow() {

  const __filename = fileURLToPath(import.meta.url); // get the resolved path to the file
  const __dirname = path.dirname(__filename); // get the name of the directory

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

    // Receive messages from Python script
    pyshell.on('message', (message) => {
      if (mainWindow) {
        mainWindow.webContents.send('python-message', message);
      }
    });

    pyshell.on('close', () => {
      activeShells.delete(shellId);
      resolve({ shellId }); // Resolve with the shellId

    });

    pyshell.on('error', (error) => {
      console.log(error);
      reject(error);
    });

    pyshell.end(function (err, code, signal) {
      if (err) throw err;
      console.log('The exit code was: ' + code);
      console.log('The exit signal was: ' + signal);
      console.log('finished');
    });
  });

  activeShells.set(shellId, newShell);
  console.log(shellId);
  return {
    shellId: shellId
  }
});

// Later, retrieve the promise using shellId
ipcMain.handle('get-python-result', async (event, shellId) => {
  const promise = activePromises.get(shellId);
  if (!promise) throw new Error("Invalid shellId");
  return await promise;
});

//send message to a python shell
ipcMain.handle('send-message-to-python', async (event, shellId, message) => {
  const pyshell = activeShells.get(shellId);
  if (!pyshell) {
    throw new Error(`Python shell ${shellId} not found`);
  }
  pyshell.send(message);
  return { success: true };
});

app.on('ready', createWindow);

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