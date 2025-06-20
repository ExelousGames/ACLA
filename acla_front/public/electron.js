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
        mainWindow.webContents.send('python-message', shellId, message);
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