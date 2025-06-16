const electron = require('electron');
const app = electron.app;
const BrowserWindow = electron.BrowserWindow;
const ipcMain = electron.ipcMain;
const PythonShell = require('python-shell');

const path = require('path');
const url = require('url');
const isDev = app.isPackaged ? false : require('electron-is-dev');

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 900, height: 680,
    webPreferences: {
      //To send messages to the listener created above, you can use the ipcRenderer.send API. 
      // By default, the renderer process has no Node.js or Electron module access. 
      // As an app developer, you need to choose which APIs to expose from your preload script using the contextBridge API.
      // !!!!!! We don't directly expose the whole ipcRenderer.send API for security reasons. Make sure to limit the renderer's access to Electron APIs as much as possible.
      preload: path.join(nodePath, '/utils/preload.js')
    }
  });
  mainWindow.loadURL(isDev ? 'http://localhost:3000' : `file://${path.join(__dirname, '../build/index.html')}`);
  mainWindow.on('closed', () => mainWindow = null);
}

// Handle running Python scripts
ipcMain.handle('run-python-script', async (event, scriptPath, options) => {

  return new Promise((resolve, reject) => {

    const pyshell = new PythonShell(scriptPath, options);

    let output = [];

    // Receive messages from Python script
    pyshell.on('message', (message) => {
      if (mainWindow) {
        mainWindow.webContents.send('python-message', message);
      }
      output.push(message);
    });

    pyshell.on('close', () => {
      resolve(output);
    });

    pyshell.on('error', (error) => {
      reject(error);
    });
  });
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