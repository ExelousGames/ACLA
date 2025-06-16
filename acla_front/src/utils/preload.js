const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    runPythonScript: (scriptPath) => ipcRenderer.invoke('run-python-script', scriptPath),
    onPythonMessage: (callback) => ipcRenderer.on('python-message', (event, message) => callback(message))
});