const { contextBridge, ipcRenderer } = require('electron');
/**
 * Electron's main process is a Node.js environment that has full operating system access. 
 * On top of Electron modules, you can also access Node.js built-ins, as well as any packages installed via npm. 
 * On the other hand, renderer processes run web pages and do not run Node.js by default for security reasons.
 * 
 * To bridge Electron's different process types together, we will need to use a special script called a preload.
 */
contextBridge.exposeInMainWorld('electronAPI', {
    runPythonScript: (scriptPath, options) => ipcRenderer.invoke('run-python-script', scriptPath, options),
    getPythonScriptResult: (shellId) => ipcRenderer.invoke('get-python-result', shellId),
    onPythonMessage: (callback) => ipcRenderer.on('python-message', (event, message) => callback(message)),
    sendMessageToPython: (shellId, message) => ipcRenderer.invoke('send-message-to-python', shellId, message),
});