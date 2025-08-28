const { contextBridge, ipcRenderer } = require('electron');
/**
 * Electron's main process is a Node.js environment that has full operating system access. 
 * On top of Electron modules, you can also access Node.js built-ins, as well as any packages installed via npm. 
 * On the other hand, renderer processes run web pages and do not run Node.js by default for security reasons.
 * 
 * To bridge Electron's different process types together, we will need to use a special script called a preload.
 */


//contextBridge.exposeInMainWorld makes the function available in the global electronAPI object within the renderer.
contextBridge.exposeInMainWorld('electronAPI', {

    //run script in main process using async
    runPythonScript: (scriptPath, options) => ipcRenderer.invoke('run-python-script', scriptPath, options),

    //
    onPythonEnd: (callback) => {
        ipcRenderer.once('python-end', (event, ...args) => callback(...args))
    },

    onPythonMessage: (callback) => {
        const subscription = (event, ...args) => callback(...args);
        ipcRenderer.on('python-message', subscription)
        return () => {
            ipcRenderer.off('python-message', subscription);
        }
    },

    OnPythonMessageOnce: (callback) => {
        // Deliberately strip event as it includes `sender` 
        ipcRenderer.once('python-message', (event, ...args) => callback(...args))
    },

    //This function allows the renderer to send messages to the main process via the ipcRenderer.send API.
    //The main process would then need to listen for these messages using ipcMain.on.
    sendMessageToPython: (shellId, message) => ipcRenderer.invoke('send-message-to-python', shellId, message),
});