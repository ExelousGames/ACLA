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

    //listen to python end in main process by using ipcRenderer.on
    onPythonEnd: (callback) => {
        const wrappedCallback = (event, shellId) => callback(shellId);
        ipcRenderer.on('python-end', wrappedCallback)
    },
    offPythonEnd: (callback) => {
        const wrappedCallback = (event, shellId) => callback(shellId);
        ipcRenderer.off('python-end', wrappedCallback)
    },

    onPythonMessage: (callback) => {
        const listener = (event, shellId, message) => callback(shellId, message);
        callback.__listener = listener;  // Attach the listener to the callback
        ipcRenderer.on('python-message', listener)
    },
    offPythonMessage: (callback) => {
        // Retrieve the stored listener
        console.log(callback);
        const listener = callback.__listener;
        if (listener) {
            ipcRenderer.off('python-message', listener)
            delete callback.__listener;  // Clean up
        }

    },

    //This function allows the renderer to send messages to the main process via the ipcRenderer.send API.
    //The main process would then need to listen for these messages using ipcMain.on.
    sendMessageToPython: (shellId, message) => ipcRenderer.invoke('send-message-to-python', shellId, message),
});