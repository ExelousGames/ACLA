const { contextBridge, ipcRenderer } = require('electron');
const { DESKTOP_GAME_SET, isRecordingStartFailure } = require('../../electron/recording/recording-protocol');
const { validateStandardTelemetrySample } = require('../../electron/recording/telemetry-contract');
/**
 * Electron's main process is a Node.js environment that has full operating system access. 
 * On top of Electron modules, you can also access Node.js built-ins, as well as any packages installed via npm. 
 * On the other hand, renderer processes run web pages and do not run Node.js by default for security reasons.
 * 
 * To bridge Electron's different process types together, we will need to use a special script called a preload.
 */


const recordingViewCallbacks = new Set();
const recordingEndedCallbacks = new Set();
const recordedFileCallbacks = new Set();
const recordedFilePorts = new Map();
let activeRecordingPort = null;
let recordingEndDelivered = false;

const fanOut = (callbacks, payload) => {
    for (const callback of Array.from(callbacks)) {
        try {
            callback(payload);
        } catch (error) {
            console.error('Electron bridge callback failed', error);
        }
    }
};

const fanOutAsync = async (callbacks, payload) => {
    for (const callback of Array.from(callbacks)) {
        try {
            await callback(payload);
        } catch (error) {
            console.error('Electron bridge callback failed', error);
        }
    }
};

const subscribe = (callbacks, callback, name) => {
    if (typeof callback !== 'function') throw new TypeError(`${name} requires a callback function`);
    callbacks.add(callback);
    let subscribed = true;
    return () => {
        if (!subscribed) return;
        subscribed = false;
        callbacks.delete(callback);
    };
};

const isValidGame = (game) => DESKTOP_GAME_SET.has(game);
const isSafeCount = (value) => Number.isSafeInteger(value) && value >= 0;

const closeLivePort = (terminal) => {
    const active = activeRecordingPort;
    if (!active) return;
    active.terminal = true;
    activeRecordingPort = null;
    try { active.port.close(); } catch { /* already closed */ }
    // A direct error does not yet contain the writer's final partial-file
    // summary. The main lifecycle event delivers that richer result once.
    if (terminal && !terminal.error && !recordingEndDelivered) {
        recordingEndDelivered = true;
        fanOut(recordingEndedCallbacks, terminal);
    }
};

const validateViewMessage = (message, state) => {
    if (!message || typeof message !== 'object' || message.game !== state.game) return false;
    if (message.type === 'frame') {
        const keys = Object.keys(message);
        if (keys.length !== 6
            || !['type', 'game', 'sample', 'sequence', 'committedSequence', 'committedCount']
                .every((key) => Object.prototype.hasOwnProperty.call(message, key))
            || !validateStandardTelemetrySample(message.sample).ok
            || !isSafeCount(message.sequence)
            || !isSafeCount(message.committedSequence)
            || !isSafeCount(message.committedCount)
            || message.sequence !== state.latestSequence + 1
            || message.committedSequence < state.committedSequence
            || message.committedCount !== message.committedSequence) return false;
        state.latestSequence = message.sequence;
        state.committedSequence = message.committedSequence;
        return true;
    }
    if (message.type === 'terminal') {
        if (typeof message.error === 'string' && message.error.length > 0) {
            return Object.keys(message).length === 3;
        }
        return Object.keys(message).length === 4
            && typeof message.filePath === 'string'
            && message.filePath.length > 0
            && isSafeCount(message.writtenSamples)
            && message.writtenSamples >= state.committedSequence;
    }
    return false;
};

ipcRenderer.on('recording-view-port', (event, descriptor) => {
    const port = event?.ports?.[0];
    if (!descriptor || Object.keys(descriptor).length !== 1
        || !isValidGame(descriptor.game) || event?.ports?.length !== 1
        || !port || activeRecordingPort) {
        try { port?.close(); } catch { /* invalid transfer */ }
        return;
    }

    const state = {
        game: descriptor.game,
        port,
        terminal: false,
        latestSequence: 0,
        committedSequence: 0,
    };
    activeRecordingPort = state;
    recordingEndDelivered = false;
    port.onmessage = (messageEvent) => {
        const message = messageEvent?.data;
        if (!validateViewMessage(message, state)) {
            closeLivePort({ game: state.game, error: 'Invalid live recording update.' });
            return;
        }
        if (message.type === 'frame') fanOut(recordingViewCallbacks, message);
        else closeLivePort(message);
    };
    port.onmessageerror = () => closeLivePort({ game: state.game, error: 'Live recording port failed.' });
    port.onclose = () => {
        if (activeRecordingPort === state && !state.terminal) {
            activeRecordingPort = null;
        }
    };
    port.start();
    port.postMessage({ type: 'ready', game: state.game });
});

const validateEndedResult = (result) => Boolean(
    result
    && isValidGame(result.game)
    && ((typeof result.filePath === 'string' && isSafeCount(result.writtenSamples))
        || (typeof result.error === 'string' && result.error.length > 0)),
);

ipcRenderer.on('recording-session-ended', (_event, result) => {
    if (!validateEndedResult(result)) return;
    if (activeRecordingPort?.game === result.game) closeLivePort(result);
    else if (!recordingEndDelivered) {
        recordingEndDelivered = true;
        fanOut(recordingEndedCallbacks, result);
    }
});

const validateRecordedFileEvent = (payload, readId, game) => {
    if (!payload || payload.readId !== readId || typeof payload.type !== 'string') return false;
    switch (payload.type) {
        case 'format':
            return Object.keys(payload).length === 4
                && payload.format === 'standard-flat' && payload.game === game;
        case 'chunk':
            return Object.keys(payload).length === 4
                && Array.isArray(payload.rows)
                && payload.rows.length > 0
                && isSafeCount(payload.chunkIndex)
                && payload.rows.every((row) => validateStandardTelemetrySample(row).ok);
        case 'progress':
            return Object.keys(payload).length === 5
                && isSafeCount(payload.rowsRead)
                && isSafeCount(payload.bytesRead)
                && isSafeCount(payload.totalBytes)
                && payload.bytesRead <= payload.totalBytes;
        case 'complete':
            return Object.keys(payload).length === 6
                && payload.format === 'standard-flat'
                && payload.game === game
                && isSafeCount(payload.rowCount)
                && isSafeCount(payload.totalBytes);
        case 'error':
            return Object.keys(payload).every((key) => ['type', 'readId', 'message', 'row', 'byteOffset'].includes(key))
                && typeof payload.message === 'string'
                && payload.message.length > 0
                && (payload.row === undefined || isSafeCount(payload.row))
                && (payload.byteOffset === undefined || isSafeCount(payload.byteOffset));
        default:
            return false;
    }
};

const closeRecordedFilePort = (readId) => {
    const state = recordedFilePorts.get(readId);
    if (!state) return;
    state.terminal = true;
    recordedFilePorts.delete(readId);
    try { state.port.close(); } catch { /* already closed */ }
};

ipcRenderer.on('recorded-file-read-port', (event, descriptor) => {
    const port = event?.ports?.[0];
    if (!descriptor || Object.keys(descriptor).length !== 2
        || typeof descriptor.readId !== 'string' || !descriptor.readId
        || !isValidGame(descriptor.game) || event?.ports?.length !== 1
        || !port || recordedFilePorts.has(descriptor.readId)) {
        try { port?.close(); } catch { /* invalid transfer */ }
        return;
    }
    const state = { port, game: descriptor.game, terminal: false };
    recordedFilePorts.set(descriptor.readId, state);
    port.onmessage = async (messageEvent) => {
        const payload = messageEvent?.data;
        if (!validateRecordedFileEvent(payload, descriptor.readId, descriptor.game)) {
            fanOut(recordedFileCallbacks, {
                type: 'error',
                readId: descriptor.readId,
                message: 'Invalid recorded-file event.',
            });
            closeRecordedFilePort(descriptor.readId);
            return;
        }
        if (payload.type === 'chunk') {
            const { chunkIndex, ...publicPayload } = payload;
            await fanOutAsync(recordedFileCallbacks, publicPayload);
            if (recordedFilePorts.get(descriptor.readId) === state && !state.terminal) {
                port.postMessage({ type: 'chunk-consumed', readId: descriptor.readId, chunkIndex });
            }
            return;
        }
        fanOut(recordedFileCallbacks, payload);
        if (payload.type === 'complete' || payload.type === 'error') closeRecordedFilePort(descriptor.readId);
    };
    port.onmessageerror = () => {
        fanOut(recordedFileCallbacks, { type: 'error', readId: descriptor.readId, message: 'Recorded-file port failed.' });
        closeRecordedFilePort(descriptor.readId);
    };
    port.onclose = () => {
        if (recordedFilePorts.get(descriptor.readId) === state && !state.terminal) {
            recordedFilePorts.delete(descriptor.readId);
            fanOut(recordedFileCallbacks, {
                type: 'error',
                readId: descriptor.readId,
                message: 'Recorded-file port closed unexpectedly.',
            });
        }
    };
    port.start();
    port.postMessage({ type: 'ready', readId: descriptor.readId });
});

const validateRecordingStartResult = (result) => {
    if (isRecordingStartFailure(result)) return result;
    if (result?.ok === true && Object.keys(result).length === 4 && isValidGame(result.game)
        && typeof result.filePath === 'string' && result.filePath.length > 0
        && Number.isFinite(result.startedAt)) return result;
    throw new Error('Main process returned an invalid recording start result.');
};

const cleanupPrivatePorts = () => {
    closeLivePort(null);
    for (const readId of Array.from(recordedFilePorts.keys())) closeRecordedFilePort(readId);
};
globalThis.addEventListener?.('unload', cleanupPrivatePorts, { once: true });

//contextBridge.exposeInMainWorld makes the function available in the global electronAPI object within the renderer.
contextBridge.exposeInMainWorld('electronAPI', {

    detectDesktopGame: () => ipcRenderer.invoke('detect-desktop-game'),

    startRecordingSession: async (config) => validateRecordingStartResult(
        await ipcRenderer.invoke('recording-session-start', config),
    ),
    stopRecordingSession: () => ipcRenderer.invoke('recording-session-stop'),
    onRecordingViewUpdate: (callback) => subscribe(recordingViewCallbacks, callback, 'onRecordingViewUpdate'),
    onRecordingSessionEnded: (callback) => subscribe(recordingEndedCallbacks, callback, 'onRecordingSessionEnded'),
    startRecordedFileRead: async (request) => {
        const result = await ipcRenderer.invoke('recorded-file-read-start', request);
        if (!result || typeof result.readId !== 'string' || !result.readId) {
            throw new Error('Main process returned an invalid recorded-file read id.');
        }
        return result;
    },
    cancelRecordedFileRead: async (readId) => {
        await ipcRenderer.invoke('recorded-file-read-cancel', readId);
        closeRecordedFilePort(readId);
    },
    onRecordedFileReadEvent: (callback) => subscribe(recordedFileCallbacks, callback, 'onRecordedFileReadEvent'),

    //run script in main process using async
    runPythonScript: (scriptPath, options) => ipcRenderer.invoke('run-python-script', scriptPath, options),
    stopPythonScript: (shellId) => ipcRenderer.invoke('stop-python-script', shellId),

    writeTempFile: (options) => ipcRenderer.invoke('write-temp-file', options),
    deleteTempFile: (filePath) => ipcRenderer.invoke('delete-temp-file', filePath),
    validateTelemetryFile: (filePath) => ipcRenderer.invoke('validate-telemetry-file', filePath),

    //
    onPythonEnd: (listenerIdOrCallback, maybeCallback) => {
        const listenerId = typeof listenerIdOrCallback === 'string' ? listenerIdOrCallback : undefined;
        const callback = typeof listenerIdOrCallback === 'function' ? listenerIdOrCallback : maybeCallback;

        if (typeof callback !== 'function') {
            throw new Error('onPythonEnd requires a callback function');
        }

        if (listenerId) {
            console.log("preload onPythonEnd called by", listenerId);
        } else {
            console.log("preload onPythonEnd called");
        }

        const subscription = (event, ...args) => {
            callback(...args, listenerId);
        };
        ipcRenderer.on('python-end', subscription);
        return () => {
            ipcRenderer.off('python-end', subscription);
        };
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

    // Speech Recognition API
    isSpeechRecognitionAvailable: () => ipcRenderer.invoke('check-speech-recognition-availability'),
    startSpeechRecognition: () => ipcRenderer.invoke('start-speech-recognition'),
    stopSpeechRecognition: () => ipcRenderer.invoke('stop-speech-recognition'),

    // Speech recognition event listeners
    onSpeechRecognitionStatus: (callback) => {
        const subscription = (event, ...args) => callback(...args);
        ipcRenderer.on('speech-recognition-status', subscription);
        return () => {
            ipcRenderer.off('speech-recognition-status', subscription);
        };
    },

    onSpeechRecognitionComplete: (callback) => {
        const subscription = (event, ...args) => callback(...args);
        ipcRenderer.on('speech-recognition-complete', subscription);
        return () => {
            ipcRenderer.off('speech-recognition-complete', subscription);
        };
    },

    // Session-scoped floating overlay and typed display broker.
    createOverlaySession: (descriptor) => ipcRenderer.invoke('overlay-session-create', descriptor),
    destroyOverlaySession: (presentationId) => ipcRenderer.invoke('overlay-session-destroy', presentationId),
    setOverlayEnabled: (enabled) => ipcRenderer.invoke('overlay-session-set-enabled', Boolean(enabled)),
    isOverlayEnabled: () => ipcRenderer.invoke('overlay-session-is-enabled'),
    sendOverlayPresentation: (presentation) => (
        ipcRenderer.invoke('overlay-presentation-submit', presentation)
    ),
    onOverlayRendererEvent: (callback) => {
        const subscription = (_event, rendererEvent) => callback(rendererEvent);
        ipcRenderer.on('overlay-renderer-event', subscription);
        return () => ipcRenderer.off('overlay-renderer-event', subscription);
    },
    onOverlayPresentation: (callback) => {
        const subscription = (_event, presentation) => callback(presentation);
        ipcRenderer.on('overlay-presentation-snapshot', subscription);
        return () => ipcRenderer.off('overlay-presentation-snapshot', subscription);
    },
    acknowledgeOverlayPresentation: (acknowledgement) => (
        ipcRenderer.send('overlay-presentation-acknowledgement', acknowledgement)
    ),
    emitOverlayRendererEvent: (event) => ipcRenderer.send('overlay-renderer-event', event),
    reportOverlayReady: () => ipcRenderer.send('overlay-renderer-ready'),
    resizeFloatingChat: (width, height) => ipcRenderer.invoke('resize-floating-chat', { width, height }),
    onFloatingChatClosed: (callback) => {
        const subscription = () => callback();
        ipcRenderer.on('floating-chat-closed', subscription);
        return () => {
            ipcRenderer.off('floating-chat-closed', subscription);
        };
    },
});
