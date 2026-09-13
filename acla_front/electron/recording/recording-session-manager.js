'use strict';

const path = require('path');
const {
  DESKTOP_GAME_SET,
  isReaderLaunchConfigResult,
  validateRecordingStartConfig,
} = require('./recording-protocol');

const STARTUP_TIMEOUT_MS = 10000;
const SHUTDOWN_TIMEOUT_MS = 5000;
const WORKER_CONSOLE_LABELS = Object.freeze({
  reader: 'collect worker',
  writer: 'write worker',
  view: 'view worker',
});

function workerLifecycleLogMessage(service, state) {
  const label = WORKER_CONSOLE_LABELS[service];
  if (!label) throw new TypeError(`Unknown recording worker service: ${String(service)}.`);
  if (state !== 'started' && state !== 'ended') {
    throw new TypeError(`Unknown recording worker lifecycle state: ${String(state)}.`);
  }
  return `[recording] ${label} ${state}.`;
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject, settled: false };
}

function settleResolve(target, value) {
  if (target.settled) return;
  target.settled = true;
  target.resolve(value);
}

function settleReject(target, error) {
  if (target.settled) return;
  target.settled = true;
  target.reject(error instanceof Error ? error : new Error(String(error)));
}

function withTimeout(promise, timeoutMs, message) {
  let timer;
  return Promise.race([
    promise,
    new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(message)), timeoutMs);
    }),
  ]).finally(() => clearTimeout(timer));
}

function utilityMessageData(message) {
  return message && Object.prototype.hasOwnProperty.call(message, 'data') ? message.data : message;
}

function safePostMessage(processHandle, message, ports) {
  if (!processHandle || typeof processHandle.postMessage !== 'function') {
    throw new TypeError('Recording utility process is unavailable.');
  }
  if (ports) processHandle.postMessage(message, ports);
  else processHandle.postMessage(message);
}

function safeKill(processHandle) {
  if (!processHandle || typeof processHandle.kill !== 'function') return;
  try { processHandle.kill(); } catch { /* already exited */ }
}

function pathsEqual(left, right) {
  const normalizedLeft = path.resolve(left);
  const normalizedRight = path.resolve(right);
  return process.platform === 'win32'
    ? normalizedLeft.toLowerCase() === normalizedRight.toLowerCase()
    : normalizedLeft === normalizedRight;
}

class RecordingSessionManager {
  constructor({ utilityProcess, MessageChannelMain, getMainWindow, getReaderLaunchConfig, recordingDirectory } = {}) {
    if (!utilityProcess || typeof utilityProcess.fork !== 'function') {
      throw new TypeError('utilityProcess.fork is required.');
    }
    if (typeof MessageChannelMain !== 'function') {
      throw new TypeError('MessageChannelMain constructor is required.');
    }
    if (typeof getMainWindow !== 'function') throw new TypeError('getMainWindow is required.');
    if (typeof getReaderLaunchConfig !== 'function') throw new TypeError('getReaderLaunchConfig is required.');
    if (typeof recordingDirectory !== 'string' || !recordingDirectory || !path.isAbsolute(recordingDirectory)) {
      throw new TypeError('recordingDirectory must be a non-empty absolute path.');
    }

    this.utilityProcess = utilityProcess;
    this.MessageChannelMain = MessageChannelMain;
    this.getMainWindow = getMainWindow;
    this.getReaderLaunchConfig = getReaderLaunchConfig;
    this.recordingDirectory = path.resolve(recordingDirectory);
    this.active = null;
  }

  hasActiveSession() {
    return Boolean(this.active && this.active.status !== 'terminated');
  }

  isOwnedBy(ownerWebContentsId) {
    return this.hasActiveSession() && this.active.ownerWebContentsId === ownerWebContentsId;
  }

  getActiveRecordedFileReadLimit({ ownerWebContentsId, game, filePath } = {}) {
    const managed = this.active;
    if (!managed || managed.status === 'terminated') return null;
    if (managed.ownerWebContentsId !== ownerWebContentsId) {
      throw new Error('Active recording file belongs to another renderer.');
    }
    if (managed.game !== game) {
      throw new Error('Active recording file game does not match the read request.');
    }
    if (typeof filePath !== 'string' || !path.isAbsolute(filePath)
      || !managed.activeFilePath || !pathsEqual(managed.activeFilePath, filePath)) {
      throw new Error('Only the active writer file can be read while recording.');
    }
    return managed.committedRowCount;
  }

  async startSession({ game, ownerWebContentsId } = {}) {
    const requestFailure = validateRecordingStartConfig({ game });
    if (requestFailure) return requestFailure;
    if (!Number.isSafeInteger(ownerWebContentsId) || ownerWebContentsId < 0) {
      throw new TypeError('Recording owner webContents id is invalid.');
    }
    if (this.hasActiveSession()) throw new Error('A recording session is already active.');

    const launchResult = await this.getReaderLaunchConfig(game);
    if (!isReaderLaunchConfigResult(launchResult)) {
      throw new TypeError('Reader launch resolver returned an invalid result.');
    }
    if (!launchResult.ok) return launchResult;
    const { config } = launchResult;
    if (config.game !== game || !path.isAbsolute(config.readerEntryPath)) {
      throw new TypeError('Reader launch config does not match the requested game.');
    }

    const mainWindow = this.getMainWindow();
    if (!mainWindow || mainWindow.isDestroyed?.()
      || mainWindow.webContents?.id !== ownerWebContentsId
      || typeof mainWindow.webContents?.postMessage !== 'function') {
      throw new Error('The recording owner is no longer available.');
    }

    const writerEntryPath = path.join(__dirname, 'workers', 'writer-worker.js');
    const viewEntryPath = path.join(__dirname, 'workers', 'view-worker.js');
    const startedProcesses = [];
    const startWorker = (service, entryPath, serviceName) => {
      const processHandle = this.utilityProcess.fork(entryPath, [], {
        serviceName,
        stdio: 'pipe',
      });
      startedProcesses.push({ processHandle, service });
      console.log(workerLifecycleLogMessage(service, 'started'));
      return processHandle;
    };
    let writer;
    let view;
    let reader;
    try {
      writer = startWorker('writer', writerEntryPath, 'ACLA Recording Writer');
      view = startWorker('view', viewEntryPath, 'ACLA Recording View');
      reader = startWorker('reader', config.readerEntryPath, `ACLA ${game.toUpperCase()} Telemetry Reader`);
    } catch (error) {
      for (const { processHandle, service } of startedProcesses) {
        safeKill(processHandle);
        console.log(workerLifecycleLogMessage(service, 'ended'));
      }
      throw error;
    }

    const signals = {
      readerReady: deferred(),
      writerReady: deferred(),
      viewReady: deferred(),
      readerStopped: deferred(),
      writerFinalized: deferred(),
      viewStopped: deferred(),
      failure: deferred(),
    };
    // Avoid an unhandled rejection when no startup waiter has attached yet.
    signals.failure.promise.catch(() => undefined);
    const listeners = [];
    const managed = {
      game,
      status: 'starting',
      ownerWebContentsId,
      reader,
      writer,
      view,
      readyWorkers: new Set(),
      stopPromise: null,
      signals,
      listeners,
      endedWorkers: new Set(),
      activeFilePath: null,
      committedRowCount: 0,
    };
    this.active = managed;

    const fail = (error) => {
      const failure = error instanceof Error ? error : new Error(String(error));
      settleReject(signals.failure, failure);
      if (this.active === managed && managed.status === 'running') {
        void this._stopManaged(managed, failure);
      }
    };
    const attach = (processHandle, service) => {
      const onMessage = (rawMessage) => {
        const message = utilityMessageData(rawMessage);
        if (!message || typeof message !== 'object') return fail(new Error(`${service} sent an invalid control message.`));
        if (message.type === 'fatal') return fail(new Error(message.error || `${service} failed.`));
        if (message.service && message.service !== service) return fail(new Error(`${service} sent a mismatched service name.`));
        if (message.type === 'ready') {
          if (message.game !== game) {
            fail(new Error(`${service} sent readiness for another game.`));
          } else if (service === 'writer' && typeof message.filePath === 'string' && path.isAbsolute(message.filePath)) {
            managed.activeFilePath = path.resolve(message.filePath);
            managed.readyWorkers.add('writer');
            settleResolve(signals.writerReady, message);
          } else if (service === 'view') {
            managed.readyWorkers.add('view');
            settleResolve(signals.viewReady, message);
          } else if (service === 'reader') {
            managed.readyWorkers.add('reader');
            settleResolve(signals.readerReady, message);
          } else fail(new Error(`${service} sent an invalid ready message.`));
        } else if (message.type === 'committed' && service === 'writer') {
          if (message.game !== game
            || !Number.isSafeInteger(message.fromSequence)
            || !Number.isSafeInteger(message.toSequence)
            || !Number.isSafeInteger(message.committedCount)
            || message.fromSequence !== managed.committedRowCount + 1
            || message.toSequence < message.fromSequence
            || message.committedCount !== message.toSequence) {
            fail(new Error('Writer sent a non-contiguous commit range.'));
          } else {
            managed.committedRowCount = message.committedCount;
          }
        } else if (message.type === 'stopped') {
          if (message.game && message.game !== game) return fail(new Error(`${service} stopped for another game.`));
          if (managed.status !== 'stopping') return fail(new Error(`${service} stopped unexpectedly.`));
          if (service === 'reader') settleResolve(signals.readerStopped, message);
          else if (service === 'view') settleResolve(signals.viewStopped, message);
          else if (service === 'writer') settleResolve(signals.writerFinalized, message);
          else fail(new Error(`${service} sent an unexpected stopped message.`));
        } else if (message.type === 'finalized' && service === 'writer'
          && message.game === game
          && typeof message.filePath === 'string'
          && path.isAbsolute(message.filePath)
          && managed.activeFilePath
          && pathsEqual(message.filePath, managed.activeFilePath)
          && Number.isSafeInteger(message.writtenSamples)
          && message.writtenSamples === managed.committedRowCount) {
          settleResolve(signals.writerFinalized, message);
        } else fail(new Error(`${service} sent an unknown or invalid control message.`));
      };
      const onExit = (code) => {
        this._logWorkerEnded(managed, service);
        if (managed.status !== 'stopping' && managed.status !== 'terminated') {
          fail(new Error(`${service} utility exited unexpectedly (code ${String(code)}).`));
        }
      };
      const onError = (error) => fail(error);
      processHandle.on?.('message', onMessage);
      processHandle.on?.('exit', onExit);
      processHandle.on?.('error', onError);
      listeners.push({ processHandle, onMessage, onExit, onError });
    };
    attach(writer, 'writer');
    attach(view, 'view');
    attach(reader, 'reader');

    const untransferred = new Set();
    const createChannel = () => {
      const channel = new this.MessageChannelMain();
      if (!channel?.port1 || !channel?.port2) throw new TypeError('MessageChannelMain did not create two endpoints.');
      untransferred.add(channel.port1);
      untransferred.add(channel.port2);
      return channel;
    };

    try {
      const readerToWriter = createChannel();
      const readerToView = createChannel();
      const writerToView = createChannel();
      const viewToPreload = createChannel();

      safePostMessage(writer, {
        type: 'initialize',
        game,
        recordingDirectory: this.recordingDirectory,
        portRoles: ['frameFromReader', 'progressToView'],
      }, [readerToWriter.port2, writerToView.port1]);
      untransferred.delete(readerToWriter.port2);
      untransferred.delete(writerToView.port1);

      safePostMessage(view, {
        type: 'initialize',
        game,
        portRoles: ['frameFromReader', 'progressFromWriter', 'updatesToPreload'],
      }, [readerToView.port2, writerToView.port2, viewToPreload.port1]);
      untransferred.delete(readerToView.port2);
      untransferred.delete(writerToView.port2);
      untransferred.delete(viewToPreload.port1);

      mainWindow.webContents.postMessage('recording-view-port', { game }, [viewToPreload.port2]);
      untransferred.delete(viewToPreload.port2);

      const waitStartup = (promise, label) => withTimeout(
        Promise.race([promise, signals.failure.promise]),
        STARTUP_TIMEOUT_MS,
        `${label} did not become ready within ${STARTUP_TIMEOUT_MS}ms.`,
      );
      const [writerReady] = await Promise.all([
        waitStartup(signals.writerReady.promise, 'Recording writer'),
        waitStartup(signals.viewReady.promise, 'Recording preload'),
      ]);

      safePostMessage(reader, {
        type: 'initialize',
        game,
        readerOptions: config.readerOptions,
        portRoles: ['frameToWriter', 'frameToView'],
      }, [readerToWriter.port1, readerToView.port1]);
      untransferred.delete(readerToWriter.port1);
      untransferred.delete(readerToView.port1);

      await waitStartup(signals.readerReady.promise, `${game.toUpperCase()} telemetry reader`);
      managed.status = 'running';
      safePostMessage(writer, { type: 'published' });
      return {
        ok: true,
        game,
        filePath: writerReady.filePath,
        startedAt: Date.now(),
      };
    } catch (error) {
      for (const port of untransferred) {
        try { port.close(); } catch { /* already closed */ }
      }
      await this._rollbackStartup(managed);
      throw error;
    }
  }

  async stopSession(ownerWebContentsId) {
    const managed = this.active;
    if (!managed || managed.status === 'terminated') throw new Error('No recording session is active.');
    if (managed.ownerWebContentsId !== ownerWebContentsId) throw new Error('Recording session belongs to another renderer.');
    return this._stopManaged(managed, null);
  }

  async _stopManaged(managed, failure) {
    if (managed.stopPromise) return managed.stopPromise;
    managed.status = 'stopping';
    managed.stopPromise = (async () => {
      try {
        safePostMessage(managed.reader, { type: 'stop' });
        if (failure) {
          safePostMessage(managed.view, { type: 'stop', error: failure.message });
        }
        const results = await withTimeout(Promise.all([
          managed.signals.readerStopped.promise,
          managed.signals.writerFinalized.promise,
          managed.signals.viewStopped.promise,
        ]), SHUTDOWN_TIMEOUT_MS, 'Recording pipeline did not stop within five seconds.');
        const writerResult = results[1];
        const result = {
          game: managed.game,
          filePath: writerResult.filePath,
          writtenSamples: writerResult.writtenSamples,
          ...(failure ? { error: failure.message } : {}),
        };
        this._emitEnded(managed, result);
        return result;
      } catch (error) {
        const terminalError = failure || error;
        safePostMessage(managed.writer, { type: 'stop' });
        safePostMessage(managed.view, { type: 'stop', error: terminalError?.message || String(terminalError) });
        const result = { game: managed.game, error: terminalError?.message || String(terminalError) };
        this._emitEnded(managed, result);
        if (!failure) throw error;
        return result;
      } finally {
        this._terminateManaged(managed);
      }
    })();
    return managed.stopPromise;
  }

  async _rollbackStartup(managed) {
    managed.status = 'stopping';
    try { safePostMessage(managed.reader, { type: 'stop' }); } catch { /* partial startup */ }
    try { safePostMessage(managed.writer, { type: 'rollback' }); } catch { /* partial startup */ }
    try { safePostMessage(managed.view, { type: 'stop', error: 'Recording startup failed.' }); } catch { /* partial startup */ }
    await Promise.race([
      managed.signals.writerFinalized.promise.catch(() => undefined),
      new Promise((resolve) => setTimeout(resolve, 750)),
    ]);
    this._terminateManaged(managed);
  }

  _emitEnded(managed, result) {
    const window = this.getMainWindow();
    if (window && !window.isDestroyed?.() && window.webContents?.id === managed.ownerWebContentsId) {
      window.webContents.send?.('recording-session-ended', result);
    }
  }

  _terminateManaged(managed) {
    if (managed.status === 'terminated') return;
    managed.status = 'terminated';
    for (const listener of managed.listeners) {
      listener.processHandle.off?.('message', listener.onMessage);
      listener.processHandle.off?.('exit', listener.onExit);
      listener.processHandle.off?.('error', listener.onError);
    }
    for (const [service, processHandle] of [
      ['reader', managed.reader],
      ['writer', managed.writer],
      ['view', managed.view],
    ]) {
      safeKill(processHandle);
      this._logWorkerEnded(managed, service);
    }
    if (this.active === managed) this.active = null;
  }

  _logWorkerEnded(managed, service) {
    if (managed.endedWorkers.has(service)) return;
    managed.endedWorkers.add(service);
    console.log(workerLifecycleLogMessage(service, 'ended'));
  }

  async shutdownAll() {
    if (!this.active) return;
    try {
      await this._stopManaged(this.active, new Error('Application is shutting down.'));
    } catch {
      if (this.active) this._terminateManaged(this.active);
    }
  }
}

module.exports = {
  RecordingSessionManager,
  SHUTDOWN_TIMEOUT_MS,
  STARTUP_TIMEOUT_MS,
  deferred,
  workerLifecycleLogMessage,
  pathsEqual,
  withTimeout,
};
