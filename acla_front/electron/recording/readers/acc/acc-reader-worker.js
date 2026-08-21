'use strict';

const { AccPythonReader } = require('./acc-python-reader');
const { assertReader, validateReaderEvent } = require('../reader-contract');

const parentPort = process.parentPort;

function eventData(event) {
  return event && Object.prototype.hasOwnProperty.call(event, 'data') ? event.data : event;
}

function parentSend(message) {
  parentPort?.postMessage(message);
}

function startPort(port) {
  if (!port || typeof port.postMessage !== 'function'
    || typeof port.on !== 'function'
    || typeof port.start !== 'function') {
    throw new TypeError('ACC reader received an invalid MessagePort.');
  }
  return port;
}

function runAccReaderWorker() {
  if (!parentPort) return;
  let initialized = false;
  let stopping = false;
  let reader = null;
  let frameToWriter = null;
  let frameToView = null;
  let stopPromise = null;
  let portFailureSent = false;

  const stop = () => {
    if (stopPromise) return stopPromise;
    stopping = true;
    stopPromise = (async () => {
      await reader?.stop();
      for (const port of [frameToWriter, frameToView]) {
        try { port?.postMessage({ type: 'end', game: 'acc' }); } catch { /* peer failed */ }
      }
      parentSend({ type: 'stopped', service: 'reader', game: 'acc' });
      for (const port of [frameToWriter, frameToView]) {
        try { port?.close(); } catch { /* already closed */ }
      }
    })().catch((error) => {
      parentSend({ type: 'fatal', service: 'reader', error: error?.message || String(error) });
    });
    return stopPromise;
  };

  parentPort.on('message', (event) => {
    const message = eventData(event);
    if (message?.type === 'stop') {
      void stop();
      return;
    }
    if (message?.type !== 'initialize' || initialized) {
      parentSend({ type: 'fatal', service: 'reader', error: 'Invalid or duplicate reader initialization.' });
      return;
    }

    try {
      if (message.game !== 'acc'
        || message.readerOptions?.runtime !== 'python'
        || !Array.isArray(message.portRoles)
        || message.portRoles.length !== 2
        || message.portRoles[0] !== 'frameToWriter'
        || message.portRoles[1] !== 'frameToView'
        || !Array.isArray(event.ports)
        || event.ports.length !== 2
        || event.ports[0] === event.ports[1]) {
        throw new TypeError('ACC reader initialization descriptor or port order is invalid.');
      }
      frameToWriter = startPort(event.ports[0]);
      frameToView = startPort(event.ports[1]);
      reader = assertReader(new AccPythonReader(message.readerOptions));
      for (const port of [frameToWriter, frameToView]) {
        port.on('close', () => {
          if (!stopping && !portFailureSent) {
            portFailureSent = true;
            parentSend({ type: 'fatal', service: 'reader', error: 'ACC recording data port closed unexpectedly.' });
          }
        });
        port.start();
      }
      initialized = true;

      const emit = (readerEvent) => {
        if (stopping) return;
        const validation = validateReaderEvent(readerEvent, 'acc');
        if (!validation.ok) {
          parentSend({ type: 'fatal', service: 'reader', error: validation.error });
          return;
        }
        if (readerEvent.type === 'fatal') {
          parentSend({ type: 'fatal', service: 'reader', error: readerEvent.error });
          return;
        }
        const portMessage = { type: 'frame', frame: readerEvent.frame };
        frameToWriter.postMessage(portMessage);
        frameToView.postMessage(portMessage);
      };

      void reader.start(emit).then(() => {
        if (!stopping) parentSend({ type: 'ready', service: 'reader', game: 'acc' });
      }).catch((error) => {
        if (!stopping) parentSend({ type: 'fatal', service: 'reader', error: error?.message || String(error) });
      });
    } catch (error) {
      parentSend({ type: 'fatal', service: 'reader', error: error?.message || String(error) });
      void stop();
    }
  });

  parentPort.start?.();
}

runAccReaderWorker();

module.exports = { runAccReaderWorker };
