'use strict';

const { DESKTOP_GAME_SET } = require('../recording-protocol');
const { validateSourceFrame } = require('../telemetry-contract');

function eventData(event) {
  return event && Object.prototype.hasOwnProperty.call(event, 'data') ? event.data : event;
}

class RecordingView {
  constructor({ game, updatesPort, parentSend }) {
    if (!DESKTOP_GAME_SET.has(game)) throw new TypeError('View game is invalid.');
    this.game = game;
    this.updatesPort = updatesPort;
    this.parentSend = parentSend;
    this.receivedSequence = 0;
    this.committedSequence = 0;
    this.committedCount = 0;
    this.readerEnded = false;
    this.writerFinal = null;
    this.terminalSent = false;
  }

  acceptFrame(frame) {
    if (this.terminalSent) return;
    const validation = validateSourceFrame(frame, this.game);
    if (!validation.ok) return this.fail(validation.error);
    this.receivedSequence += 1;
    this.updatesPort.postMessage({
      type: 'frame',
      game: this.game,
      sample: frame.sample,
      sequence: this.receivedSequence,
      committedSequence: this.committedSequence,
      committedCount: this.committedCount,
    });
  }

  acceptProgress(message) {
    if (!message || message.game !== this.game) return this.fail('View received progress for another game.');
    if (message.type === 'committed') {
      if (!Number.isSafeInteger(message.fromSequence)
        || !Number.isSafeInteger(message.toSequence)
        || message.fromSequence !== this.committedSequence + 1
        || message.toSequence < message.fromSequence
        || !Number.isSafeInteger(message.committedCount)
        || message.committedCount !== message.toSequence) {
        return this.fail('View received a non-contiguous writer commit range.');
      }
      this.committedSequence = message.toSequence;
      this.committedCount = message.committedCount;
      return;
    }
    if (message.type === 'final') {
      if (!Number.isSafeInteger(message.committedCount)
        || message.committedCount !== this.committedCount
        || typeof message.filePath !== 'string'
        || !Number.isSafeInteger(message.writtenSamples)
        || message.writtenSamples !== this.committedCount) {
        return this.fail('View received an invalid writer final summary.');
      }
      this.writerFinal = message;
      this.maybeFinish();
      return;
    }
    if (message.type === 'fatal' && typeof message.error === 'string') {
      this.fail(message.error);
      return;
    }
    this.fail('View received an unknown writer progress message.');
  }

  markReaderEnded() {
    this.readerEnded = true;
    this.maybeFinish();
  }

  maybeFinish() {
    if (!this.readerEnded || !this.writerFinal || this.terminalSent) return;
    this.terminalSent = true;
    this.updatesPort.postMessage({
      type: 'terminal',
      game: this.game,
      filePath: this.writerFinal.filePath,
      writtenSamples: this.writerFinal.writtenSamples,
    });
    this.parentSend({ type: 'stopped', service: 'view', game: this.game });
    setImmediate(() => {
      try { this.updatesPort.close(); } catch { /* already closed */ }
    });
  }

  fail(error) {
    if (this.terminalSent) return;
    this.terminalSent = true;
    const message = error instanceof Error ? error.message : String(error);
    try { this.updatesPort.postMessage({ type: 'terminal', game: this.game, error: message }); } catch { /* peer failed */ }
    this.parentSend({ type: 'fatal', service: 'view', error: message });
  }
}

function runViewWorker() {
  const parentPort = process.parentPort;
  if (!parentPort) return;
  let view = null;
  let initialized = false;
  let readerPort = null;
  let progressPort = null;
  let updatesPort = null;
  let preloadReady = false;
  const parentSend = (message) => parentPort.postMessage(message);

  parentPort.on('message', (event) => {
    const message = eventData(event);
    if (message?.type === 'stop') {
      if (view && !view.terminalSent) view.fail(message.error || 'Recording view stopped.');
      parentSend({ type: 'stopped', service: 'view', game: view?.game });
      return;
    }
    if (message?.type !== 'initialize' || initialized) {
      parentSend({ type: 'fatal', service: 'view', error: 'Invalid or duplicate view initialization.' });
      return;
    }
    try {
      if (!DESKTOP_GAME_SET.has(message.game)
        || !Array.isArray(message.portRoles)
        || message.portRoles.join('|') !== 'frameFromReader|progressFromWriter|updatesToPreload'
        || !Array.isArray(event.ports)
        || event.ports.length !== 3
        || new Set(event.ports).size !== 3) {
        throw new TypeError('View initialization descriptor or port order is invalid.');
      }
      [readerPort, progressPort, updatesPort] = event.ports;
      if ([readerPort, progressPort].some((port) => typeof port?.on !== 'function' || typeof port?.start !== 'function')
        || typeof updatesPort?.postMessage !== 'function'
        || typeof updatesPort?.on !== 'function'
        || typeof updatesPort?.start !== 'function') {
        throw new TypeError('View initialization ports are invalid.');
      }
      view = new RecordingView({ game: message.game, updatesPort, parentSend });
      readerPort.on('message', (portEvent) => {
        const portMessage = eventData(portEvent);
        if (portMessage?.type === 'frame') view.acceptFrame(portMessage.frame);
        else if (portMessage?.type === 'end' && portMessage.game === message.game) view.markReaderEnded();
        else view.fail('View received an invalid reader message.');
      });
      readerPort.on('close', () => {
        if (!view.terminalSent && !view.readerEnded) view.fail('Reader/view port closed before end-of-stream.');
      });
      progressPort.on('message', (portEvent) => view.acceptProgress(eventData(portEvent)));
      progressPort.on('close', () => {
        if (!view.terminalSent && !view.writerFinal) view.fail('Writer/view progress port closed before finalization.');
      });
      updatesPort.on('message', (portEvent) => {
        const portMessage = eventData(portEvent);
        if (portMessage?.type === 'ready' && portMessage.game === message.game && !preloadReady) {
          preloadReady = true;
          parentSend({ type: 'ready', service: 'view', game: message.game });
        } else view.fail('View received an invalid preload acknowledgement.');
      });
      updatesPort.on('close', () => {
        if (!view.terminalSent) view.fail('View/preload port closed unexpectedly.');
      });
      for (const port of [readerPort, progressPort, updatesPort]) port.start();
      initialized = true;
    } catch (error) {
      parentSend({ type: 'fatal', service: 'view', error: error?.message || String(error) });
    }
  });
  parentPort.start?.();
}

runViewWorker();

module.exports = { RecordingView, runViewWorker };
