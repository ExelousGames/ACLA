'use strict';

const { IRacingReader } = require('./iracing-reader');
const { validateReaderEvent } = require('../reader-contract');

const MAX_IN_FLIGHT = 120;
const data = (event) => event && Object.prototype.hasOwnProperty.call(event, 'data') ? event.data : event;

// Credits bound both MessagePort queues. Writer credits arrive only after disk commit.
class DeliveryWindow {
  constructor(limit = MAX_IN_FLIGHT, timeoutMs = 5000) {
    this.limit = limit;
    this.timeoutMs = timeoutMs;
    this.sent = 0;
    this.acknowledged = { writer: 0, view: 0 };
    this.waiter = null;
  }

  sentFrame() {
    this.sent += 1;
    if (this.sent - Math.min(...Object.values(this.acknowledged)) < this.limit) return undefined;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { this.waiter = null; reject(new Error('iRacing recording consumers are not keeping up.')); }, this.timeoutMs);
      this.waiter = { resolve, timer };
    });
  }

  acknowledge(peer, sequence) {
    if (!Object.prototype.hasOwnProperty.call(this.acknowledged, peer)
      || !Number.isSafeInteger(sequence) || sequence < this.acknowledged[peer] || sequence > this.sent) {
      throw new Error('Invalid iRacing delivery acknowledgement.');
    }
    this.acknowledged[peer] = sequence;
    if (this.waiter && this.sent - Math.min(...Object.values(this.acknowledged)) < this.limit) this.close();
  }

  close() {
    if (this.waiter) { clearTimeout(this.waiter.timer); this.waiter.resolve(); this.waiter = null; }
  }
}

function runIRacingReaderWorker(parentPort = process.parentPort) {
  if (!parentPort) return;
  let reader;
  let ports = [];
  let initialized = false;
  let stopping = false;
  let stopPromise;
  const window = new DeliveryWindow();
  const send = (message) => parentPort.postMessage({ service: 'reader', game: 'iracing', ...message });
  const stop = () => {
    if (stopPromise) return stopPromise;
    stopping = true;
    window.close();
    stopPromise = (async () => {
      await reader?.stop();
      for (const port of ports) { try { port.postMessage({ type: 'end', game: 'iracing' }); } catch { /* failed peer */ } }
      send({ type: 'stopped' });
      for (const port of ports) { try { port.close(); } catch { /* already closed */ } }
    })();
    return stopPromise;
  };
  const fatal = (error) => {
    if (stopping) return;
    send({ type: 'fatal', error: error?.message || String(error) });
    void stop().catch((failure) => send({ type: 'fatal', error: failure.message }));
  };
  parentPort.on('message', (event) => {
    const message = data(event);
    if (message?.type === 'stop') { void stop(); return; }
    try {
      if (stopping || initialized || message?.type !== 'initialize' || message.game !== 'iracing'
        || JSON.stringify(message.portRoles) !== JSON.stringify(['frameToWriter', 'frameToView'])
        || event.ports?.length !== 2 || event.ports[0] === event.ports[1]) {
        throw new Error('Invalid iRacing reader initialization.');
      }
      ports = event.ports;
      ports.forEach((port, index) => {
        if (typeof port?.postMessage !== 'function' || typeof port?.on !== 'function' || typeof port?.start !== 'function') {
          throw new Error('Invalid iRacing recording data port.');
        }
        const peer = index === 0 ? 'writer' : 'view';
        port.on('message', (event) => {
          if (stopping) return;
          try {
            const ack = data(event);
            if (ack?.type !== 'ack' || ack.game !== 'iracing') throw new Error('Invalid iRacing delivery message.');
            window.acknowledge(peer, ack.sequence);
          } catch (error) { fatal(error); }
        });
        port.on('close', () => { if (!stopping) fatal(new Error('iRacing recording data port closed unexpectedly.')); });
        port.start();
      });
      initialized = true;
      reader = new IRacingReader(message.readerOptions);
      void reader.start((event) => {
        if (stopping) return undefined;
        const validation = validateReaderEvent(event, 'iracing');
        if (!validation.ok) throw new Error(validation.error);
        if (event.type === 'fatal') { fatal(new Error(event.error)); return undefined; }
        const wait = window.sentFrame();
        for (const port of ports) port.postMessage({ type: 'frame', frame: event.frame });
        return wait;
      }).then(() => { if (!stopping) send({ type: 'ready' }); }).catch(fatal);
    } catch (error) { fatal(error); }
  });
  parentPort.start?.();
}

runIRacingReaderWorker();

module.exports = { DeliveryWindow, MAX_IN_FLIGHT, runIRacingReaderWorker };
