'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { DESKTOP_GAME_SET } = require('../recording-protocol');
const { validateSourceFrame } = require('../telemetry-contract');

const BATCH_INTERVAL_MS = 100;
const MAX_BATCH_SAMPLES = 30;

function eventData(event) {
  return event && Object.prototype.hasOwnProperty.call(event, 'data') ? event.data : event;
}

function isContainedPath(parent, candidate) {
  const relative = path.relative(path.resolve(parent), path.resolve(candidate));
  return relative !== '' && !relative.startsWith('..') && !path.isAbsolute(relative);
}

function streamWrite(stream, content) {
  return new Promise((resolve, reject) => {
    stream.write(content, 'utf8', (error) => (error ? reject(error) : resolve()));
  });
}

function streamEnd(stream) {
  return new Promise((resolve, reject) => {
    const onError = (error) => {
      stream.off('error', onError);
      reject(error);
    };
    stream.once('error', onError);
    stream.end(() => {
      stream.off('error', onError);
      resolve();
    });
  });
}

class RecordingWriter {
  constructor({ game, recordingDirectory, progressPort, parentSend, fsModule = fs }) {
    if (!DESKTOP_GAME_SET.has(game)) throw new TypeError('Writer game is invalid.');
    if (typeof recordingDirectory !== 'string' || !path.isAbsolute(recordingDirectory)) {
      throw new TypeError('Writer recording directory must be absolute.');
    }
    if (!progressPort || typeof progressPort.postMessage !== 'function') {
      throw new TypeError('Writer progress port is invalid.');
    }
    this.game = game;
    this.recordingDirectory = path.resolve(recordingDirectory);
    this.progressPort = progressPort;
    this.parentSend = parentSend;
    this.fs = fsModule;
    this.stream = null;
    this.filePath = null;
    this.pathPublished = false;
    this.sequence = 0;
    this.committedCount = 0;
    this.batch = [];
    this.batchTimer = null;
    this.writeChain = Promise.resolve();
    this.ending = false;
    this.finalized = false;
    this.failed = false;
    this.rolledBack = false;
  }

  async open() {
    await this.fs.promises.mkdir(this.recordingDirectory, { recursive: true });
    for (let attempt = 0; attempt < 20; attempt += 1) {
      if (this.rolledBack) throw new Error('Writer startup was rolled back.');
      const fileName = `${this.game}_${Date.now()}_${crypto.randomBytes(6).toString('hex')}.jsonl`;
      const candidate = path.join(this.recordingDirectory, fileName);
      if (!isContainedPath(this.recordingDirectory, candidate)) {
        throw new Error('Generated recording path escaped the recording directory.');
      }
      try {
        this.filePath = candidate;
        this.stream = this.fs.createWriteStream(candidate, { flags: 'wx', encoding: 'utf8' });
        await new Promise((resolve, reject) => {
          const onOpen = () => {
            this.stream.off('error', onError);
            resolve();
          };
          const onError = (error) => {
            this.stream.off('open', onOpen);
            reject(error);
          };
          this.stream.once('open', onOpen);
          this.stream.once('error', onError);
        });
        this.stream.on('error', (error) => this.fail(error));
        if (this.rolledBack) {
          await this.rollback();
          throw new Error('Writer startup was rolled back.');
        }
        return this.filePath;
      } catch (error) {
        this.stream = null;
        this.filePath = null;
        if (error?.code !== 'EEXIST') throw error;
      }
    }
    throw new Error('Unable to reserve a unique recording file.');
  }

  markPublished() {
    this.pathPublished = true;
  }

  acceptFrame(frame) {
    if (this.ending || this.finalized || this.failed) return;
    const validation = validateSourceFrame(frame, this.game);
    if (!validation.ok) {
      this.fail(new Error(validation.error));
      return;
    }
    const sequence = ++this.sequence;
    this.batch.push({ sequence, sample: frame.sample });
    if (this.batch.length >= MAX_BATCH_SAMPLES) {
      this.flush();
    } else if (this.batchTimer === null) {
      this.batchTimer = setTimeout(() => this.flush(), BATCH_INTERVAL_MS);
    }
  }

  flush() {
    if (this.batchTimer !== null) {
      clearTimeout(this.batchTimer);
      this.batchTimer = null;
    }
    if (this.batch.length === 0 || this.failed) return this.writeChain;
    const batch = this.batch;
    this.batch = [];
    this.writeChain = this.writeChain.then(async () => {
      const lines = `${batch.map(({ sample }) => JSON.stringify(sample)).join('\n')}\n`;
      await streamWrite(this.stream, lines);
      const fromSequence = batch[0].sequence;
      const toSequence = batch[batch.length - 1].sequence;
      this.committedCount += batch.length;
      this.progressPort.postMessage({
        type: 'committed',
        game: this.game,
        fromSequence,
        toSequence,
        committedCount: this.committedCount,
      });
      this.parentSend({
        type: 'committed',
        service: 'writer',
        game: this.game,
        fromSequence,
        toSequence,
        committedCount: this.committedCount,
      });
    }).catch((error) => this.fail(error));
    return this.writeChain;
  }

  async end() {
    if (this.finalized) {
      return { filePath: this.filePath, writtenSamples: this.committedCount };
    }
    if (this.ending) return this.finalizePromise;
    this.ending = true;
    this.finalizePromise = (async () => {
      await this.flush();
      await this.writeChain;
      if (this.failed) throw new Error('Writer failed before finalization.');
      await streamEnd(this.stream);
      this.finalized = true;
      const result = { filePath: this.filePath, writtenSamples: this.committedCount };
      this.progressPort.postMessage({
        type: 'final',
        game: this.game,
        committedCount: this.committedCount,
        ...result,
      });
      this.parentSend({ type: 'finalized', service: 'writer', game: this.game, ...result });
      return result;
    })();
    return this.finalizePromise;
  }

  fail(error) {
    if (this.failed || this.finalized) return;
    this.failed = true;
    if (this.batchTimer !== null) clearTimeout(this.batchTimer);
    this.batchTimer = null;
    const message = error?.message || String(error);
    try { this.progressPort.postMessage({ type: 'fatal', game: this.game, error: message }); } catch { /* peer failed */ }
    this.parentSend({ type: 'fatal', service: 'writer', error: message });
  }

  async rollback() {
    this.rolledBack = true;
    if (this.batchTimer !== null) clearTimeout(this.batchTimer);
    this.batchTimer = null;
    if (this.stream && !this.stream.closed) {
      const stream = this.stream;
      await new Promise((resolve) => {
        stream.once('close', resolve);
        if (!stream.destroyed) stream.destroy();
      });
    }
    if (!this.pathPublished && this.filePath) {
      try { await this.fs.promises.unlink(this.filePath); } catch (error) {
        if (error?.code !== 'ENOENT') throw error;
      }
    }
  }
}

function runWriterWorker() {
  const parentPort = process.parentPort;
  if (!parentPort) return;
  let writer = null;
  let initialized = false;
  let framePort = null;
  let progressPort = null;

  const parentSend = (message) => parentPort.postMessage(message);
  parentPort.on('message', (event) => {
    const message = eventData(event);
    if (message?.type === 'published') {
      writer?.markPublished();
      return;
    }
    if (message?.type === 'rollback' || message?.type === 'stop') {
      void writer?.rollback().finally(() => parentSend({ type: 'stopped', service: 'writer' }));
      return;
    }
    if (message?.type !== 'initialize' || initialized) {
      parentSend({ type: 'fatal', service: 'writer', error: 'Invalid or duplicate writer initialization.' });
      return;
    }

    try {
      if (!DESKTOP_GAME_SET.has(message.game)
        || typeof message.recordingDirectory !== 'string'
        || !path.isAbsolute(message.recordingDirectory)
        || !Array.isArray(message.portRoles)
        || message.portRoles.length !== 2
        || message.portRoles[0] !== 'frameFromReader'
        || message.portRoles[1] !== 'progressToView'
        || !Array.isArray(event.ports)
        || event.ports.length !== 2
        || event.ports[0] === event.ports[1]) {
        throw new TypeError('Writer initialization descriptor or port order is invalid.');
      }
      framePort = event.ports[0];
      progressPort = event.ports[1];
      if (typeof framePort?.on !== 'function' || typeof framePort?.start !== 'function'
        || typeof progressPort?.postMessage !== 'function'
        || typeof progressPort?.on !== 'function'
        || typeof progressPort?.start !== 'function') {
        throw new TypeError('Writer initialization ports are invalid.');
      }
      writer = new RecordingWriter({
        game: message.game,
        recordingDirectory: message.recordingDirectory,
        progressPort,
        parentSend,
      });
      framePort.on('message', (portEvent) => {
        const portMessage = eventData(portEvent);
        if (portMessage?.type === 'frame') writer.acceptFrame(portMessage.frame);
        else if (portMessage?.type === 'end' && portMessage.game === message.game) {
          void writer.end().catch((error) => writer.fail(error));
        } else writer.fail(new Error('Writer received an invalid reader message.'));
      });
      framePort.on('close', () => {
        if (!writer.ending && !writer.finalized) writer.fail(new Error('Reader/writer port closed before end-of-stream.'));
      });
      progressPort.on('close', () => {
        if (!writer.finalized && !writer.failed) writer.fail(new Error('Writer/view progress port closed unexpectedly.'));
      });
      framePort.start();
      progressPort.start();
      initialized = true;
      void writer.open().then((filePath) => {
        parentSend({ type: 'ready', service: 'writer', game: message.game, filePath });
      }).catch((error) => writer.fail(error));
    } catch (error) {
      parentSend({ type: 'fatal', service: 'writer', error: error?.message || String(error) });
    }
  });
  parentPort.start?.();
}

runWriterWorker();

module.exports = {
  BATCH_INTERVAL_MS,
  MAX_BATCH_SAMPLES,
  RecordingWriter,
  isContainedPath,
  runWriterWorker,
};
