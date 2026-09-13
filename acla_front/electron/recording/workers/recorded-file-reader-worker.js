'use strict';

const fs = require('fs');
const path = require('path');
const { DESKTOP_GAME_SET } = require('../recording-protocol');
const { validateStandardTelemetrySample } = require('../telemetry-contract');

const READ_CHUNK_ROWS = 250;
const PROGRESS_INTERVAL_MS = 100;

function eventData(event) {
  return event && Object.prototype.hasOwnProperty.call(event, 'data') ? event.data : event;
}

function isContainedPath(parent, candidate) {
  const relative = path.relative(path.resolve(parent), path.resolve(candidate));
  return relative === '' || (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative));
}

async function isRegularFile(filePath, fsModule = fs) {
  try {
    return (await fsModule.promises.stat(filePath)).isFile();
  } catch {
    return false;
  }
}

class RecordedFileReader {
  constructor({ readId, filePath, game, purpose, recordingDirectory, rowLimit, eventPort, parentSend, fsModule = fs }) {
    if (typeof readId !== 'string' || !readId) throw new TypeError('Recorded readId is required.');
    if (typeof filePath !== 'string' || !path.isAbsolute(filePath)) throw new TypeError('Recorded file path must be absolute.');
    if (!DESKTOP_GAME_SET.has(game)) throw new TypeError('Recorded file game is invalid.');
    if (purpose !== 'validate' && purpose !== 'consume') throw new TypeError('Recorded file purpose is invalid.');
    if (typeof recordingDirectory !== 'string' || !path.isAbsolute(recordingDirectory)) {
      throw new TypeError('Recording directory must be absolute.');
    }
    if (rowLimit !== undefined && (!Number.isSafeInteger(rowLimit) || rowLimit < 0)) {
      throw new TypeError('Recorded file row limit must be a non-negative safe integer.');
    }
    this.readId = readId;
    this.filePath = path.resolve(filePath);
    this.game = game;
    this.purpose = purpose;
    this.recordingDirectory = path.resolve(recordingDirectory);
    this.rowLimit = rowLimit ?? null;
    this.eventPort = eventPort;
    this.parentSend = parentSend;
    this.fs = fsModule;
    this.stream = null;
    this.cancelled = false;
    this.terminal = false;
    this.lastProgressAt = 0;
    this.nextChunkIndex = 0;
    this.pendingChunkAck = null;
  }

  send(event) {
    if (!this.terminal) this.eventPort.postMessage({ ...event, readId: this.readId });
  }

  async start() {
    let realDirectory;
    let realFilePath;
    try {
      [realDirectory, realFilePath] = await Promise.all([
        this.fs.promises.realpath(this.recordingDirectory),
        this.fs.promises.realpath(this.filePath),
      ]);
    } catch {
      throw new Error('Recorded file is not an authorized regular file.');
    }
    if (!isContainedPath(realDirectory, realFilePath)
      || !(await isRegularFile(realFilePath, this.fs))) {
      throw new Error('Recorded file is not an authorized regular file.');
    }
    this.filePath = realFilePath;
    this.recordingDirectory = realDirectory;
    const stat = await this.fs.promises.stat(this.filePath);
    const totalBytes = stat.size;
    this.stream = this.fs.createReadStream(this.filePath);

    let pending = Buffer.alloc(0);
    let row = 0;
    let rowCount = 0;
    let lineStartOffset = 0;
    let bytesRead = 0;
    let formatSent = false;
    let rows = [];
    let reachedRowLimit = this.rowLimit === 0;

    const emitProgress = (force = false) => {
      const now = Date.now();
      if (!force && now - this.lastProgressAt < PROGRESS_INTERVAL_MS) return;
      this.lastProgressAt = now;
      this.send({ type: 'progress', rowsRead: rowCount, bytesRead, totalBytes });
    };
    const emitRows = async () => {
      if (this.purpose === 'consume' && rows.length > 0) {
        const chunkIndex = this.nextChunkIndex;
        this.nextChunkIndex += 1;
        const chunkRows = rows;
        rows = [];
        this.send({ type: 'chunk', rows: chunkRows, chunkIndex });
        await new Promise((resolve, reject) => {
          this.pendingChunkAck = { chunkIndex, resolve, reject };
        });
        return;
      }
      rows = [];
    };
    const parseLine = async (lineBuffer, byteOffset) => {
      row += 1;
      let effective = lineBuffer;
      if (effective.length > 0 && effective[effective.length - 1] === 13) {
        effective = effective.subarray(0, effective.length - 1);
      }
      const line = effective.toString('utf8');
      if (!line.trim()) return;
      let sample;
      try {
        sample = JSON.parse(line);
      } catch (error) {
        const failure = new Error(`Malformed JSON at row ${row}, byte ${byteOffset}.`);
        failure.row = row;
        failure.byteOffset = byteOffset;
        throw failure;
      }
      const validation = validateStandardTelemetrySample(sample);
      if (!validation.ok) {
        const failure = new Error(`${validation.error} (row ${row}, byte ${byteOffset})`);
        failure.row = row;
        failure.byteOffset = byteOffset;
        throw failure;
      }
      if (!formatSent) {
        formatSent = true;
        this.send({ type: 'format', format: 'standard-flat', game: this.game });
      }
      rowCount += 1;
      if (this.purpose === 'consume') rows.push(sample);
      if (rows.length >= READ_CHUNK_ROWS) await emitRows();
      return this.rowLimit !== null && rowCount >= this.rowLimit;
    };

    try {
      for await (const chunk of reachedRowLimit ? [] : this.stream) {
        if (this.cancelled) throw new Error('Recorded file read cancelled.');
        const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
        bytesRead += buffer.length;
        pending = pending.length ? Buffer.concat([pending, buffer]) : buffer;
        let newlineIndex;
        while ((newlineIndex = pending.indexOf(10)) !== -1) {
          const line = pending.subarray(0, newlineIndex);
          reachedRowLimit = await parseLine(line, lineStartOffset);
          pending = pending.subarray(newlineIndex + 1);
          lineStartOffset += newlineIndex + 1;
          if (reachedRowLimit) break;
        }
        emitProgress();
        if (reachedRowLimit) break;
      }
      if (!reachedRowLimit && pending.length > 0) {
        reachedRowLimit = await parseLine(pending, lineStartOffset);
      }
      await emitRows();
      if (this.rowLimit !== null && rowCount !== this.rowLimit) {
        throw new Error(`Recorded file ended before the committed row limit of ${this.rowLimit}.`);
      }
      emitProgress(true);
      if (this.cancelled) throw new Error('Recorded file read cancelled.');
      const complete = {
        type: 'complete',
        readId: this.readId,
        format: 'standard-flat',
        game: this.game,
        rowCount,
        totalBytes,
      };
      this.eventPort.postMessage(complete);
      this.terminal = true;
      this.parentSend({ type: 'complete', readId: this.readId });
    } catch (error) {
      if (!this.terminal) {
        try {
          this.eventPort.postMessage({
            type: 'error',
            readId: this.readId,
            message: error?.message || String(error),
            ...(Number.isInteger(error?.row) ? { row: error.row } : {}),
            ...(Number.isInteger(error?.byteOffset) ? { byteOffset: error.byteOffset } : {}),
          });
        } catch { /* preload port already closed */ }
        this.terminal = true;
      }
      this.parentSend({ type: 'failed', readId: this.readId, error: error?.message || String(error) });
    } finally {
      this.stream?.destroy();
      this.stream = null;
      this.pendingChunkAck = null;
      setTimeout(() => {
        try { this.eventPort.close(); } catch { /* already closed */ }
        if (process.parentPort) setTimeout(() => process.exit(0), 10);
      }, 0);
    }
  }

  cancel() {
    if (this.terminal) return;
    this.cancelled = true;
    this.pendingChunkAck?.reject(new Error('Recorded file read cancelled.'));
    this.pendingChunkAck = null;
    this.stream?.destroy(new Error('Recorded file read cancelled.'));
  }

  acknowledgeChunk(chunkIndex) {
    const pending = this.pendingChunkAck;
    if (!pending || chunkIndex !== pending.chunkIndex) {
      this.cancel();
      return;
    }
    this.pendingChunkAck = null;
    pending.resolve();
  }
}

function runRecordedFileReaderWorker() {
  const parentPort = process.parentPort;
  if (!parentPort) return;
  let reader = null;
  let initialized = false;
  let started = false;
  let preloadReady = false;
  const parentSend = (message) => parentPort.postMessage(message);
  parentPort.on('message', (event) => {
    const message = eventData(event);
    if (message?.type === 'cancel') {
      reader?.cancel();
      return;
    }
    if (message?.type === 'start') {
      if (!reader || !initialized || !preloadReady || started) {
        parentSend({ type: 'failed', readId: message.readId, error: 'Recorded reader was not initialized.' });
        return;
      }
      started = true;
      void reader.start();
      return;
    }
    if (message?.type !== 'initialize' || initialized) {
      parentSend({ type: 'failed', readId: message?.readId, error: 'Invalid or duplicate recorded-reader initialization.' });
      return;
    }
    try {
      if (!Array.isArray(message.portRoles)
        || message.portRoles.length !== 1
        || message.portRoles[0] !== 'eventsToPreload'
        || !Array.isArray(event.ports)
        || event.ports.length !== 1) {
        throw new TypeError('Recorded-reader initialization descriptor or port order is invalid.');
      }
      const eventPort = event.ports[0];
      if (typeof eventPort?.on !== 'function' || typeof eventPort?.start !== 'function') {
        throw new TypeError('Recorded-reader event port is invalid.');
      }
      reader = new RecordedFileReader({ ...message, eventPort, parentSend });
      eventPort.on('message', (portEvent) => {
        const portMessage = eventData(portEvent);
        if (portMessage?.type === 'ready' && portMessage.readId === message.readId && !preloadReady) {
          preloadReady = true;
          parentSend({ type: 'preload-ready', readId: message.readId });
        } else if (portMessage?.type === 'chunk-consumed'
          && portMessage.readId === message.readId
          && Number.isSafeInteger(portMessage.chunkIndex)
          && portMessage.chunkIndex >= 0) {
          reader.acknowledgeChunk(portMessage.chunkIndex);
        } else reader.cancel();
      });
      eventPort.on('close', () => reader.cancel());
      eventPort.start();
      initialized = true;
      parentSend({ type: 'ready', readId: message.readId });
    } catch (error) {
      parentSend({ type: 'failed', readId: message.readId, error: error?.message || String(error) });
    }
  });
  parentPort.start?.();
}

runRecordedFileReaderWorker();

module.exports = {
  PROGRESS_INTERVAL_MS,
  READ_CHUNK_ROWS,
  RecordedFileReader,
  isRegularFile,
  runRecordedFileReaderWorker,
};
