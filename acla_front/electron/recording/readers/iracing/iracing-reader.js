'use strict';

const { spawn } = require('child_process');
const path = require('path');
const { IRacingAdapter, IRACING_VARIABLES } = require('./iracing-adapter');

const MAX_LINE_BYTES = 8 * 1024 * 1024;

class IRacingReader {
  constructor(options, dependencies = {}) {
    if (options?.runtime !== 'python'
      || ['pythonExecutable', 'scriptDirectory', 'scriptName'].some((key) => typeof options[key] !== 'string' || !options[key])) {
      throw new TypeError('iRacing reader requires a Python capture runtime.');
    }
    this.game = 'iracing';
    this.options = options;
    this.spawn = dependencies.spawn || spawn;
    this.adapter = new IRacingAdapter();
    this.child = null;
    this.stopping = false;
    this.failed = false;
    this.buffer = '';
    this.stderr = '';
    this.pending = false;
    this.ready = false;
    this.startPromise = null;
    this.stopPromise = null;
    this.responseTimer = null;
  }

  start(emit) {
    if (this.startPromise) return this.startPromise;
    this.emit = emit;
    this.startPromise = new Promise((resolve, reject) => {
      this.resolveStart = resolve;
      this.rejectStart = reject;
      try {
        this.child = this.spawn(this.options.pythonExecutable, [
          '-u', path.join(this.options.scriptDirectory, this.options.scriptName),
          '--variables', IRACING_VARIABLES.join(','),
        ], { cwd: this.options.scriptDirectory, stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true });
        this.child.stdout.setEncoding('utf8');
        this.child.stdout.on('data', (chunk) => this.consume(chunk));
        this.child.stderr.on('data', (chunk) => { this.stderr = `${this.stderr}${chunk}`.slice(-4096); });
        this.child.stdin.on('error', (error) => this.fail(error));
        this.child.once('error', (error) => this.fail(error));
        this.child.once('close', (code, signal) => {
          this.child = null;
          clearTimeout(this.responseTimer);
          if (!this.stopping) this.fail(new Error(`iRacing capture exited (code ${code}, signal ${signal})${this.stderr.trim() ? `: ${this.stderr.trim()}` : ''}`));
        });
        this.armTimeout();
      } catch (error) { this.fail(error); }
    });
    return this.startPromise;
  }

  armTimeout() {
    clearTimeout(this.responseTimer);
    this.responseTimer = setTimeout(() => this.fail(new Error('iRacing capture stopped responding.')), 5000);
  }

  requestNext() {
    if (this.stopping || this.failed || this.pending || !this.child) return;
    this.pending = true;
    this.armTimeout();
    this.child.stdin.write('next\n');
  }

  consume(chunk) {
    if (this.stopping || this.failed) return;
    this.buffer += chunk;
    if (Buffer.byteLength(this.buffer, 'utf8') > MAX_LINE_BYTES) return this.fail(new Error('iRacing capture message exceeds the size limit.'));
    const newline = this.buffer.indexOf('\n');
    if (newline < 0) return;
    const line = this.buffer.slice(0, newline);
    this.buffer = this.buffer.slice(newline + 1);
    // Capture sends exactly one response per request, so a second line is a protocol error.
    if (this.buffer.length) return this.fail(new Error('Unexpected unsolicited iRacing capture data.'));
    clearTimeout(this.responseTimer);
    try {
      const packet = JSON.parse(line);
      if (!this.ready) {
        if (packet.type !== 'ready') throw new Error('iRacing capture did not send its ready handshake.');
        this.ready = true;
        this.resolveStart();
        this.requestNext();
        return;
      }
      if (!this.pending) throw new Error('Unexpected iRacing capture response.');
      this.pending = false;
      let delivered;
      if (packet.type === 'sample') {
        const sample = this.adapter.adapt(packet);
        delivered = this.emit({ type: 'frame', frame: { game: this.game, sample } });
      } else if (packet.type === 'disconnected') {
        this.adapter.reset();
      } else if (packet.type !== 'idle') {
        throw new Error(`Unknown iRacing capture message: ${packet.type}`);
      }
      // Consumer backpressure holds the next SDK request, with no raw frame queue.
      Promise.resolve(delivered).then(() => this.requestNext()).catch((error) => this.fail(error));
    } catch (error) { this.fail(error); }
  }

  fail(error) {
    if (this.stopping || this.failed) return;
    this.failed = true;
    clearTimeout(this.responseTimer);
    const failure = error instanceof Error ? error : new Error(String(error));
    this.rejectStart?.(failure);
    this.emit?.({ type: 'fatal', error: failure.message });
    void this.stop();
  }

  stop() {
    if (this.stopPromise) return this.stopPromise;
    this.stopping = true;
    clearTimeout(this.responseTimer);
    this.rejectStart?.(new Error('iRacing capture stopped before startup completed.'));
    const child = this.child;
    this.stopPromise = new Promise((resolve) => {
      if (!child) { resolve(); return; }
      const timer = setTimeout(() => { try { child.kill('SIGKILL'); } catch { /* already exited */ } }, 1000);
      child.once('close', () => { clearTimeout(timer); resolve(); });
      // EOF permits the capture process to release its read-only map and event.
      child.stdin.end();
    });
    return this.stopPromise;
  }
}

module.exports = { IRacingReader, MAX_LINE_BYTES };
