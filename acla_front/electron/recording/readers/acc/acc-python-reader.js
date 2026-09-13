'use strict';

const { spawn: defaultSpawn } = require('child_process');
const path = require('path');
const readline = require('readline');
const { validateStandardTelemetrySample } = require('../../telemetry-contract');

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

class AccPythonReader {
  constructor(options, dependencies = {}) {
    if (!options || options.runtime !== 'python') {
      throw new TypeError('ACC reader options must select the Python runtime.');
    }
    for (const key of ['pythonExecutable', 'scriptDirectory', 'scriptName']) {
      if (typeof options[key] !== 'string' || !options[key]) {
        throw new TypeError(`ACC reader option ${key} is required.`);
      }
    }
    this.game = 'acc';
    this.options = Object.freeze({ ...options });
    this.spawn = dependencies.spawn || defaultSpawn;
    this.createInterface = dependencies.createInterface || readline.createInterface;
    this.child = null;
    this.lines = null;
    this.emit = null;
    this.startPromise = null;
    this.stopPromise = null;
    this.started = false;
    this.stopping = false;
    this.stderr = '';
  }

  start(emit) {
    if (this.startPromise) return this.startPromise;
    if (typeof emit !== 'function') {
      return Promise.reject(new TypeError('ACC reader start requires an event callback.'));
    }
    this.emit = emit;

    this.startPromise = new Promise((resolve, reject) => {
      let settled = false;
      const resolveStarted = () => {
        if (settled) return;
        settled = true;
        this.started = true;
        resolve();
      };
      const rejectStarted = (error) => {
        if (settled) return;
        settled = true;
        reject(error instanceof Error ? error : new Error(String(error)));
      };

      const scriptPath = path.join(this.options.scriptDirectory, this.options.scriptName);
      try {
        this.child = this.spawn(
          this.options.pythonExecutable,
          ['-u', scriptPath, '--stream'],
          {
            cwd: this.options.scriptDirectory,
            stdio: ['ignore', 'pipe', 'pipe'],
            windowsHide: true,
          },
        );
      } catch (error) {
        rejectStarted(error);
        return;
      }

      const child = this.child;
      if (!child?.stdout || !child?.stderr || typeof child.on !== 'function') {
        rejectStarted(new Error('ACC Python reader did not provide piped output streams.'));
        return;
      }

      child.stderr.on('data', (chunk) => {
        this.stderr = `${this.stderr}${String(chunk)}`.slice(-4096);
      });

      this.lines = this.createInterface({ input: child.stdout, crlfDelay: Infinity });
      this.lines.on('line', (line) => {
        if (this.stopping || typeof line !== 'string' || !line.trim()) return;
        let sample;
        try {
          sample = JSON.parse(line);
        } catch (error) {
          const failure = new Error(`ACC reader emitted invalid JSON: ${errorMessage(error)}`);
          rejectStarted(failure);
          this._emitFatal(failure.message);
          return;
        }

        if (sample && sample.available === false && Object.keys(sample).length === 1) {
          return;
        }
        const validation = validateStandardTelemetrySample(sample);
        if (!validation.ok) {
          const failure = new Error(validation.error);
          rejectStarted(failure);
          this._emitFatal(failure.message);
          return;
        }

        const frame = { game: 'acc', sample };
        this.emit?.({ type: 'frame', frame });
        resolveStarted();
      });

      child.once('error', (error) => {
        rejectStarted(error);
        if (!this.stopping) this._emitFatal(`ACC Python reader failed: ${errorMessage(error)}`);
      });
      child.once('close', (code, signal) => {
        this.child = null;
        this.lines?.close?.();
        this.lines = null;
        if (this.stopping) return;
        const detail = this.stderr.trim();
        const failure = new Error(
          `ACC Python reader exited before stop (code ${String(code)}, signal ${String(signal)})${detail ? `: ${detail}` : ''}`,
        );
        rejectStarted(failure);
        this._emitFatal(failure.message);
      });
    });

    return this.startPromise;
  }

  _emitFatal(message) {
    if (!this.stopping && this.emit) this.emit({ type: 'fatal', error: message });
  }

  stop() {
    if (this.stopPromise) return this.stopPromise;
    this.stopping = true;
    this.emit = null;
    const child = this.child;
    this.stopPromise = (async () => {
      this.lines?.close?.();
      this.lines = null;
      if (!child) return;

      await new Promise((resolve) => {
        let settled = false;
        const finish = () => {
          if (settled) return;
          settled = true;
          clearTimeout(forceTimer);
          resolve();
        };
        child.once('close', finish);
        const forceTimer = setTimeout(() => {
          try { child.kill('SIGKILL'); } catch { /* already gone */ }
          finish();
        }, 2000);
        try {
          if (!child.killed) child.kill('SIGTERM');
          else finish();
        } catch {
          finish();
        }
      });
      if (this.child === child) this.child = null;
    })();
    return this.stopPromise;
  }
}

module.exports = { AccPythonReader };
