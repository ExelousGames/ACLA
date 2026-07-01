/**
 * AudioWorklet processor for capturing mic input as PCM16 mono at the
 * AudioContext's sample rate.
 *
 * Posts two kinds of messages on the port:
 *   { type: 'pcm',   buffer: ArrayBuffer }            // transferred, every quantum
 *   { type: 'level', rms: number, peak: number }     // ~15Hz, both in [0, 1]
 */

const LEVEL_HZ = 15;

class PcmCaptureProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this._levelSamples = 0;
        this._levelSumSq = 0;
        this._levelPeak = 0;
        this._levelInterval = Math.max(1, Math.floor(sampleRate / LEVEL_HZ));
    }

    process(inputs) {
        const input = inputs[0];
        if (!input || input.length === 0) return true;

        const ch0 = input[0];
        if (!ch0 || ch0.length === 0) return true;

        const monoFloat = ch0;
        const pcm16 = new Int16Array(monoFloat.length);
        let sumSq = 0;
        let peak = 0;
        for (let i = 0; i < monoFloat.length; i++) {
            const s = Math.max(-1, Math.min(1, monoFloat[i]));
            const abs = s < 0 ? -s : s;
            if (abs > peak) peak = abs;
            sumSq += s * s;
            pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }

        this._levelSumSq += sumSq;
        this._levelSamples += monoFloat.length;
        if (peak > this._levelPeak) this._levelPeak = peak;

        this.port.postMessage({ type: 'pcm', buffer: pcm16.buffer }, [pcm16.buffer]);

        if (this._levelSamples >= this._levelInterval) {
            const rms = Math.sqrt(this._levelSumSq / this._levelSamples);
            this.port.postMessage({ type: 'level', rms, peak: this._levelPeak });
            this._levelSumSq = 0;
            this._levelSamples = 0;
            this._levelPeak = 0;
        }

        return true;
    }
}

registerProcessor('pcm-capture', PcmCaptureProcessor);
