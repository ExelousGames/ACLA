/**
 * AudioWorklet processor for capturing mic input as PCM16 mono at the
 * AudioContext's sample rate (Phase 3 — voice conversation).
 *
 * Loaded by `audioContext.audioWorklet.addModule('/pcm-capture-worklet.js')`
 * from the renderer process, then a `new AudioWorkletNode(ctx, 'pcm-capture')`
 * forwards Float32 samples here. We convert to Int16 and post them back to
 * the main thread, which forwards them over the WebSocket to Pipecat.
 *
 * The worklet runs on the audio rendering thread — keep the math in
 * `process()` tight. ~3ms of audio per quantum (128 samples at 48kHz).
 */

class PcmCaptureProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
    }

    /**
     * Pulled by the audio engine every quantum.
     * inputs: Float32Array[][]  — [input][channel][sample]
     * outputs: ignored (this is a capture-only node)
     */
    process(inputs) {
        const input = inputs[0];
        if (!input || input.length === 0) return true;

        // Down-mix to mono if the source is stereo.
        const ch0 = input[0];
        if (!ch0 || ch0.length === 0) return true;

        const monoFloat = ch0;
        // Convert float [-1, 1] to int16 [-32768, 32767].
        const pcm16 = new Int16Array(monoFloat.length);
        for (let i = 0; i < monoFloat.length; i++) {
            // Clamp + scale + round-to-nearest.
            const s = Math.max(-1, Math.min(1, monoFloat[i]));
            pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }

        // Transfer the underlying buffer to avoid a copy.
        this.port.postMessage(pcm16.buffer, [pcm16.buffer]);
        return true;
    }
}

registerProcessor('pcm-capture', PcmCaptureProcessor);
