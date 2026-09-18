// All detectors in this renderer share ONNX's WebGPU runtime and must take turns.
let pending: Promise<unknown> = Promise.resolve();

/** Queue a complete GPU operation, including cleanup. Callbacks must not enqueue more work. */
export function runWithVisionGpuQueue<T>(provider: 'webgpu' | 'wasm', operation: () => Promise<T>): Promise<T> {
    if (provider !== 'webgpu') return operation();
    const result = pending.then(operation);
    // Preserve the caller's error without blocking the next detector.
    pending = result.catch(() => undefined);
    return result;
}
