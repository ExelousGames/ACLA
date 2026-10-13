// TypeScript 4's node resolver does not follow package.json subpath exports.
// These runtime entries expose the same public API as the main runtime entry.
declare module 'onnxruntime-web/wasm' {
    export * from 'onnxruntime-web';
}

declare module 'onnxruntime-web/webgpu' {
    export * from 'onnxruntime-web';
}
