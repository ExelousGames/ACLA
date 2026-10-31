/** Records API usage in unit tests. Shader arithmetic is verified in Chromium separately. */
export function mockGpuDevice() {
    Object.assign(globalThis, {
        GPUBufferUsage: { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 },
        GPUTextureUsage: { COPY_DST: 2, TEXTURE_BINDING: 4, RENDER_ATTACHMENT: 16 }, GPUMapMode: { READ: 1 },
    });
    const pass = { setPipeline: jest.fn(), setBindGroup: jest.fn(), dispatchWorkgroups: jest.fn(), draw: jest.fn(), end: jest.fn() };
    const device = {
        createBuffer: jest.fn(({ size }) => ({ size, destroy: jest.fn(), mapAsync: jest.fn().mockResolvedValue(undefined),
            getMappedRange: () => new Float32Array([0.1, 1]).buffer })),
        createTexture: jest.fn(({ size }) => ({ width: size[0], height: size[1], createView: jest.fn(() => ({})), destroy: jest.fn() })),
        createShaderModule: jest.fn((descriptor) => descriptor),
        createComputePipeline: jest.fn(() => ({ getBindGroupLayout: jest.fn(() => ({})) })),
        createRenderPipeline: jest.fn(() => ({ getBindGroupLayout: jest.fn(() => ({})) })),
        createBindGroup: jest.fn((descriptor) => descriptor), createSampler: jest.fn(() => ({})),
        createCommandEncoder: jest.fn(() => ({ beginComputePass: () => pass, beginRenderPass: () => pass,
            copyBufferToBuffer: jest.fn(), finish: jest.fn(() => ({})) })),
        queue: { writeBuffer: jest.fn(), copyExternalImageToTexture: jest.fn(), submit: jest.fn(),
            onSubmittedWorkDone: jest.fn().mockResolvedValue(undefined) },
    };
    return { device: device as unknown as GPUDevice, calls: device, pass };
}
