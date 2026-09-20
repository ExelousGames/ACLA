import { readVisionModel, visionAssetUrl } from './vision-assets';

let request: { open: jest.Mock; send: jest.Mock; status: number; response: ArrayBuffer | null; onload: () => void; onerror: () => void; ontimeout: () => void };
beforeEach(() => {
    request = { open: jest.fn(), send: jest.fn(), status: 200, response: new ArrayBuffer(16), onload: () => {}, onerror: () => {}, ontimeout: () => {} };
    jest.spyOn(window, 'XMLHttpRequest').mockImplementation(() => request as unknown as XMLHttpRequest);
});
afterEach(() => jest.restoreAllMocks());

it('resolves local HTTP assets and accepts packaged Electron file responses', async () => {
    expect(visionAssetUrl('vision-models/yolo26n-depth.onnx')).toBe('http://localhost/vision-models/yolo26n-depth.onnx');
    const http = readVisionModel('http://localhost/vision-models/yolo26n-depth.onnx');
    request.onload();
    await expect(http).resolves.toBe(request.response);
    const packaged = readVisionModel('file:///app/build/vision-models/yolo26n-depth.onnx');
    request.status = 0;
    request.onload();
    await expect(packaged).resolves.toBe(request.response);
});

it.each([0, 404, 500])('rejects failed HTTP responses (%s)', async (status) => {
    const promise = readVisionModel('http://localhost/model.onnx');
    request.status = status;
    request.onload();
    await expect(promise).rejects.toThrow('setup:vision');
});

it('rejects empty weights and network timeouts with recovery instructions', async () => {
    const empty = readVisionModel('file:///app/model.onnx');
    request.response = new ArrayBuffer(0);
    request.onload();
    await expect(empty).rejects.toThrow('setup:vision-models');
    const timedOut = readVisionModel('http://localhost/model.onnx');
    request.ontimeout();
    await expect(timedOut).rejects.toThrow('setup:vision');
});

const publicUrl = process.env.PUBLIC_URL;
afterEach(() => { process.env.PUBLIC_URL = publicUrl; });

it('resolves the locally shipped runtime at the root or configured public path', () => {
    process.env.PUBLIC_URL = '.';
    expect(visionAssetUrl('vision-runtime/')).toBe('http://localhost/vision-runtime/');
    process.env.PUBLIC_URL = '/app';
    expect(visionAssetUrl('vision-runtime/')).toBe('http://localhost/app/vision-runtime/');
});
