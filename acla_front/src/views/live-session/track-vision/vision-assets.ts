export function visionAssetUrl(asset: string): string {
    const publicPath = process.env.PUBLIC_URL;
    const base = window.location.protocol === 'file:'
        ? new URL('./', window.location.href).href
        : new URL(publicPath && publicPath !== '.' ? `${publicPath.replace(/\/$/, '')}/` : '/', window.location.origin).href;
    return new URL(asset, base).href;
}

/** XHR supports packaged Electron file:// assets and the Electron development server's HTTP assets. */
export function readVisionModel(url: string): Promise<ArrayBuffer> {
    return new Promise((resolve, reject) => {
        const request = new XMLHttpRequest();
        request.open('GET', url);
        request.responseType = 'arraybuffer';
        request.timeout = 60000;
        const fail = () => reject(new Error('Built-in model unavailable. Run npm run setup:vision, then restart the app, or load a custom ONNX model.'));
        request.onload = () => {
            const success = request.status === 200 || (new URL(url).protocol === 'file:' && request.status === 0);
            if (success && request.response instanceof ArrayBuffer && request.response.byteLength) resolve(request.response);
            else fail();
        };
        request.onerror = fail;
        request.ontimeout = fail;
        request.send();
    });
}
