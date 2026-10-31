export const POLICY_CAMERA_WIDTH = 768;
export const POLICY_CAMERA_HEIGHT = 768;
export const POLICY_CAMERA_INPUTS = POLICY_CAMERA_WIDTH * POLICY_CAMERA_HEIGHT * 3;

export interface PolicyCameraImage {
    width: number;
    height: number;
    /** Row-major RGB bytes copied from the clean capture, before overlays or inference. */
    rgb: Uint8Array;
}

export function capturePolicyCameraImage(source: HTMLCanvasElement): PolicyCameraImage {
    const canvas = document.createElement('canvas');
    canvas.width = POLICY_CAMERA_WIDTH;
    canvas.height = POLICY_CAMERA_HEIGHT;
    const context = canvas.getContext('2d');
    if (!context) throw new Error('Camera pixels are unavailable for driving observations.');
    context.drawImage(source, 0, 0, canvas.width, canvas.height);
    const rgba = context.getImageData(0, 0, canvas.width, canvas.height).data;
    const rgb = new Uint8Array(POLICY_CAMERA_INPUTS);
    for (let pixel = 0; pixel < canvas.width * canvas.height; pixel++) {
        rgb[pixel * 3] = rgba[pixel * 4];
        rgb[pixel * 3 + 1] = rgba[pixel * 4 + 1];
        rgb[pixel * 3 + 2] = rgba[pixel * 4 + 2];
    }
    return { width: canvas.width, height: canvas.height, rgb };
}
