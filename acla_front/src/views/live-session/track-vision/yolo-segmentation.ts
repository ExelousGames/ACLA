export interface FloatTensor { dims: readonly number[]; data: Float32Array }

export function letterbox(width: number, height: number, size: number) {
    const scale = Math.min(size / width, size / height);
    const resizedWidth = Math.round(width * scale);
    const resizedHeight = Math.round(height * scale);
    return {
        resizedWidth, resizedHeight,
        padX: Math.floor((size - resizedWidth) / 2),
        padY: Math.floor((size - resizedHeight) / 2),
    };
}

export function rgbaToChw(rgba: Uint8ClampedArray): Float32Array {
    const pixels = rgba.length / 4;
    const data = new Float32Array(pixels * 3);
    for (let i = 0; i < pixels; i++) {
        data[i] = rgba[i * 4] / 255;
        data[pixels + i] = rgba[i * 4 + 1] / 255;
        data[pixels * 2 + i] = rgba[i * 4 + 2] / 255;
    }
    return data;
}
