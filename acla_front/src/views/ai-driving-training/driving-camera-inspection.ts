/** Post-activation convolution outputs, stored in row/column/channel order. */
export interface DrivingCameraLayer {
    index: number;
    width: number;
    height: number;
    channels: number;
    values: Float32Array;
}

export interface DrivingCameraInspection { layers: DrivingCameraLayer[] }

/** Raw activations remain comparable across frames; display normalization must not affect ranking. */
export function cameraChannelStats(layer: DrivingCameraLayer, channel: number): { mean: number; peak: number; peakIndex: number } {
    if (!Number.isInteger(channel) || channel < 0 || channel >= layer.channels) throw new Error('Select a valid camera channel.');
    const pixels = layer.width * layer.height;
    if (!pixels || layer.values.length !== pixels * layer.channels) throw new Error('Invalid camera feature map shape.');
    let sum = 0, peak = -Infinity, peakIndex = 0;
    for (let pixel = 0; pixel < pixels; pixel++) {
        const value = layer.values[pixel * layer.channels + channel];
        if (!Number.isFinite(value)) throw new Error('The camera feature map contains invalid activations.');
        sum += value;
        if (value > peak) { peak = value; peakIndex = pixel; }
    }
    return { mean: sum / pixels, peak, peakIndex };
}
