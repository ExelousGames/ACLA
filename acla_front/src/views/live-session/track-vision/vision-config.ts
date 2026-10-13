/** Default segmentation input size and shared letterbox coordinate system. */
export const VISION_INPUT_SIZE = 768;
export const DEPTH_INPUT_SIZE = 518;

export const MODEL_INPUT_RESOLUTIONS = {
    segment: [
        { label: 'Low', size: 384 },
        { label: 'Medium', size: 640 },
        { label: 'High', size: VISION_INPUT_SIZE },
    ],
    depth: [
        { label: 'Low', size: 252 },
        { label: 'Medium', size: 392 },
        { label: 'High', size: DEPTH_INPUT_SIZE },
    ],
};
