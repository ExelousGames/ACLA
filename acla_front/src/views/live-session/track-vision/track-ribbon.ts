export interface ImagePoint { x: number; y: number }
export interface TrackRibbonPair {
    left: ImagePoint;
    right: ImagePoint;
}
export interface TrackRibbon {
    /** Fifty cross-sections ordered from the distant end toward the camera. */
    pairs: TrackRibbonPair[];
}

const RIBBON_PAIRS = 50;

/** Fit paired mask observations to a fixed ribbon, without extrapolating across gaps. */
export function fitTrackRibbon(observations: TrackRibbonPair[]): TrackRibbon | null {
    if (observations.length < 2) return null;
    const firstY = observations[0].left.y, lastY = observations[observations.length - 1].left.y;
    if (lastY <= firstY) return null;
    const spacing = (lastY - firstY) / (RIBBON_PAIRS - 1);
    const pixelHeight = (lastY - firstY) / (observations.length - 1);
    // Local linear regression preserves perspective tapers and follows multiple bends.
    // A shared neighborhood for both edges keeps their cross-sections aligned.
    const bandwidth = Math.max(pixelHeight * 2.5, spacing * 1.5);
    const pairs = Array.from({ length: RIBBON_PAIRS }, (_, index) => {
        const y = firstY + index * spacing;
        const samples = observations.map((pair) => {
            const dy = (pair.left.y - y) / bandwidth;
            const weight = Math.abs(dy) < 1 ? (1 - Math.abs(dy) ** 3) ** 3 : 0;
            return { pair, dy, weight };
        }).filter(({ weight }) => weight > 0);
        let sum = 0, sumY = 0, sumYY = 0;
        for (const { dy, weight } of samples) {
            sum += weight; sumY += weight * dy; sumYY += weight * dy * dy;
        }
        const determinant = sum * sumYY - sumY * sumY;
        const fitEdge = (side: 'left' | 'right'): ImagePoint => {
            let sumX = 0, sumXY = 0, minX = Infinity, maxX = -Infinity;
            for (const { pair, dy, weight } of samples) {
                const x = pair[side].x;
                sumX += weight * x; sumXY += weight * x * dy;
                minX = Math.min(minX, x); maxX = Math.max(maxX, x);
            }
            const x = determinant > 1e-10 ? (sumX * sumYY - sumXY * sumY) / determinant : sumX / sum;
            return { x: Math.max(minX, Math.min(maxX, x)), y };
        };
        const left = fitEdge('left'), right = fitEdge('right');
        // Keep the ribbon open even if a sharp width change overshoots the local fit.
        if (right.x <= left.x) {
            left.x = samples.reduce((value, { pair, weight }) => value + pair.left.x * weight, 0) / sum;
            right.x = samples.reduce((value, { pair, weight }) => value + pair.right.x * weight, 0) / sum;
        }
        return { left, right };
    });
    return { pairs };
}
