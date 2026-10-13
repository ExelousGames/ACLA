import type { GroundPoint, RoadPolynomial } from './track-vision-types';

export const evaluateRoad = (curve: RoadPolynomial, y: number) => {
    const [a, b, c] = curve.coefficients;
    return a + y * (b + y * c);
};
export const roadCurvature = (curve: RoadPolynomial, y: number) =>
    2 * curve.coefficients[2] / (1 + (curve.coefficients[1] + 2 * curve.coefficients[2] * y) ** 2) ** 1.5;

/** Robust quadratic least squares, with normalized distances to condition the solve. */
export function fitRoadPolynomial(points: GroundPoint[]): RoadPolynomial | null {
    if (points.length < 8 || points.some(({ x, y }) => !Number.isFinite(x) || !Number.isFinite(y))) return null;
    const minY = Math.min(...points.map(({ y }) => y)), maxY = Math.max(...points.map(({ y }) => y));
    const middle = (minY + maxY) / 2, scale = (maxY - minY) / 2;
    if (scale < 5) return null;
    let weights = points.map(() => 1);
    let coefficients: [number, number, number] = [0, 0, 0];
    for (let pass = 0; pass < 4; pass++) {
        const matrix = Array.from({ length: 3 }, () => [0, 0, 0, 0]);
        const passWeights = weights;
        points.forEach(({ x, y }, index) => {
            const t = (y - middle) / scale, basis = [1, t, t * t], weight = passWeights[index];
            for (let row = 0; row < 3; row++) {
                for (let column = 0; column < 3; column++) matrix[row][column] += weight * basis[row] * basis[column];
                matrix[row][3] += weight * basis[row] * x;
            }
        });
        for (let column = 0; column < 3; column++) {
            let pivot = column;
            for (let row = column + 1; row < 3; row++) if (Math.abs(matrix[row][column]) > Math.abs(matrix[pivot][column])) pivot = row;
            [matrix[column], matrix[pivot]] = [matrix[pivot], matrix[column]];
            const divisor = matrix[column][column];
            if (Math.abs(divisor) < 1e-8) return null;
            for (let j = column; j < 4; j++) matrix[column][j] /= divisor;
            for (let row = 0; row < 3; row++) {
                if (row === column) continue;
                const factor = matrix[row][column];
                for (let j = column; j < 4; j++) matrix[row][j] -= factor * matrix[column][j];
            }
        }
        const [a, b, c] = matrix.map((row) => row[3]);
        const fitted: [number, number, number] = [a - b * middle / scale + c * middle ** 2 / scale ** 2,
            b / scale - 2 * c * middle / scale ** 2, c / scale ** 2];
        coefficients = fitted;
        const residuals = points.map(({ x, y }) => Math.abs(x - fitted[0] - fitted[1] * y - fitted[2] * y * y));
        const sorted = [...residuals].sort((x, y) => x - y);
        const cutoff = Math.max(0.15, 2.5 * sorted[Math.floor(sorted.length / 2)]);
        weights = residuals.map((error) => Math.min(1, cutoff / Math.max(error, 1e-8)));
    }
    const rmseM = Math.sqrt(points.reduce((sum, { x, y }) => sum
        + (x - coefficients[0] - coefficients[1] * y - coefficients[2] * y * y) ** 2, 0) / points.length);
    return { coefficients, minY, maxY, rmseM };
}
