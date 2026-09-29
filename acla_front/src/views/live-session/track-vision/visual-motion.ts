import type { GroundPoint } from './track-vision-types';

export interface MotionImage {
    width: number;
    height: number;
    gray: Uint8Array;
    /** Only confidently segmented static surfaces may supply motion features. */
    support: Uint8Array;
    lift(x: number, y: number): GroundPoint | null;
}
export interface PointMatch { previous: GroundPoint; current: GroundPoint }
export interface RigidMotion { rotation: number[]; translation: GroundPoint }
const axes = ['x', 'y', 'z'] as const;
const distance = (a: GroundPoint, b: GroundPoint) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);

export function transformPoint(point: GroundPoint, motion: RigidMotion): GroundPoint {
    const r = motion.rotation, t = motion.translation;
    return { x: r[0] * point.x + r[1] * point.y + r[2] * point.z + t.x,
        y: r[3] * point.x + r[4] * point.y + r[5] * point.z + t.y,
        z: r[6] * point.x + r[7] * point.y + r[8] * point.z + t.z };
}

function spread(points: GroundPoint[]) {
    const a = points[0];
    const b = points.reduce((best, point) => distance(a, point) > distance(a, best) ? point : best, a);
    const length = distance(a, b);
    if (length < 1) return false;
    // Collinear matches cannot constrain rotation around their common axis.
    return points.some((c) => Math.hypot(
        (b.y - a.y) * (c.z - a.z) - (b.z - a.z) * (c.y - a.y),
        (b.z - a.z) * (c.x - a.x) - (b.x - a.x) * (c.z - a.z),
        (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)) / length > 0.5);
}

/** Horn's unit-quaternion rigid fit, with fixed scale (monocular depth supplies meters).
 * https://people.csail.mit.edu/bkph/papers/Absolute_Orientation.pdf */
function fitRigid(matches: PointMatch[]): RigidMotion | null {
    if (!spread(matches.map((m) => m.previous)) || !spread(matches.map((m) => m.current))) return null;
    const centroid = (key: 'previous' | 'current') => Object.fromEntries(axes.map((axis) =>
        [axis, matches.reduce((sum, match) => sum + match[key][axis], 0) / matches.length])) as unknown as GroundPoint;
    const p = centroid('previous'), q = centroid('current');
    const s = axes.flatMap((a) => axes.map((b) => matches.reduce((sum, m) =>
        sum + (m.previous[a] - p[a]) * (m.current[b] - q[b]), 0)));
    const [xx, xy, xz, yx, yy, yz, zx, zy, zz] = s;
    const matrix = [
        [xx + yy + zz, yz - zy, zx - xz, xy - yx],
        [yz - zy, xx - yy - zz, xy + yx, zx + xz],
        [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
        [xy - yx, zx + xz, yz + zy, -xx - yy + zz],
    ];
    const vectors = matrix.map((_, i) => matrix.map((__, j) => Number(i === j)));
    // Jacobi diagonalization finds the largest algebraic eigenvalue, including planar scenes.
    for (let iteration = 0; iteration < 40; iteration++) {
        let a = 0, b = 1;
        for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) {
            if (Math.abs(matrix[i][j]) > Math.abs(matrix[a][b])) { a = i; b = j; }
        }
        if (Math.abs(matrix[a][b]) < 1e-9) break;
        const angle = 0.5 * Math.atan2(2 * matrix[a][b], matrix[b][b] - matrix[a][a]);
        const c = Math.cos(angle), sn = Math.sin(angle);
        const aa = matrix[a][a], bb = matrix[b][b], ab = matrix[a][b];
        for (let k = 0; k < 4; k++) {
            if (k !== a && k !== b) {
                const ka = matrix[k][a], kb = matrix[k][b];
                matrix[k][a] = matrix[a][k] = c * ka - sn * kb;
                matrix[k][b] = matrix[b][k] = sn * ka + c * kb;
            }
            const va = vectors[k][a], vb = vectors[k][b];
            vectors[k][a] = c * va - sn * vb;
            vectors[k][b] = sn * va + c * vb;
        }
        matrix[a][a] = c * c * aa - 2 * sn * c * ab + sn * sn * bb;
        matrix[b][b] = sn * sn * aa + 2 * sn * c * ab + c * c * bb;
        matrix[a][b] = matrix[b][a] = 0;
    }
    const best = [0, 1, 2, 3].reduce((a, b) => matrix[a][a] > matrix[b][b] ? a : b);
    const [w, x, y, z] = vectors.map((row) => row[best]);
    const rotation = [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)];
    const rotated = transformPoint(p, { rotation, translation: { x: 0, y: 0, z: 0 } });
    return { rotation, translation: { x: q.x - rotated.x, y: q.y - rotated.y, z: q.z - rotated.z } };
}

/** Estimate the transform from the previous local frame into the current local frame. */
export function estimateRigidMotion(input: PointMatch[]) {
    const matches = input.filter((m) => [...Object.values(m.previous), ...Object.values(m.current)].every(Number.isFinite));
    if (matches.length < 12) return null;
    const threshold = (m: PointMatch) => Math.min(0.8, 0.15 + Math.min(m.previous.y, m.current.y) * 0.012);
    const inliersFor = (motion: RigidMotion) => matches.filter((m) => distance(transformPoint(m.previous, motion), m.current) < threshold(m));
    let best: PointMatch[] = [];
    let seed = 173;
    const randomIndex = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed % matches.length; };
    for (let iteration = 0; iteration < 96; iteration++) {
        const sample = iteration === 0 ? matches : [matches[randomIndex()], matches[randomIndex()], matches[randomIndex()]];
        const motion = fitRigid(sample);
        if (!motion) continue;
        const inliers = inliersFor(motion);
        if (inliers.length > best.length) best = inliers;
    }
    const minimum = Math.max(12, Math.ceil(matches.length * 0.65));
    if (best.length < minimum) return null;
    let motion = fitRigid(best);
    if (!motion) return null;
    best = inliersFor(motion);
    if (best.length < minimum) return null;
    motion = fitRigid(best);
    if (!motion) return null;
    const errors = best.map((m) => distance(transformPoint(m.previous, motion!), m.current));
    const rmseM = Math.sqrt(errors.reduce((sum, error) => sum + error * error, 0) / errors.length);
    const angle = Math.acos(Math.max(-1, Math.min(1, (motion.rotation[0] + motion.rotation[4] + motion.rotation[8] - 1) / 2)));
    if (rmseM > 0.5 || angle > Math.PI / 9 || Math.hypot(...Object.values(motion.translation)) > 20) return null;
    return { motion, inliers: best.length, rmseM };
}

const PATCH = 2;
const SEARCH = 16;
function validPatch(image: MotionImage, x: number, y: number) {
    if (x < PATCH + 1 || y < PATCH + 1 || x >= image.width - PATCH - 1 || y >= image.height - PATCH - 1) return false;
    for (let dy = -PATCH; dy <= PATCH; dy++) for (let dx = -PATCH; dx <= PATCH; dx++) {
        if (!image.support[(y + dy) * image.width + x + dx]) return false;
    }
    return true;
}

function corners(image: MotionImage) {
    const candidates: Array<{ x: number; y: number; score: number }> = [];
    for (let y = 4; y < image.height - 4; y += 2) for (let x = 4; x < image.width - 4; x += 2) {
        if (!validPatch(image, x, y)) continue;
        let xx = 0, xy = 0, yy = 0;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
            const i = (y + dy) * image.width + x + dx;
            const gx = image.gray[i + 1] - image.gray[i - 1], gy = image.gray[i + image.width] - image.gray[i - image.width];
            xx += gx * gx; xy += gx * gy; yy += gy * gy;
        }
        const score = (xx + yy - Math.hypot(xx - yy, 2 * xy)) / 2;
        if (score > 1200) candidates.push({ x, y, score });
    }
    candidates.sort((a, b) => b.score - a.score);
    const selected: typeof candidates = [];
    for (const point of candidates) {
        if (selected.every((other) => Math.hypot(other.x - point.x, other.y - point.y) >= 10)) selected.push(point);
        if (selected.length === 96) break;
    }
    return selected;
}

function matchPatch(source: MotionImage, target: MotionImage, x: number, y: number) {
    const patch: number[] = [];
    for (let dy = -PATCH; dy <= PATCH; dy++) for (let dx = -PATCH; dx <= PATCH; dx++) patch.push(source.gray[(y + dy) * source.width + x + dx]);
    const scores: Array<{ x: number; y: number; error: number }> = [];
    for (let ty = Math.max(3, y - SEARCH); ty <= Math.min(target.height - 4, y + SEARCH); ty++) {
        for (let tx = Math.max(3, x - SEARCH); tx <= Math.min(target.width - 4, x + SEARCH); tx++) {
            if (!validPatch(target, tx, ty)) continue;
            let difference = 0, squared = 0, i = 0;
            for (let dy = -PATCH; dy <= PATCH; dy++) for (let dx = -PATCH; dx <= PATCH; dx++) {
                const delta = patch[i++] - target.gray[(ty + dy) * target.width + tx + dx];
                difference += delta; squared += delta * delta;
            }
            // Zero-mean SSD tolerates small exposure changes without accepting flat patches.
            scores.push({ x: tx, y: ty, error: Math.max(0, (squared - difference * difference / patch.length) / patch.length) });
        }
    }
    scores.sort((a, b) => a.error - b.error);
    const best = scores[0];
    const alternative = best && scores.find((p) => Math.hypot(p.x - best.x, p.y - best.y) > 2);
    return best && alternative && best.error < 500 && best.error < alternative.error * 0.65 ? best : null;
}

/** Sparse, mutual patch matches; no simulator motion or persistent object tracks. */
export function matchVisualFeatures(previous: MotionImage, current: MotionImage): PointMatch[] {
    if (previous.width !== current.width || previous.height !== current.height) return [];
    const matches: PointMatch[] = [], used = new Set<number>();
    for (const feature of corners(previous)) {
        const p = previous.lift(feature.x, feature.y);
        if (!p) continue;
        const match = matchPatch(previous, current, feature.x, feature.y);
        if (!match || used.has(match.y * current.width + match.x)) continue;
        const reverse = matchPatch(current, previous, match.x, match.y);
        if (!reverse || Math.hypot(reverse.x - feature.x, reverse.y - feature.y) > 1) continue;
        const q = current.lift(match.x, match.y);
        if (!q) continue;
        used.add(match.y * current.width + match.x);
        matches.push({ previous: p, current: q });
    }
    return matches;
}
