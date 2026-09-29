import { createCameraProjection } from './camera-projection';
import { createDepthProjection } from './depth-projection';
import { createSegmentationLayers } from './segmentation-layers';
import { letterbox } from './yolo-segmentation';
import { estimateRigidMotion, matchVisualFeatures, MotionImage, transformPoint } from './visual-motion';
import type { GroundPoint, LocalTrackScene, SceneMemoryPoint, TrackSceneMemory, TrackVisionFrame } from './track-vision-types';
import { VISION_MAX_AGE_MS } from './track-vision-types';

export const SCENE_MEMORY_MAX_AGE_MS = 3000;
export const SCENE_MEMORY_MAX_POINTS = 2000;
const MAX_FRAME_GAP_MS = 1000;
const VOXEL_M = 0.4;
const inRange = (p: GroundPoint) => [p.x, p.y, p.z].every(Number.isFinite)
    && p.y >= -10 && p.y <= 80 && Math.abs(p.x) <= 40 && Math.abs(p.z) <= 15;
export interface SceneImage { width: number; height: number; gray: Uint8Array }

/** Read only the clean captured image, never the segmentation/camera overlays. */
export function readSceneImage(source: HTMLCanvasElement, scratch: HTMLCanvasElement): SceneImage {
    const scale = Math.min(1, 256 / Math.max(source.width, source.height));
    scratch.width = Math.max(1, Math.round(source.width * scale));
    scratch.height = Math.max(1, Math.round(source.height * scale));
    const context = scratch.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('Scene memory image is unavailable.');
    context.drawImage(source, 0, 0, scratch.width, scratch.height);
    const { data } = context.getImageData(0, 0, scratch.width, scratch.height);
    const gray = new Uint8Array(scratch.width * scratch.height);
    for (let i = 0; i < gray.length; i++) gray[i] = (77 * data[4 * i] + 150 * data[4 * i + 1] + 29 * data[4 * i + 2]) >> 8;
    return { width: scratch.width, height: scratch.height, gray };
}

function observe(frame: TrackVisionFrame, scene: LocalTrackScene, pixels: SceneImage) {
    const depth = createDepthProjection(frame), segment = frame.detections.segment;
    if (!depth || segment?.task !== 'segment' || pixels.width < 8 || pixels.height < 8
        || pixels.width > 256 || pixels.height > 256 || pixels.gray.length !== pixels.width * pixels.height) return null;
    // Lower-confidence traffic/cockpit masks still veto static samples.
    const layers = createSegmentationLayers(segment, 0.35);
    if (!layers) return null;
    const staticMask = new Uint8Array(segment.width * segment.height);
    for (const item of layers.instances) {
        if (item.confidence < 0.65 || item.mask.length !== staticMask.length) continue;
        for (let i = 0; i < staticMask.length; i++) {
            if (item.mask[i] && (item.kind === 'track' || layers.roadsideMask[i])) staticMask[i] = 1;
        }
    }
    const { padX, padY, resizedWidth, resizedHeight } = letterbox(frame.width, frame.height, 640);
    const maskIndex = (u: number, v: number) => {
        if (u < 0 || u >= 1 || v < 0 || v >= 1) return -1;
        const x = Math.floor((padX + u * resizedWidth) / 640 * segment.width);
        const y = Math.floor((padY + v * resizedHeight) / 640 * segment.height);
        return y * segment.width + x;
    };
    const surface = (u: number, v: number): SceneMemoryPoint['surface'] | null => {
        const i = maskIndex(u, v);
        if (!staticMask[i] || layers.trafficMask[i] || layers.obstacleMask[i]) return null;
        return layers.roadsideMask[i] ? 'roadside' : 'road';
    };
    const support = Uint8Array.from(pixels.gray, (_, i) => Number(Boolean(surface(
        (i % pixels.width + 0.5) / pixels.width, (Math.floor(i / pixels.width) + 0.5) / pixels.height))));
    const image: MotionImage = { ...pixels, support, lift(x, y) {
        const u = (x + 0.5) / pixels.width, v = (y + 0.5) / pixels.height;
        const kind = surface(u, v);
        const point = kind && depth(u, v, (su, sv) => surface(su, sv) === kind);
        return point && inRange(point) ? point : null;
    } };
    const points: SceneMemoryPoint[] = [];
    for (let y = 2; y < pixels.height; y += 4) for (let x = 2; x < pixels.width; x += 4) {
        const point = image.lift(x, y);
        if (point) points.push({ ...point, surface: surface((x + 0.5) / pixels.width, (y + 0.5) / pixels.height)!,
            lastSeenAt: frame.capturedAt, observations: 1 });
    }
    const camera = createCameraProjection(frame.calibration!);
    for (const [edge, kind] of [[scene.leftBoundary, 'left-edge'], [scene.rightBoundary, 'right-edge']] as const) {
        for (const point of edge) {
            if (point.estimated || !inRange(point)) continue;
            const pixel = camera.localToImage(point), i = pixel ? maskIndex(pixel.u, pixel.v) : -1;
            if (layers.trafficMask[i] || layers.obstacleMask[i]) continue;
            points.push({ ...point, surface: kind, lastSeenAt: frame.capturedAt, observations: 1 });
        }
    }
    return { image, points };
}

function voxelKey(p: SceneMemoryPoint) {
    return `${p.surface}:${Math.floor(p.x / VOXEL_M)}:${Math.floor(p.y / VOXEL_M)}:${Math.floor(p.z / VOXEL_M)}`;
}

/** Bounded map in the current camera's local frame. No absolute/world pose is accumulated. */
export class RollingSceneMemory {
    private previous: { capturedAt: number; calibrationKey: string; image: MotionImage } | null = null;
    private result: TrackSceneMemory | null = null;

    reset() { this.previous = null; this.result = null; }

    update(frame: TrackVisionFrame | null, scene: LocalTrackScene | null, pixels: SceneImage | null, now = Date.now()): TrackSceneMemory | null {
        if (!frame || !scene || !pixels || !frame.calibration || !Number.isFinite(frame.capturedAt)
            || frame.capturedAt > now || now - frame.capturedAt >= VISION_MAX_AGE_MS) {
            this.reset(); return null;
        }
        const calibrationKey = JSON.stringify([frame.width, frame.height, frame.calibration.heightM, frame.calibration.pitchDeg,
            frame.calibration.yawDeg, frame.calibration.horizontalFovDeg, frame.calibration.lateralOffsetM, frame.calibration.forwardOffsetM]);
        const previous = this.previous;
        // Reapplying unchanged calibration must not count the same capture twice.
        if (previous?.capturedAt === frame.capturedAt && previous.calibrationKey === calibrationKey) return this.result;
        const observation = observe(frame, scene, pixels);
        if (!observation) { this.reset(); return null; }
        let status: TrackSceneMemory['status'] = 'seeded', reason = 'Waiting for visual motion.', matchedFeatures = 0;
        let alignment: ReturnType<typeof estimateRigidMotion> = null;
        if (previous) {
            status = 'reset';
            const gap = frame.capturedAt - previous.capturedAt;
            if (previous.calibrationKey !== calibrationKey) reason = 'Camera calibration changed.';
            else if (gap <= 0 || gap > MAX_FRAME_GAP_MS) reason = 'Capture timing changed.';
            else {
                const matches = matchVisualFeatures(previous.image, observation.image);
                matchedFeatures = matches.length;
                alignment = estimateRigidMotion(matches);
                if (alignment) { status = 'aligned'; reason = 'Visual motion aligned.'; }
                else reason = 'Visual motion uncertain; memory restarted.';
            }
        }
        const voxels = new Map<string, SceneMemoryPoint>();
        const camera = createCameraProjection(frame.calibration);
        if (alignment) for (const point of this.result?.points ?? []) {
            if (frame.capturedAt - point.lastSeenAt >= SCENE_MEMORY_MAX_AGE_MS) continue;
            const moved = { ...point, ...transformPoint(point, alignment.motion) };
            if (!inRange(moved)) continue;
            const pixel = camera.localToImage(moved);
            const measured = pixel && pixel.u >= 0 && pixel.u < 1 && pixel.v >= 0 && pixel.v < 1
                ? observation.image.lift(Math.floor(pixel.u * pixels.width), Math.floor(pixel.v * pixels.height)) : null;
            // Remove surfaces contradicted by observed free space; occlusion alone is not deletion.
            const range = (p: GroundPoint) => Math.hypot(p.x - frame.calibration!.lateralOffsetM,
                p.y - frame.calibration!.forwardOffsetM, p.z - frame.calibration!.heightM);
            if (measured && range(measured) - range(moved) > Math.max(0.6, range(moved) * 0.04)) continue;
            const key = voxelKey(moved), existing = voxels.get(key);
            if (!existing || moved.lastSeenAt > existing.lastSeenAt) voxels.set(key, moved);
        }
        // Each voxel gets at most one observation per frame, irrespective of image sampling density.
        const current = new Map<string, SceneMemoryPoint>();
        for (const point of observation.points) {
            const key = voxelKey(point), existing = current.get(key);
            if (!existing) current.set(key, { ...point });
            else {
                const n = existing.observations;
                for (const axis of ['x', 'y', 'z'] as const) existing[axis] = (existing[axis] * n + point[axis]) / (n + 1);
                existing.observations++;
            }
        }
        current.forEach((point, key) => {
            const existing = voxels.get(key), weight = Math.min(3, existing?.observations ?? 0);
            voxels.set(key, { ...point, observations: Math.min(8, (existing?.observations ?? 0) + 1),
                x: ((existing?.x ?? 0) * weight + point.x) / (weight + 1),
                y: ((existing?.y ?? 0) * weight + point.y) / (weight + 1),
                z: ((existing?.z ?? 0) * weight + point.z) / (weight + 1) });
        });
        const points = Array.from(voxels.values()).sort((a, b) => b.lastSeenAt - a.lastSeenAt
            || Number(b.surface.endsWith('edge')) - Number(a.surface.endsWith('edge')) || a.y - b.y).slice(0, SCENE_MEMORY_MAX_POINTS);
        this.previous = { capturedAt: frame.capturedAt, calibrationKey, image: observation.image };
        this.result = { capturedAt: frame.capturedAt, points, status, reason, matchedFeatures,
            inliers: alignment?.inliers ?? 0, alignmentErrorM: alignment?.rmseM ?? null };
        return this.result;
    }
}
