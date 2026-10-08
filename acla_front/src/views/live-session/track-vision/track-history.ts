import type { StandardTelemetrySample } from '../live-session-types';
import type { BirdsEyePoint, BirdsEyeScene } from './birds-eye-scene';
import { sameCalibration } from './camera-projection';
import type { TrackVisionFrame } from './track-vision-types';

const TELEMETRY_MAX_AGE_MS = 500;
const HISTORY_MAX_AGE_MS = 4000;
const SIDES = ['leftBoundary', 'rightBoundary', 'centerline'] as const;
type Edges = Pick<BirdsEyeScene, typeof SIDES[number]>;
export interface VisionMotion {
    receivedAt: number;
    generation: number;
    headingRad: number;
    forwardMps: number;
    rightMps: number;
}

const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

/** Only motion fields are read. Simulator XYZ positions and lap progress are never used. */
export function readVisionMotion(sample: StandardTelemetrySample, receivedAt: number, generation: number): VisionMotion | undefined {
    if (sample.Graphics_status !== 2 || !finite(sample.Physics_heading)) return undefined;
    const localForward = sample.Physics_local_velocity_z, localRight = sample.Physics_local_velocity_x;
    const speed = sample.Physics_speed_kmh, gear = sample.Physics_gear;
    const local = finite(localForward) && finite(localRight);
    // Standard gear 0 is reverse. Without signed velocity, a moving neutral car is ambiguous.
    if (!local && (!finite(speed) || speed < 0 || !finite(gear) || gear < 0 || (gear === 1 && speed > 1))) return undefined;
    const forwardMps = local ? localForward : speed! / 3.6 * (gear === 0 ? -1 : 1);
    const rightMps = local ? localRight : 0;
    if (Math.hypot(forwardMps, rightMps) > 150) return undefined;
    return { receivedAt, generation, headingRad: sample.Physics_heading, forwardMps, rightMps };
}

type Interval = [number, number];
const extent = (line: BirdsEyePoint[]): Interval => [Math.min(...line.map(({ y }) => y)), Math.max(...line.map(({ y }) => y))];

/** Clip to supported intervals, interpolating endpoints without connecting separate source sections. */
function clipLine(line: BirdsEyePoint[], [low, high]: Interval): BirdsEyePoint[][] {
    const lines: BirdsEyePoint[][] = [];
    let section: BirdsEyePoint[] = [];
    for (let i = 1; i < line.length; i++) {
        const a = line[i - 1], b = line[i], dy = b.y - a.y;
        const start = dy ? Math.max(0, Math.min((low - a.y) / dy, (high - a.y) / dy)) : 0;
        const end = dy ? Math.min(1, Math.max((low - a.y) / dy, (high - a.y) / dy)) : 1;
        if (start >= end || (!dy && (a.y < low || a.y > high))) {
            if (section.length > 1) lines.push(section);
            section = [];
            continue;
        }
        const at = (t: number): BirdsEyePoint => ({ ...a,
            x: a.x + (b.x - a.x) * t, y: a.y + dy * t, z: a.z + (b.z - a.z) * t });
        const first = at(start), last = at(end), previous = section[section.length - 1];
        if (previous && Math.hypot(previous.x - first.x, previous.y - first.y) > 0.001) {
            if (section.length > 1) lines.push(section);
            section = [];
        }
        if (!section.length) section.push(first);
        section.push(last);
    }
    if (section.length > 1) lines.push(section);
    return lines;
}

/** Short-lived static road memory, always expressed relative to the current captured car pose. */
export class TrackHistory {
    private previous?: TrackVisionFrame;
    private frames: Array<{ observedAt: number; edges: Edges }> = [];
    private result: BirdsEyeScene | null = null;

    reset() { this.previous = undefined; this.frames = []; this.result = null; }

    update(frame: TrackVisionFrame | null, current: BirdsEyeScene | null): BirdsEyeScene | null {
        const motion = frame?.motion;
        if (!frame || !current || !motion || frame.capturedAt < motion.receivedAt
            || frame.capturedAt - motion.receivedAt > TELEMETRY_MAX_AGE_MS) {
            this.reset();
            return current;
        }
        const previous = this.previous;
        if (previous && (!sameCalibration(previous.calibration, frame.calibration)
            || previous.filterConfidence !== frame.filterConfidence || previous.motion?.generation !== motion.generation)) this.reset();
        if (this.previous && frame.capturedAt === this.previous.capturedAt) return this.result;
        if (this.previous) {
            const dt = (frame.capturedAt - this.previous.capturedAt) / 1000;
            const before = this.previous.motion!;
            const turn = Math.atan2(Math.sin(motion.headingRad - before.headingRad), Math.cos(motion.headingRad - before.headingRad));
            if (dt <= 0 || dt > 1 || Math.abs(turn) > Math.max(0.15, dt * 2)) this.reset();
            else {
                // Integrate mean body velocity along the heading arc, then rotate old road into the new car frame.
                const half = turn / 2, arc = Math.abs(half) < 0.00001 ? 1 : Math.sin(half) / half;
                const forward = (before.forwardMps + motion.forwardMps) / 2 * dt * arc;
                const right = (before.rightMps + motion.rightMps) / 2 * dt * arc;
                const dx = Math.cos(half) * right + Math.sin(half) * forward;
                const dy = -Math.sin(half) * right + Math.cos(half) * forward;
                const c = Math.cos(turn), s = Math.sin(turn);
                this.frames = this.frames.filter(({ observedAt }) => frame.capturedAt - observedAt < HISTORY_MAX_AGE_MS)
                    .map(({ observedAt, edges }) => ({ observedAt, edges: Object.fromEntries(SIDES.map((side) => [side,
                        edges[side].map((line) => line.map((point) => ({
                            x: c * (point.x - dx) - s * (point.y - dy),
                            y: s * (point.x - dx) + c * (point.y - dy), z: point.z, observedAt,
                        }))),
                    ])) as Edges }));
            }
        }
        const result: BirdsEyeScene = { ...current };
        for (const side of SIDES) {
            // Memory fills only the nearby blind region. Current observations always win.
            let uncovered: Interval[] = [[-5, Math.min(20, ...current[side].flat().map(({ y }) => y))]];
            const remembered: BirdsEyePoint[][] = [];
            for (const { edges } of this.frames) {
                for (const line of edges[side]) {
                    const sections = uncovered.flatMap((interval) => clipLine(line, interval));
                    remembered.push(...sections);
                    for (const section of sections) {
                        const [low, high] = extent(section);
                        uncovered = uncovered.flatMap(([a, b]) => high <= a || low >= b ? [[a, b] as Interval]
                            : [...(a < low ? [[a, low] as Interval] : []), ...(high < b ? [[high, b] as Interval] : [])]);
                    }
                }
            }
            result[side] = [...current[side], ...remembered];
        }
        // Never feed completed/estimated edges or moving traffic back into observation history.
        this.frames.unshift({ observedAt: frame.capturedAt, edges: {
            leftBoundary: current.leftBoundary, rightBoundary: current.rightBoundary, centerline: current.centerline,
        } });
        this.frames = this.frames.slice(0, 24);
        this.previous = frame;
        this.result = result;
        return result;
    }
}

