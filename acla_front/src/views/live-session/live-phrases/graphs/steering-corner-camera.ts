import type { SteeringCornerPoint } from '../steering-corner-overlay';

const NEAR_PLANE = 0.5;
const EYE_HEIGHT = 1.6;
const ENTRY_DISTANCE = 3;

/** A vehicle-height camera looking along the reference line at corner entry. */
export function createSteeringCornerCamera(line: SteeringCornerPoint[], width: number, height: number) {
    const origin = line[0];
    const ahead = line.find((point) => Math.hypot(point.x - origin.x, point.y - origin.y) > 0.001);
    const dx = ahead ? ahead.x - origin.x : 0, dy = ahead ? ahead.y - origin.y : 1;
    const length = Math.hypot(dx, dy);
    const forward = { x: dx / length, y: dy / length };
    const horizon = height * 0.25, focalLength = width * 0.625;
    const toCamera = (point: SteeringCornerPoint) => {
        const x = point.x - origin.x, y = point.y - origin.y;
        return {
            x: x * forward.y - y * forward.x,
            depth: ENTRY_DISTANCE + x * forward.x + y * forward.y,
        };
    };
    type CameraPoint = ReturnType<typeof toCamera>;
    const project = (point: CameraPoint) => ({
        x: width / 2 + focalLength * point.x / point.depth,
        y: horizon + focalLength * EYE_HEIGHT / point.depth,
    });
    const intersection = (from: CameraPoint, to: CameraPoint): CameraPoint => ({
        x: from.x + (to.x - from.x) * (NEAR_PLANE - from.depth) / (to.depth - from.depth),
        depth: NEAR_PLANE,
    });
    const command = (verb: 'M' | 'L', point: CameraPoint) => {
        const { x, y } = project(point);
        return `${verb} ${x.toFixed(3)} ${y.toFixed(3)}`;
    };
    return {
        horizon,
        project: (point: SteeringCornerPoint) => {
            const cameraPoint = toCamera(point);
            return cameraPoint.depth >= NEAR_PLANE ? project(cameraPoint) : undefined;
        },
        path: (points: SteeringCornerPoint[], closed = false): string => {
            const cameraPoints = points.map(toCamera);
            if (closed) {
                // Clip the road surface before projection to prevent it folding over the horizon.
                const clipped: CameraPoint[] = [];
                cameraPoints.forEach((point, index) => {
                    const next = cameraPoints[(index + 1) % cameraPoints.length];
                    if (point.depth >= NEAR_PLANE) clipped.push(point);
                    if ((point.depth >= NEAR_PLANE) !== (next.depth >= NEAR_PLANE)) {
                        clipped.push(intersection(point, next));
                    }
                });
                return clipped.length ? `${clipped.map((point, index) => command(index ? 'L' : 'M', point)).join(' ')} Z` : '';
            }
            const commands: string[] = [];
            cameraPoints.forEach((point, index) => {
                const previous = cameraPoints[index - 1];
                if (point.depth >= NEAR_PLANE) {
                    if (!previous || previous.depth < NEAR_PLANE) {
                        commands.push(command('M', previous ? intersection(previous, point) : point));
                    }
                    if (previous) commands.push(command('L', point));
                } else if (previous && previous.depth >= NEAR_PLANE) {
                    commands.push(command('L', intersection(previous, point)));
                }
            });
            return commands.join(' ');
        },
    };
}
