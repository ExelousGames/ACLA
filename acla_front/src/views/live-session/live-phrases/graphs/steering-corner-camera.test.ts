import { createSteeringCornerCamera } from './steering-corner-camera';
import type { SteeringCornerPoint } from '../steering-corner-overlay';

const line = [{ x: 0, y: 0 }, { x: 0, y: 10 }, { x: 5, y: 30 }];
const camera = () => createSteeringCornerCamera(line, 480, 280);

it('anchors entry below the horizon and narrows the road with distance', () => {
    const { project, horizon } = camera();
    const entry = project(line[0])!, exit = project(line[2])!;
    expect(entry.x).toBe(240);
    expect(entry.y).toBeGreaterThan(210);
    expect(entry.y).toBeLessThan(280);
    expect(exit.y).toBeGreaterThan(horizon);
    expect(exit.y).toBeLessThan(entry.y);
    const nearWidth = project({ x: 5, y: 10 })!.x - project({ x: -5, y: 10 })!.x;
    const farWidth = project({ x: 5, y: 30 })!.x - project({ x: -5, y: 30 })!.x;
    expect(farWidth).toBeGreaterThan(0);
    expect(farWidth).toBeLessThan(nearWidth / 2);
});

it.each([0, Math.PI / 2, Math.PI, -Math.PI / 3])('keeps the view aligned with travel at heading %s', (angle) => {
    const transform = ({ x, y }: SteeringCornerPoint) => ({
        x: 500 + x * Math.cos(angle) - y * Math.sin(angle),
        y: -200 + x * Math.sin(angle) + y * Math.cos(angle),
    });
    const transformed = line.map(transform);
    const rotated = createSteeringCornerCamera(transformed, 480, 280);
    expect(rotated.path(transformed)).toBe(camera().path(line));
});

it('preserves left and right turns in the vehicle frame', () => {
    const { project } = camera();
    expect(project({ x: -5, y: 30 })!.x).toBeLessThan(240);
    expect(project({ x: 5, y: 30 })!.x).toBeGreaterThan(240);
});

it('clips paths behind the camera and starts a new segment when they reappear', () => {
    const { project, path } = camera();
    expect(project({ x: 0, y: -3 })).toBeUndefined();
    const clipped = path([{ x: 0, y: 0 }, { x: 2, y: -3 }, { x: -2, y: -3 }, { x: 0, y: 10 }]);
    expect(clipped.match(/M /g)).toHaveLength(2);
    expect(clipped).not.toMatch(/NaN|Infinity/);
    expect(path([{ x: 0, y: -4 }, { x: 1, y: -5 }])).toBe('');
});

it('clips the road surface at the near plane without projecting it above the horizon', () => {
    const { path, horizon } = camera();
    const road = path([{ x: -5, y: -10 }, { x: -5, y: 30 }, { x: 5, y: 30 }, { x: 5, y: -10 }], true);
    expect(road).toMatch(/^M .* Z$/);
    const coordinates = road.match(/-?\d+\.\d+/g)!.map(Number);
    expect(coordinates.every(Number.isFinite)).toBe(true);
    expect(coordinates.filter((_, index) => index % 2 === 1).every((y) => y > horizon)).toBe(true);
    expect(path([{ x: -5, y: -10 }, { x: 5, y: -10 }, { x: 0, y: -20 }], true)).toBe('');
});

it('skips duplicate entry samples and keeps stationary geometry finite', () => {
    const repeated = [line[0], ...line];
    expect(createSteeringCornerCamera(repeated, 480, 280).path(line)).toBe(camera().path(line));
    const stationary = [line[0], line[0]];
    expect(createSteeringCornerCamera(stationary, 480, 280).path(stationary)).not.toMatch(/NaN|Infinity/);
});
