import { createSteeringCornerCamera } from './steering-corner-camera';
import { buildSteeringCornerChevrons } from './steering-corner-chevrons';

it('points arrows forward in lap order and makes distant arrows smaller in the vehicle view', () => {
    const line = [{ x: 0, y: 0 }, { x: 0, y: 80 }];
    const chevrons = buildSteeringCornerChevrons(line);
    const { project } = createSteeringCornerCamera(line, 480, 280);
    const widths = chevrons.map((points) => project(points[2])!.x - project(points[0])!.x);
    expect(chevrons.length).toBeGreaterThan(3);
    chevrons.forEach((points, index) => {
        expect(points[1].y).toBeGreaterThan(points[4].y);
        if (index) {
            expect(points[1].y).toBeGreaterThan(chevrons[index - 1][1].y);
            expect(widths[index]).toBeLessThan(widths[index - 1]);
        }
    });
});

it.each([-1, 1])('turns the arrows with a %s-direction bend', (direction) => {
    const line = [{ x: 0, y: 0 }, { x: 0, y: 10 }, { x: 50 * direction, y: 10 }];
    const chevrons = buildSteeringCornerChevrons(line);
    const far = chevrons[chevrons.length - 1];
    expect((far[1].x - far[4].x) * direction).toBeGreaterThan(0);
    expect(far[1].y).toBeCloseTo(10);
    expect(far[4].y).toBeCloseTo(10);
});

it('ignores repeated samples and omits arrows on stationary geometry', () => {
    const start = { x: 0, y: 0 }, end = { x: 0, y: 20 };
    expect(buildSteeringCornerChevrons([start, start])).toEqual([]);
    expect(buildSteeringCornerChevrons([start, start, end, end]))
        .toEqual(buildSteeringCornerChevrons([start, end]));
});

it('keeps short, looping and behind-camera arrow geometry finite and clipped', () => {
    for (const line of [
        [{ x: 0, y: 0 }, { x: 0, y: 0.1 }],
        [{ x: 0, y: 0 }, { x: 0, y: 10 }, { x: 2, y: -20 }],
    ]) {
        const { path } = createSteeringCornerCamera(line, 480, 280);
        const chevrons = buildSteeringCornerChevrons(line);
        expect(chevrons.length).toBeGreaterThan(0);
        expect(chevrons.length).toBeLessThanOrEqual(16);
        for (const points of chevrons) expect(path(points, true)).not.toMatch(/NaN|Infinity/);
    }
});
