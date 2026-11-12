import type { SteeringCornerPoint } from '../steering-corner-overlay';

/** Ground-plane arrows follow lap order and shrink naturally through the vehicle camera. */
export function buildSteeringCornerChevrons(line: SteeringCornerPoint[]): SteeringCornerPoint[][] {
    const distances = [0];
    for (let i = 1; i < line.length; i += 1) {
        distances.push(distances[i - 1] + Math.hypot(line[i].x - line[i - 1].x, line[i].y - line[i - 1].y));
    }
    const total = distances[distances.length - 1];
    if (total < 0.01) return [];
    const sample = (distance: number): SteeringCornerPoint => {
        const index = distances.findIndex((value) => value > distance);
        if (index < 0) return line[line.length - 1];
        const from = line[index - 1], to = line[index];
        const fraction = (distance - distances[index - 1]) / (distances[index] - distances[index - 1]);
        return { x: from.x + (to.x - from.x) * fraction, y: from.y + (to.y - from.y) * fraction };
    };
    const size = Math.min(1, total / 6);
    const chevrons: SteeringCornerPoint[][] = [];
    for (let distance = 1.5 * size; distance < total - 0.6 * size && chevrons.length < 16;
        distance += Math.max(3 * size, distance * 0.5)) {
        const center = sample(distance);
        const before = sample(distance - 0.6 * size), after = sample(distance + 0.6 * size);
        const length = Math.hypot(after.x - before.x, after.y - before.y);
        if (length < 0.001) continue;
        const forward = { x: (after.x - before.x) / length, y: (after.y - before.y) / length };
        chevrons.push([[-1, -0.25], [0, 0.6], [1, -0.25], [1, -0.65], [0, 0.15], [-1, -0.65]]
            .map(([side, ahead]) => ({
                x: center.x + size * (side * forward.y + ahead * forward.x),
                y: center.y + size * (-side * forward.x + ahead * forward.y),
            })));
    }
    return chevrons;
}
