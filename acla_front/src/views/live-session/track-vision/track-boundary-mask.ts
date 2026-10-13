import type { createSegmentationLayers } from './segmentation-layers';

type Layers = NonNullable<ReturnType<typeof createSegmentationLayers>>;
type Span = { left: number; right: number };

/** Construct a road corridor from track seeds and the inner edges of nearby ground labels. */
export function createTrackBoundaryMask(layers: Layers, width: number, height: number) {
    const { trackMask, trafficMask, excludedMask, roadsideMask, obstacleMask, carInteriorMask } = layers;
    // Keep overlapping cockpit pixels until the final cutout, so holes in the road do not
    // masquerade as a narrow corridor followed by an abrupt widening.
    const mask = trackMask.map((active, i) => excludedMask[i] && !carInteriorMask[i] ? 0 : active);
    const rows: Span[][] = Array.from({ length: height }, () => []);
    // Scale with both the image and observed road width so distant runoff cannot widen the track.
    const reach = (span: Span) => Math.max(2, Math.floor(Math.min(width * 0.04, (span.right - span.left + 1) * 0.35)));
    const margin = (row: number, start: number, direction: number, distance: number) => {
        for (let step = 1; step <= distance; step++) {
            const column = start + direction * step;
            if (column < 0 || column >= width) return null;
            const i = row * width + column;
            if (trafficMask[i] || obstacleMask[i]) return null;
            if (roadsideMask[i]) return column - direction;
            if (excludedMask[i] || trackMask[i]) return null;
        }
        return null;
    };
    for (let row = 0; row < height; row++) {
        const offset = row * width;
        for (let column = 0; column < width; column++) {
            if (!trackMask[offset + column] || excludedMask[offset + column] || trafficMask[offset + column]) continue;
            const start = column;
            while (column + 1 < width && !excludedMask[offset + column + 1]
                && (trackMask[offset + column + 1] || trafficMask[offset + column + 1])) column++;
            if (column - start < 3) continue;
            const span = { left: start, right: column }, distance = reach(span);
            span.left = margin(row, start, -1, distance) ?? start;
            if (!trafficMask[offset + column]) span.right = margin(row, column, 1, distance) ?? column;
            mask.fill(1, offset + span.left, offset + span.right + 1);
            rows[row].push(span);
        }
    }
    // At most two absent rows, bracketed by observed track. Both roadside edges must be visible.
    // Use only original seed rows so recovered pixels cannot recursively grow the corridor.
    for (let row = 1; row < height - 1; row++) {
        if (rows[row].length) continue;
        let above = row - 1, below = row + 1;
        while (above >= 0 && row - above < 3 && !rows[above].length) above--;
        while (below < height && below - row < 3 && !rows[below].length) below++;
        if (above < 0 || below >= height || below - above > 3) continue;
        for (const near of rows[above]) {
            const far = rows[below].find((span) => Math.max(Math.abs(span.left - near.left), Math.abs(span.right - near.right)) <= reach(near));
            if (!far) continue;
            const t = (row - above) / (below - above);
            const expected = { left: Math.round(near.left + t * (far.left - near.left)),
                right: Math.round(near.right + t * (far.right - near.right)) };
            const distance = Math.min(reach(expected), Math.floor((expected.right - expected.left) / 3));
            const left = margin(row, expected.left + distance, -1, distance * 2 + 1);
            const right = margin(row, expected.right - distance, 1, distance * 2 + 1);
            if (left === null || right === null || right - left < 3) continue;
            const offset = row * width;
            if (excludedMask.subarray(offset + left, offset + right + 1).some(Boolean)
                || trafficMask[offset + left] || trafficMask[offset + right]) continue;
            mask.fill(1, offset + left, offset + right + 1);
        }
    }
    // Follow the corridor from the distance toward the car. A bottom-up scan can start
    // on a mislabelled bonnet and use its width to reject the real road above it.
    type Reference = Span & { growth: number };
    let previous: Reference[] = [], missingRows = 0;
    for (let row = 0; row < height; row++) {
        const offset = row * width, next: Reference[] = [];
        for (let column = 0; column < width; column++) {
            if (!mask[offset + column] || (trafficMask[offset + column] && !trackMask[offset + column])) continue;
            const span = { left: column, right: column };
            let visible = !carInteriorMask[offset + column] && !trafficMask[offset + column];
            while (column + 1 < width && (mask[offset + column + 1] || trafficMask[offset + column + 1])) {
                column++;
                // Traffic may bridge road observations, but its overhang is not road width.
                if (mask[offset + column] && (!trafficMask[offset + column] || trackMask[offset + column])) span.right = column;
                visible ||= Boolean(mask[offset + column] && !carInteriorMask[offset + column] && !trafficMask[offset + column]);
            }
            if (!visible || span.right - span.left < 3) continue;
            // When a car hides either edge, the remaining visible strip cannot establish
            // corridor width. Resume width checks once both road edges are observed again.
            if (trafficMask[offset + span.left] || trafficMask[offset + span.right]
                || (span.left > 0 && trafficMask[offset + span.left - 1])
                || (span.right + 1 < width && trafficMask[offset + span.right + 1])) continue;
            const overlaps = previous.filter((old) => old.left <= span.right && old.right >= span.left);
            const reference = overlaps.length ? { left: Math.min(...overlaps.map((old) => old.left)),
                right: Math.max(...overlaps.map((old) => old.right)),
                growth: Math.max(...overlaps.map((old) => old.growth)) } : null;
            const previousWidth = reference ? reference.right - reference.left + 1 : 0;
            // Allow perspective growth and pixel quantization, but do not learn from rejected
            // rows: a sustained false widening must not become a new boundary seed.
            const growth = Math.max(4, previousWidth * 0.2, (reference?.growth ?? 0) * 2 * (missingRows + 1) + 2);
            if (reference && span.right - span.left + 1 > previousWidth + growth) {
                mask.fill(0, offset + span.left, offset + column + 1);
                next.push(reference);
            } else next.push({ ...span, growth: reference
                ? Math.max(0, (span.right - span.left + 1 - previousWidth) / (missingRows + 1)) : width * 0.04 });
        }
        if (next.length) {
            previous = next;
            missingRows = 0;
        } else if (++missingRows > 2) previous = [];
    }
    // Subtract bodywork after all expansion and gap filling; no constructed road survives
    // inside the bonnet, dashboard or windshield/pillar mask.
    carInteriorMask.forEach((active, i) => { if (active) mask[i] = 0; });
    return mask;
}
