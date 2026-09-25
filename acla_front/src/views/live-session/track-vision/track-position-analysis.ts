import type { CornerPosition, SegmentResult, TrackVisionAnalysis, TrackVisionFrame } from './track-vision-types';
import { PLAYER_TRACK_ROW } from './track-vision-types';
import { letterbox } from './yolo-segmentation';

export const VISION_CONFIDENCE = 0.65;
const TRACK_LABELS = ['track', 'road', 'asphalt', 'tarmac'];
const CAR_LABELS = ['car', 'cars', 'vehicle', 'vehicles', 'race car', 'racecar', 'opponent', 'opponent car'];
const BOUNDARY_LABELS = ['left_boundary', 'right_boundary'];

type Box = SegmentResult['instances'][number]['box'];
type TrackRow = { left: number; right: number; center: number; y: number };

/** Visual estimates using a calibrated car-center reference for the current camera. */
export function analyzeTrackPositions(vision: TrackVisionFrame | null): TrackVisionAnalysis {
    const segment = vision?.detections.segment;
    const playerCenterX = vision?.playerCenterX;
    if (playerCenterX === undefined || !Number.isFinite(playerCenterX) || playerCenterX <= 0 || playerCenterX >= 1) return {};
    if (!vision || segment?.task !== 'segment' || !Number.isFinite(vision.width) || !Number.isFinite(vision.height)
        || vision.width <= 0 || vision.height <= 0 || !Number.isInteger(segment.width) || !Number.isInteger(segment.height)
        || segment.width <= 0 || segment.height <= 0) return {};
    const labels = segment.classNames.map((label) => label.trim().toLowerCase().replace(/\s+/g, ' '));
    if (!labels.some((label) => TRACK_LABELS.includes(label)) || !labels.some((label) => CAR_LABELS.includes(label))) return {};
    const instances = segment.instances.filter((item) => Number.isFinite(item.confidence)
        && item.confidence >= VISION_CONFIDENCE && item.confidence <= 1);
    const tracks = instances.filter((item) => TRACK_LABELS.includes(labels[item.classId])
        && item.mask.length === segment.width * segment.height);
    if (!tracks.length) return {};

    // Boxes and masks share the square, letterboxed model input.
    const { padX, padY, resizedWidth, resizedHeight } = letterbox(vision.width, vision.height, 640);
    const toScreenBox = (box: Box): Box => [
        (box[0] * 640 - padX) / resizedWidth, (box[1] * 640 - padY) / resizedHeight,
        (box[2] * 640 - padX) / resizedWidth, (box[3] * 640 - padY) / resizedHeight,
    ];
    const maskAt = (mask: Uint8Array, x: number, y: number) => {
        const column = Math.floor((padX + x * resizedWidth) / 640 * segment.width);
        const row = Math.floor((padY + y * resizedHeight) / 640 * segment.height);
        return column >= 0 && column < segment.width && row >= 0 && row < segment.height
            && mask[row * segment.width + column] > 0;
    };
    const obstacles = instances.filter((item) => !TRACK_LABELS.includes(labels[item.classId])
        && !CAR_LABELS.includes(labels[item.classId]) && labels[item.classId] !== 'car pack'
        && !BOUNDARY_LABELS.includes(labels[item.classId]) && item.mask.length === segment.width * segment.height);
    const boundaries = BOUNDARY_LABELS.map((label) => instances.filter((item) => labels[item.classId] === label
        && item.mask.length === segment.width * segment.height));
    const onTrack = (x: number, y: number) => tracks.some((item) => maskAt(item.mask, x, y))
        && !obstacles.some((item) => maskAt(item.mask, x, y));
    const traffic = instances.filter((item) => CAR_LABELS.includes(labels[item.classId]) || labels[item.classId] === 'car pack')
        .map((item) => ({ box: toScreenBox(item.box), pack: labels[item.classId] === 'car pack' }))
        .filter(({ box: [left, top, right, bottom], pack }) =>
            [left, top, right, bottom].every(Number.isFinite) && left >= 0.05 && right <= 0.95
            && right - left >= 0.035 && right - left <= (pack ? 0.85 : 0.4)
            && top >= 0.2 && bottom >= 0.4 && bottom <= 0.85 && bottom - top >= 0.04);

    const columns = Math.floor(segment.width * resizedWidth / 640);
    const trackRow = (y: number, anchor: number): TrackRow | undefined => {
        const pixels = Array.from({ length: columns }, (_, index) => onTrack((index + 0.5) / columns, y));
        for (let column = 0; column < columns; column++) {
            if (!pixels[column]) continue;
            const start = column;
            let end = column;
            // Bridge car occlusion only between observed track pixels. Never bridge
            // grass/kerbs or use a car box as evidence for an unseen track edge.
            while (++column < columns) {
                const x = (column + 0.5) / columns;
                if (pixels[column]) end = column;
                else if (!traffic.some(({ box: [left, top, right, bottom] }) => x >= left && x <= right && y >= top && y <= bottom)
                    || obstacles.some((item) => maskAt(item.mask, x, y))) break;
            }
            if (start / columns > anchor || (end + 1) / columns < anchor) continue;
            // Boundary polylines describe edges, not holes in the track mask.
            // Use their inner pixels within this road region; fall back to the
            // track mask for a side whose boundary is absent at this depth.
            const edges = boundaries.map((items) => Array.from({ length: end - start + 1 }, (_, index) => start + index)
                .filter((index) => items.some((item) => maskAt(item.mask, (index + 0.5) / columns, y))));
            const leftColumn = edges[0].length ? Math.max(...edges[0]) : start;
            const rightColumn = edges[1].length ? Math.min(...edges[1]) : end;
            const left = edges[0].length ? (leftColumn + 0.5) / columns : start / columns;
            const right = edges[1].length ? (rightColumn + 0.5) / columns : (end + 1) / columns;
            if (left > anchor || right < anchor || right - left < 0.12 || leftColumn <= 0 || rightColumn >= columns - 1) return undefined;
            return { left, right, center: (left + right) / 2, y };
        }
        return undefined;
    };

    // Start at the marked vehicle centerline, then follow this road into the
    // distance. Row centers only trace the road; they never relocate the player.
    // Both boundaries must be visible to establish a position.
    const rows: TrackRow[] = [];
    for (const y of [PLAYER_TRACK_ROW, 0.7, 0.6, 0.5, 0.4]) {
        const row = trackRow(y, rows.length ? rows[rows.length - 1].center : playerCenterX);
        if (!row) return {};
        rows.push(row);
    }
    const [near, , middle, , far] = rows;
    const leftBend = far.left - 2 * middle.left + near.left;
    const rightBend = far.right - 2 * middle.right + near.right;
    // Curvature, not a left/right screen offset: an off-center straight still
    // has straight projected boundaries. Require agreement from both edges.
    const cornerDirection = leftBend < -0.025 && rightBend < -0.025 ? 'left'
        : leftBend > 0.025 && rightBend > 0.025 ? 'right' : undefined;
    if (!cornerDirection) return {};
    const direction = cornerDirection === 'left' ? -1 : 1;
    for (let index = 0; index < rows.length - 2; index++) {
        const bend = rows[index + 2].center - 2 * rows[index + 1].center + rows[index].center;
        if (bend * direction < -0.02) return {}; // Conflicting bends / chicane.
    }
    const position = (x: number, row: TrackRow): CornerPosition => {
        const across = (x - row.left) / (row.right - row.left);
        const fromInside = cornerDirection === 'left' ? across : 1 - across;
        return fromInside < 0.4 ? 'inside' : fromInside > 0.6 ? 'outside' : 'middle';
    };
    const result: TrackVisionAnalysis = { cornerDirection, playerPosition: position(playerCenterX, near), carAhead: 0 };
    // Compare the opponent with edges at its own depth, not with the player's
    // foreground edges. Off-center opponents still count when on this road.
    // A pack can establish traffic ahead, but cannot locate one opponent.
    // Prefer an individual car when its box reaches the same image depth.
    for (const { box: [left, , right, bottom], pack } of traffic.sort((a, b) => b.box[3] - a.box[3] || Number(a.pack) - Number(b.pack))) {
        const y = bottom + 0.025;
        if (y > near.y || y < far.y) continue;
        const index = rows.findIndex((row, i) => i < rows.length - 1 && row.y >= y && rows[i + 1].y <= y);
        if (index < 0) continue;
        const fraction = (rows[index].y - y) / (rows[index].y - rows[index + 1].y);
        const anchor = rows[index].center + fraction * (rows[index + 1].center - rows[index].center);
        const row = trackRow(y, anchor);
        const center = (left + right) / 2;
        if (!row || left < row.left || right > row.right
            || [-0.25, 0, 0.25].filter((offset) => onTrack(center + offset * (right - left), y)).length < 2) continue;
        return pack ? { ...result, carAhead: 1 } : { ...result, carAhead: 1, opponentPosition: position(center, row) };
    }
    return result;
}
