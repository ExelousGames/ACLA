import { getCornerAtPosition, getCornersForTrack, getNextCorner } from 'views/lap-analysis/session-intelligence/track-corners';

const DEFAULT_SAMPLE_RATE_HZ = 60;
const WINDOW_SECONDS = 5;
const NEARBY_RANGE_M = 100;
const LATERAL_TOLERANCE_M = 15;
const MIN_CLOSING_SPEED_MPS = 2;
const ACTION_WINDOW_SECONDS = 8;
const MIN_LONGITUDINAL_GAP_M = 5;
const MIN_HEADING_SPEED_MPS = 0.5;

type Vec2 = { x: number; y: number };

interface CarSample {
    key: string;
    id: number | string | null;
    slot: number;
    position: Vec2;
    time: number;
}

interface CarMotion {
    key: string;
    id: number | string | null;
    slot: number;
    position: Vec2;
    velocity: Vec2;
    speed: number;
}

interface ParsedRow {
    cars: CarSample[];
    playerKey: string | null;
    time: number;
}

export interface TacticalToolStatus {
    event: 'attack_window' | 'defense_threat';
    mode: 'attack' | 'defense';
    opponent_id: number | string | null;
    opponent_slot: number;
    distance_m: number;
    longitudinal_gap_m: number;
    lateral_offset_m: number;
    closing_speed_mps: number;
    time_to_overlap_seconds: number;
    projected_track_position: number | null;
    projected_section: string | null;
    next_corner: { name: string; trackPosition: number; distanceAhead: number } | null;
    confidence: number;
}

export type TacticalDetectionResult =
    | { status: 'insufficient_data'; reason: string }
    | { status: 'neutral'; reason: string; opponent_count: number }
    | ({ status: 'actionable' } & TacticalToolStatus);

const toFiniteNumber = (value: unknown): number | null => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
};

const vectorLength = (v: Vec2): number => Math.hypot(v.x, v.y);

const normalize = (v: Vec2): Vec2 | null => {
    const length = vectorLength(v);
    return length > 1e-6 ? { x: v.x / length, y: v.y / length } : null;
};

const dot = (a: Vec2, b: Vec2): number => a.x * b.x + a.y * b.y;

const subtract = (a: Vec2, b: Vec2): Vec2 => ({ x: a.x - b.x, y: a.y - b.y });

const parseMaybeArray = (value: unknown): any[] => {
    if (Array.isArray(value)) return value;
    if (typeof value !== 'string') return [];
    try {
        const parsed = JSON.parse(value);
        return Array.isArray(parsed) ? parsed : [];
    } catch {
        return [];
    }
};

const coordToVec = (coord: any): Vec2 | null => {
    if (!coord || typeof coord !== 'object') return null;
    const x = toFiniteNumber(coord.x);
    const y = toFiniteNumber(coord.y);
    if (x === null || y === null) return null;
    if (x === 0 && y === 0) return null;
    return { x, y };
};

const getRowTimeSeconds = (row: Record<string, any>, fallbackIndex: number): number => {
    const raw = toFiniteNumber(row.Graphics_current_time ?? row.Physics_timestamp ?? row.timestamp);
    if (raw === null) return fallbackIndex / DEFAULT_SAMPLE_RATE_HZ;
    return raw > 100 ? raw / 1000 : raw;
};

const getPlayerKey = (
    coords: any[],
    carIds: any[],
    playerCarId: number | string | null,
): string | null => {
    if (playerCarId !== null && carIds.length > 0) {
        const match = carIds.findIndex((id) => String(id) === String(playerCarId));
        if (match >= 0) return `id:${String(playerCarId)}`;
    }

    const playerIndex = toFiniteNumber(playerCarId);
    if (playerIndex !== null && playerIndex >= 0 && playerIndex < coords.length) {
        const id = carIds[playerIndex];
        return id !== undefined && id !== null ? `id:${String(id)}` : `slot:${playerIndex}`;
    }

    return null;
};

const parseRow = (row: Record<string, any>, index: number): ParsedRow | null => {
    const coords = parseMaybeArray(row.Graphics_car_coordinates);
    if (coords.length === 0) return null;

    const carIds = parseMaybeArray(row.Graphics_car_id);
    const playerCarId = row.Graphics_player_car_id ?? null;
    const playerKey = getPlayerKey(coords, carIds, playerCarId);
    const time = getRowTimeSeconds(row, index);
    const cars: CarSample[] = [];

    coords.forEach((coord, slot) => {
        const position = coordToVec(coord);
        if (!position) return;
        const id = carIds[slot] ?? null;
        const key = id !== null && id !== undefined ? `id:${String(id)}` : `slot:${slot}`;
        cars.push({ key, id, slot, position, time });
    });

    return cars.length > 0 ? { cars, playerKey, time } : null;
};

const getMotion = (samples: CarSample[]): CarMotion | null => {
    if (samples.length < 2) return null;
    const first = samples[0];
    const last = samples[samples.length - 1];
    let dt = last.time - first.time;
    if (dt <= 0) dt = (samples.length - 1) / DEFAULT_SAMPLE_RATE_HZ;
    if (dt <= 0) return null;
    const velocity = {
        x: (last.position.x - first.position.x) / dt,
        y: (last.position.y - first.position.y) / dt,
    };
    return {
        key: last.key,
        id: last.id,
        slot: last.slot,
        position: last.position,
        velocity,
        speed: vectorLength(velocity),
    };
};

const getHeadingFallback = (row: Record<string, any>): Vec2 | null => {
    const raw = toFiniteNumber(row.Physics_heading);
    if (raw === null) return null;
    const radians = Math.abs(raw) > Math.PI * 2 ? raw * Math.PI / 180 : raw;
    return normalize({ x: Math.cos(radians), y: Math.sin(radians) });
};

const circularDelta = (start: number, end: number): number => {
    let delta = end - start;
    if (delta < -0.5) delta += 1;
    if (delta > 0.5) delta -= 1;
    return delta;
};

const projectTrackPosition = (
    rows: Record<string, any>[],
    parsedRows: ParsedRow[],
    secondsAhead: number,
): number | null => {
    const positions = rows
        .map((row) => toFiniteNumber(row.Graphics_normalized_car_position))
        .filter((value): value is number => value !== null);
    if (positions.length === 0) return null;

    const current = positions[positions.length - 1] % 1;
    if (positions.length < 2 || parsedRows.length < 2) return current;

    const firstPos = positions[0];
    const lastPos = positions[positions.length - 1];
    let dt = parsedRows[parsedRows.length - 1].time - parsedRows[0].time;
    if (dt <= 0) dt = (parsedRows.length - 1) / DEFAULT_SAMPLE_RATE_HZ;
    if (dt <= 0) return current;

    const rate = Math.max(0, circularDelta(firstPos, lastPos) / dt);
    return (current + rate * Math.max(0, secondsAhead)) % 1;
};

const getSectionContext = (
    rows: Record<string, any>[],
    parsedRows: ParsedRow[],
    timeToOverlap: number,
): Pick<TacticalToolStatus, 'projected_track_position' | 'projected_section' | 'next_corner'> => {
    const projected = projectTrackPosition(rows, parsedRows, timeToOverlap);
    const track = String(rows[rows.length - 1]?.Static_track ?? '');
    const corners = getCornersForTrack(track);
    if (projected === null || corners.length === 0) {
        return { projected_track_position: projected, projected_section: null, next_corner: null };
    }

    const corner = getCornerAtPosition(corners, projected);
    const next = getNextCorner(corners, projected);
    const distanceAhead = next
        ? next.from > projected ? next.from - projected : 1 - projected + next.from
        : 0;

    return {
        projected_track_position: projected,
        projected_section: corner?.name ?? next?.name ?? null,
        next_corner: next ? {
            name: next.name,
            trackPosition: next.from,
            distanceAhead,
        } : null,
    };
};

const round = (value: number, places = 2): number => {
    const scale = 10 ** places;
    return Math.round(value * scale) / scale;
};

export const detectOvertakeTacticalState = (
    telemetryRows: Record<string, any>[],
): TacticalDetectionResult => {
    const rows = telemetryRows.slice(-WINDOW_SECONDS * DEFAULT_SAMPLE_RATE_HZ);
    const parsedRows = rows
        .map(parseRow)
        .filter((row): row is ParsedRow => row !== null);

    if (parsedRows.length < 2) {
        return { status: 'insufficient_data', reason: 'coordinate_history_missing' };
    }

    const latest = parsedRows[parsedRows.length - 1];
    const playerKey = latest.playerKey;
    if (!playerKey) {
        return { status: 'insufficient_data', reason: 'player_car_id_missing' };
    }

    const samplesByCar = new Map<string, CarSample[]>();
    for (const row of parsedRows) {
        for (const car of row.cars) {
            if (!samplesByCar.has(car.key)) samplesByCar.set(car.key, []);
            samplesByCar.get(car.key)?.push(car);
        }
    }

    const playerMotion = getMotion(samplesByCar.get(playerKey) || []);
    if (!playerMotion) {
        return { status: 'insufficient_data', reason: 'player_motion_missing' };
    }

    const latestRawRow = rows[rows.length - 1] || {};
    const heading = playerMotion.speed >= MIN_HEADING_SPEED_MPS
        ? normalize(playerMotion.velocity)
        : getHeadingFallback(latestRawRow);
    if (!heading) {
        return { status: 'insufficient_data', reason: 'player_heading_unstable' };
    }

    const left = { x: -heading.y, y: heading.x };
    const candidates: TacticalToolStatus[] = [];

    for (const [key, samples] of Array.from(samplesByCar.entries())) {
        if (key === playerKey) continue;
        const motion = getMotion(samples);
        if (!motion) continue;

        const relativePosition = subtract(motion.position, playerMotion.position);
        const relativeVelocity = subtract(motion.velocity, playerMotion.velocity);
        const distance = vectorLength(relativePosition);
        if (distance > NEARBY_RANGE_M) continue;

        const longitudinalGap = dot(relativePosition, heading);
        const lateralOffset = dot(relativePosition, left);
        if (Math.abs(lateralOffset) > LATERAL_TOLERANCE_M) continue;
        if (Math.abs(longitudinalGap) < MIN_LONGITUDINAL_GAP_M) continue;

        const longitudinalRate = dot(relativeVelocity, heading);
        const isAttack = longitudinalGap > 0 && longitudinalRate < -MIN_CLOSING_SPEED_MPS;
        const isDefense = longitudinalGap < 0 && longitudinalRate > MIN_CLOSING_SPEED_MPS;
        if (!isAttack && !isDefense) continue;

        const closingSpeed = Math.abs(longitudinalRate);
        const timeToOverlap = Math.abs(longitudinalGap) / closingSpeed;
        if (timeToOverlap > ACTION_WINDOW_SECONDS) continue;

        const sectionContext = getSectionContext(rows, parsedRows, timeToOverlap);
        candidates.push({
            event: isAttack ? 'attack_window' : 'defense_threat',
            mode: isAttack ? 'attack' : 'defense',
            opponent_id: motion.id,
            opponent_slot: motion.slot,
            distance_m: round(distance),
            longitudinal_gap_m: round(longitudinalGap),
            lateral_offset_m: round(lateralOffset),
            closing_speed_mps: round(closingSpeed),
            time_to_overlap_seconds: round(timeToOverlap, 1),
            confidence: round(Math.min(1, closingSpeed / 10), 2),
            ...sectionContext,
        });
    }

    if (candidates.length === 0) {
        return { status: 'neutral', reason: 'no_actionable_relative_motion', opponent_count: samplesByCar.size - 1 };
    }

    candidates.sort((a, b) => a.time_to_overlap_seconds - b.time_to_overlap_seconds);
    return { status: 'actionable', ...candidates[0] };
};
