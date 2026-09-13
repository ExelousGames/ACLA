export type Vec3 = {
    x: number;
    y: number;
    z: number;
};

export type CarPoint = {
    key: string;
    id: number | string | null;
    slot: number;
    position: Vec3;
};

export type TelemetryFrame = {
    time: number;
    cars: CarPoint[];
    playerKey: string | null;
    sourceIndex?: number;
};

export type VisibilitySample = {
    position: Vec3;
    visible: boolean;
};

export const getPlaybackFrameIndex = (frames: TelemetryFrame[], elapsed: number): number => {
    if (frames.length === 0) return -1;

    const targetTime = frames[0].time + elapsed;
    if (targetTime > frames[frames.length - 1].time) return -1;

    let low = 0;
    let high = frames.length - 1;

    while (low < high) {
        const mid = Math.floor((low + high) / 2);
        if (frames[mid].time >= targetTime) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }

    return low;
};

const DEFAULT_SAMPLE_RATE_HZ = 60;
const LAP_TIME_RESET_THRESHOLD_SECONDS = 1;
const MIN_FRAME_STEP_SECONDS = 1 / DEFAULT_SAMPLE_RATE_HZ;
const MAX_MULTI_CAR_FRAME_GAP_SECONDS = 1;
const COMPRESSED_MULTI_CAR_FRAME_STEP_SECONDS = 0.25;
const MAX_ABSOLUTE_TRACK_COORDINATE = 1000000;

const toFiniteNumber = (value: unknown): number | null => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
};

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

const getUsableIdSlots = (carIds: any[]): Set<number> => {
    const counts = new Map<string, number>();

    carIds.forEach((id) => {
        if (id === null || id === undefined) return;
        counts.set(String(id), (counts.get(String(id)) || 0) + 1);
    });

    const usableSlots = new Set<number>();
    carIds.forEach((id, slot) => {
        if (id === null || id === undefined) return;
        if (counts.get(String(id)) === 1) usableSlots.add(slot);
    });

    return usableSlots;
};

const getCarKey = (slot: number, carIds: any[], usableIdSlots: Set<number>): string => {
    const id = carIds[slot];
    return usableIdSlots.has(slot) ? `id:${String(id)}` : `slot:${slot}`;
};

const coordToVec3 = (coord: any): Vec3 | null => {
    if (!coord || typeof coord !== 'object') return null;

    const x = toFiniteNumber(coord.x);
    const y = toFiniteNumber(coord.y);
    const z = toFiniteNumber(coord.z) ?? 0;

    if (x === null || y === null) return null;
    if (x === 0 && y === 0 && z === 0) return null;
    if (
        Math.abs(x) > MAX_ABSOLUTE_TRACK_COORDINATE
        || Math.abs(y) > MAX_ABSOLUTE_TRACK_COORDINATE
        || Math.abs(z) > MAX_ABSOLUTE_TRACK_COORDINATE
    ) return null;

    return { x, y, z };
};

const getFrameTimeSeconds = (row: Record<string, any>, fallbackIndex: number): number => {
    const currentLapTime = toFiniteNumber(row.Graphics_current_time);
    if (currentLapTime !== null) return currentLapTime / 1000;

    const raw = toFiniteNumber(row.Physics_timestamp ?? row.timestamp);
    if (raw === null) return fallbackIndex / DEFAULT_SAMPLE_RATE_HZ;
    return raw > 100 ? raw / 1000 : raw;
};

const getPlayerKey = (
    coords: any[],
    carIds: any[],
    usableIdSlots: Set<number>,
    playerCarId: number | string | null
): string | null => {
    if (playerCarId !== null && carIds.length > 0) {
        const idMatch = carIds.findIndex((id) => String(id) === String(playerCarId));
        if (idMatch >= 0 && usableIdSlots.has(idMatch)) {
            return getCarKey(idMatch, carIds, usableIdSlots);
        }
    }

    const playerIndex = toFiniteNumber(playerCarId);
    if (playerIndex !== null && playerIndex >= 0 && playerIndex < coords.length) {
        return getCarKey(playerIndex, carIds, usableIdSlots);
    }

    return coords.length > 0 ? 'slot:0' : null;
};

export const parseTelemetryFrame = (row: Record<string, any>, index: number): TelemetryFrame | null => {
    const coords = parseMaybeArray(row.Graphics_car_coordinates);
    if (coords.length === 0) return null;

    const carIds = parseMaybeArray(row.Graphics_car_id);
    const playerCarId = row.Graphics_player_car_id ?? null;
    const usableIdSlots = getUsableIdSlots(carIds);
    const playerKey = getPlayerKey(coords, carIds, usableIdSlots, playerCarId);
    const cars: CarPoint[] = [];

    coords.forEach((coord, slot) => {
        const position = coordToVec3(coord);
        if (!position) return;

        const id = carIds[slot] ?? null;
        const key = getCarKey(slot, carIds, usableIdSlots);
        cars.push({ key, id, slot, position });
    });

    const drawablePlayerKey = cars.some((car) => car.key === playerKey)
        ? playerKey
        : cars[0]?.key ?? null;

    return cars.length > 0 ? { time: getFrameTimeSeconds(row, index), cars, playerKey: drawablePlayerKey, sourceIndex: index } : null;
};

export const normalizeTelemetryFrames = (frames: TelemetryFrame[]): TelemetryFrame[] => {
    let lapOffset = 0;
    let previousRawTime: number | null = null;
    let previousSessionTime = 0;

    return frames.map((frame, index) => {
        const rawTime = frame.time;

        if (previousRawTime !== null && rawTime + LAP_TIME_RESET_THRESHOLD_SECONDS < previousRawTime) {
            lapOffset = previousSessionTime;
        }

        const candidateSessionTime = rawTime + lapOffset;
        let sessionTime = candidateSessionTime;
        if (index > 0 && candidateSessionTime <= previousSessionTime) {
            sessionTime = previousSessionTime + MIN_FRAME_STEP_SECONDS;
        } else if (
            index > 0
            && frame.cars.length > 1
            && candidateSessionTime - previousSessionTime > MAX_MULTI_CAR_FRAME_GAP_SECONDS
        ) {
            sessionTime = previousSessionTime + COMPRESSED_MULTI_CAR_FRAME_STEP_SECONDS;
        }
        previousRawTime = rawTime;
        previousSessionTime = sessionTime;

        return index === 0 || sessionTime !== rawTime
            ? { ...frame, time: sessionTime }
            : frame;
    });
};

export const parseTelemetryFrames = (rows: Record<string, any>[]): TelemetryFrame[] => (
    normalizeTelemetryFrames(
        rows
            .map((row, index) => parseTelemetryFrame(row, index))
            .filter((frame): frame is TelemetryFrame => frame !== null)
    )
);

export const segmentVisiblePoints = (samples: VisibilitySample[]): Vec3[][] => {
    const segments: Vec3[][] = [];
    let current: Vec3[] = [];

    samples.forEach((sample) => {
        if (sample.visible) {
            current.push(sample.position);
            return;
        }

        if (current.length > 1) {
            segments.push(current);
        }
        current = [];
    });

    if (current.length > 1) {
        segments.push(current);
    }

    return segments;
};
