import { CornerDefinition } from './types';

// Normalized track position ranges (0.0–1.0) for each corner.
// Add entries here as new tracks are needed.
export const TRACK_CORNERS: Record<string, CornerDefinition[]> = {
    monza: [
        { name: 'T1 Prima Variante', from: 0.06, to: 0.10 },
        { name: 'T2 Seconda Variante', from: 0.18, to: 0.22 },
        { name: 'T3 Lesmo 1', from: 0.35, to: 0.39 },
        { name: 'T4 Lesmo 2', from: 0.41, to: 0.44 },
        { name: 'T5 Ascari', from: 0.55, to: 0.62 },
        { name: 'T6 Parabolica', from: 0.84, to: 0.92 },
    ],
    spa: [
        { name: 'T1 La Source', from: 0.02, to: 0.06 },
        { name: 'T2 Eau Rouge', from: 0.08, to: 0.13 },
        { name: 'T3 Raidillon', from: 0.13, to: 0.17 },
        { name: 'T4 Les Combes', from: 0.22, to: 0.27 },
        { name: 'T5 Malmedy', from: 0.29, to: 0.33 },
        { name: 'T6 Rivage', from: 0.36, to: 0.40 },
        { name: 'T7 Pouhon', from: 0.44, to: 0.50 },
        { name: 'T8 Campus', from: 0.54, to: 0.58 },
        { name: 'T9 Stavelot', from: 0.61, to: 0.65 },
        { name: 'T10 Blanchimont', from: 0.74, to: 0.78 },
        { name: 'T11 Bus Stop', from: 0.82, to: 0.88 },
    ],
    silverstone: [
        { name: 'T1 Abbey', from: 0.04, to: 0.08 },
        { name: 'T2 Farm', from: 0.10, to: 0.13 },
        { name: 'T3 Village', from: 0.18, to: 0.23 },
        { name: 'T4 The Loop', from: 0.25, to: 0.30 },
        { name: 'T5 Aintree', from: 0.33, to: 0.37 },
        { name: 'T6 Wellington', from: 0.40, to: 0.44 },
        { name: 'T7 Brooklands', from: 0.51, to: 0.56 },
        { name: 'T8 Luffield', from: 0.62, to: 0.68 },
        { name: 'T9 Woodcote', from: 0.72, to: 0.76 },
        { name: 'T10 Copse', from: 0.79, to: 0.83 },
        { name: 'T11 Maggotts', from: 0.85, to: 0.89 },
        { name: 'T12 Becketts', from: 0.89, to: 0.93 },
        { name: 'T13 Chapel', from: 0.93, to: 0.96 },
    ],
    nurburgring_gp: [
        { name: 'T1 Mercedes', from: 0.03, to: 0.07 },
        { name: 'T2 Einfahrt', from: 0.13, to: 0.17 },
        { name: 'T3 Ford', from: 0.23, to: 0.28 },
        { name: 'T4 Dunlop', from: 0.35, to: 0.39 },
        { name: 'T5 NGK', from: 0.47, to: 0.52 },
        { name: 'T6 Bit Kurve', from: 0.61, to: 0.66 },
        { name: 'T7 Veedol', from: 0.73, to: 0.78 },
        { name: 'T8 Coca Cola', from: 0.86, to: 0.91 },
    ],
    barcelona: [
        { name: 'T1', from: 0.03, to: 0.07 },
        { name: 'T2', from: 0.10, to: 0.14 },
        { name: 'T3', from: 0.18, to: 0.22 },
        { name: 'T4 Repsol', from: 0.28, to: 0.33 },
        { name: 'T5', from: 0.38, to: 0.42 },
        { name: 'T6', from: 0.47, to: 0.51 },
        { name: 'T7 New Holland', from: 0.55, to: 0.60 },
        { name: 'T8', from: 0.65, to: 0.69 },
        { name: 'T9', from: 0.73, to: 0.77 },
        { name: 'T10 La Caixa', from: 0.81, to: 0.86 },
        { name: 'T11', from: 0.89, to: 0.93 },
        { name: 'T12 Banc Sabadell', from: 0.94, to: 0.98 },
    ],
    brands_hatch: [
        { name: 'T1 Paddock Hill Bend', from: 0.04, to: 0.10 },
        { name: 'T2 Druids', from: 0.18, to: 0.24 },
        { name: 'T3 Graham Hill Bend', from: 0.30, to: 0.35 },
        { name: 'T4 Surtees', from: 0.50, to: 0.56 },
        { name: 'T5 McLaren', from: 0.64, to: 0.70 },
        { name: 'T6 Sheene', from: 0.74, to: 0.78 },
        { name: 'T7 Stirlings', from: 0.82, to: 0.86 },
        { name: 'T8 Clearways', from: 0.88, to: 0.92 },
        { name: 'T9 Clark Curve', from: 0.93, to: 0.97 },
    ],
};

// Normalize ACC track names to keys above.
const TRACK_NAME_MAP: Record<string, string> = {
    monza: 'monza',
    spa: 'spa',
    spa_francorchamps: 'spa',
    silverstone: 'silverstone',
    nurburgring: 'nurburgring_gp',
    nurburgring_gp: 'nurburgring_gp',
    barcelona: 'barcelona',
    catalunya: 'barcelona',
    brands_hatch: 'brands_hatch',
};

export function getCornersForTrack(trackName: string): CornerDefinition[] {
    const key = TRACK_NAME_MAP[trackName.toLowerCase().replace(/\s+/g, '_')] ?? trackName.toLowerCase();
    return TRACK_CORNERS[key] ?? [];
}

// Find which corner (if any) contains the given normalized position.
export function getCornerAtPosition(corners: CornerDefinition[], pos: number): CornerDefinition | null {
    return corners.find(c => pos >= c.from && pos <= c.to) ?? null;
}

// Find the next corner ahead of the given normalized position.
// Handles lap wrap-around (pos near 1.0 → first corner of next lap).
export function getNextCorner(corners: CornerDefinition[], currentPos: number): CornerDefinition | null {
    const ahead = corners.find(c => c.from > currentPos);
    return ahead ?? (corners.length > 0 ? corners[0] : null);
}
