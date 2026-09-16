export const GAME_RECORDED_FROM_VALUES = ['acc', 'ac', 'iracing', 'iracing_live', 'iracing_recorded'] as const;

export type GameRecordedFrom = typeof GAME_RECORDED_FROM_VALUES[number];
