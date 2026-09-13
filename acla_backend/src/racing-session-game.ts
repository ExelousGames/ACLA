export const GAME_RECORDED_FROM_VALUES = ['acc', 'ac', 'iracing'] as const;

export type GameRecordedFrom = typeof GAME_RECORDED_FROM_VALUES[number];
