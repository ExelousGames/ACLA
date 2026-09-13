export const isOverlayRecord = (value: unknown): value is Record<string, unknown> => (
    Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

export const isOverlayNonEmptyString = (value: unknown): value is string => (
    typeof value === 'string' && Boolean(value.trim())
);

export const isOverlayFiniteOrNull = (value: unknown): boolean => value === null
    || (typeof value === 'number' && Number.isFinite(value));
