const FAMILY_PATTERNS: Array<[string, RegExp]> = [
    ['speed', /speed|velocity/],
    ['brake', /brake|abs/],
    ['throttle', /throttle|\bgas\b/],
    ['steering', /steer/],
    ['tire', /tire|tyre|wheel_pressure/],
    ['fuel', /fuel/],
    ['engine', /rpm|engine/],
    ['acceleration', /acceleration|g_force/],
    ['position', /position|track|lap/],
];

const normalizeTokens = (value: unknown): string[] => {
    if (Array.isArray(value)) return value.flatMap(normalizeTokens);
    if (typeof value !== 'string') return [];
    const trimmed = value.trim();
    if (!trimmed) return [];
    try {
        const parsed = JSON.parse(trimmed);
        if (Array.isArray(parsed)) return normalizeTokens(parsed);
    } catch {
        // Accept comma-delimited and ordinary field-name strings.
    }
    return trimmed
        .replace(/^\[|\]$/g, '')
        .split(',')
        .map((item) => item.replace(/^['"]|['"]$/g, '').trim())
        .filter(Boolean);
};

const GENERIC_DATA_KEYS = new Set([
    'data',
    'fields',
    'metrics',
    'rows',
    'samples',
    'series',
    'telemetry',
    'values',
]);

const collectDataTokens = (value: unknown): string[] => {
    if (Array.isArray(value)) return value.flatMap(collectDataTokens);
    if (typeof value === 'string') return normalizeTokens(value);
    if (!value || typeof value !== 'object') return [];
    return Object.entries(value as Record<string, unknown>).flatMap(([key, item]) => [
        ...(GENERIC_DATA_KEYS.has(key.toLowerCase()) ? [] : [key]),
        ...collectDataTokens(item),
    ]);
};

export const deriveTelemetryMetricFamilies = (args: Record<string, any> = {}): string[] => {
    const tokens = [
        ...normalizeTokens(args.metric_family ?? args.metricFamily),
        ...normalizeTokens(args.metric ?? args.metrics),
        ...normalizeTokens(args.fields),
        ...normalizeTokens(args.data_types ?? args.dataTypes),
        ...collectDataTokens(args.data),
    ];
    const families = new Set<string>();
    tokens.forEach((token) => {
        const normalized = token.toLowerCase().replace(/[^a-z0-9]+/g, '_');
        const match = FAMILY_PATTERNS.find(([, pattern]) => pattern.test(normalized));
        if (match) {
            families.add(match[0]);
        } else if (normalized) {
            families.add(normalized.replace(/^(physics|graphics|static)_/, '').split('_')[0]);
        }
    });
    return Array.from(families).filter(Boolean).sort();
};

export const getTelemetryComponentName = (families: readonly string[]): string => (
    `telemetry:${Array.from(new Set(families.map((family) => family.trim().toLowerCase()).filter(Boolean))).sort().join('+') || 'general'}`
);

export const getSingletonVisualizationComponentName = (type: string): string => (
    `visualization:${type.trim().toLowerCase()}`
);

export const getVisualizationComponentName = (
    type: string,
    args: Record<string, any> = {},
): string => (
    type === 'telemetry-overview'
        ? getTelemetryComponentName(deriveTelemetryMetricFamilies(args))
        : getSingletonVisualizationComponentName(type)
);

export const isTelemetryComponentName = (name: string): boolean => name.startsWith('telemetry:');
