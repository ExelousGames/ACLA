import {
    DriverExpertComparisonData,
    DriverExpertComparisonDiagnostic,
    normalizeDriverExpertComparisonData,
} from 'components/driver-expert-comparison';

export interface AnalysisResultPositionRange {
    start: number;
    end: number;
}

export interface AnalysisResultTimeGap {
    startMs?: number;
    endMs?: number;
    deltaMs?: number;
    [key: string]: unknown;
}

export interface AnalysisResultElement {
    id: string;
    labels: string[];
    title?: string;
    section?: string;
    normalizedPositionRange?: AnalysisResultPositionRange;
    timeGap?: AnalysisResultTimeGap;
    comparison?: DriverExpertComparisonData;
    comparisonDiagnostics?: DriverExpertComparisonDiagnostic[];
    metadata?: Record<string, unknown>;
}

export interface AnalysisResultsData {
    elements: AnalysisResultElement[];
}

export interface AnalysisResultElementInput extends Partial<Omit<AnalysisResultElement, 'id' | 'labels'>> {
    id?: unknown;
    labels?: unknown;
    [key: string]: unknown;
}

export interface AnalysisResultMutationResult {
    success: boolean;
    message: string;
    data?: {
        element?: AnalysisResultElement;
        id?: string;
        count: number;
    };
}

export type AnalysisResultLabelResolver = (labelId: string) => string | undefined;

const MISTAKE_PARENT_LABEL_NAMES = {
    MSP: 'Mistake (Practice)',
    MSR: 'Mistake (Racing)',
} as const;

export type MistakeParentLabelId = keyof typeof MISTAKE_PARENT_LABEL_NAMES;

export const getAnalysisResultMistakeParentLabels = (
    id: MistakeParentLabelId,
    resolveLabel?: AnalysisResultLabelResolver,
): ReadonlySet<string> => new Set([
    id,
    MISTAKE_PARENT_LABEL_NAMES[id],
    resolveLabel?.(id),
].filter((label): label is string => Boolean(label)));

let generatedIdSequence = 0;

export const createAnalysisResultElementId = (): string => (
    `analysis-result-${Date.now().toString(36)}-${(++generatedIdSequence).toString(36)}`
);

const isRecord = (value: unknown): value is Record<string, unknown> => (
    Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

const optionalText = (value: unknown): string | undefined => {
    if (typeof value !== 'string') return undefined;
    const text = value.trim();
    return text || undefined;
};

const finiteNumber = (value: unknown): number | undefined => {
    const parsed = typeof value === 'number' ? value : Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
};

const normalizeLabels = (value: unknown): string[] => {
    if (!Array.isArray(value)) return [];
    return value
        .filter((label): label is string => typeof label === 'string')
        .map((label) => label.trim())
        .filter(Boolean);
};

const normalizeComparisonDiagnostics = (
    value: unknown,
): DriverExpertComparisonDiagnostic[] | undefined => {
    if (!Array.isArray(value)) return undefined;
    const diagnostics = value.flatMap((entry): DriverExpertComparisonDiagnostic[] => {
        if (!isRecord(entry)) return [];
        const code = optionalText(entry.code);
        const message = optionalText(entry.message);
        if (!code || !message) return [];
        const details = isRecord(entry.details) ? { ...entry.details } : undefined;
        return [{ code, message, ...(details ? { details } : {}) }];
    });
    return diagnostics.length > 0 ? diagnostics : undefined;
};

const normalizePositionRange = (
    element: Record<string, unknown>,
): AnalysisResultPositionRange | undefined => {
    const source = isRecord(element.normalizedPositionRange)
        ? element.normalizedPositionRange
        : isRecord(element.normalized_position_range)
            ? element.normalized_position_range
            : null;
    const start = finiteNumber(source?.start ?? element.start_position ?? element.startPosition);
    const end = finiteNumber(source?.end ?? element.end_position ?? element.endPosition);
    return start === undefined || end === undefined ? undefined : { start, end };
};

const normalizeTimeGap = (value: unknown): AnalysisResultTimeGap | undefined => {
    if (!isRecord(value)) return undefined;
    const result: AnalysisResultTimeGap = { ...value };
    const startMs = finiteNumber(value.startMs ?? value.start_ms);
    const endMs = finiteNumber(value.endMs ?? value.end_ms);
    const deltaMs = finiteNumber(value.deltaMs ?? value.delta_ms);
    delete result.start_ms;
    delete result.end_ms;
    delete result.delta_ms;
    if (startMs !== undefined) result.startMs = startMs;
    if (endMs !== undefined) result.endMs = endMs;
    if (deltaMs !== undefined) result.deltaMs = deltaMs;
    return Object.keys(result).length > 0 ? result : undefined;
};

export const normalizeAnalysisResultElement = (
    input: unknown,
    options: { generateId?: () => string } = {},
): AnalysisResultElement | null => {
    if (!isRecord(input)) return null;

    const generateId = options.generateId ?? createAnalysisResultElementId;
    const id = optionalText(input.id) ?? generateId();
    const title = optionalText(input.title);
    const section = optionalText(input.section ?? input.track_section ?? input.trackSection);
    const normalizedPositionRange = normalizePositionRange(input);
    const timeGap = normalizeTimeGap(input.timeGap ?? input.time_gap);
    const comparison = normalizeDriverExpertComparisonData(input.comparison);
    const comparisonDiagnostics = normalizeComparisonDiagnostics(
        input.comparisonDiagnostics ?? input.comparison_diagnostics,
    );
    const metadata = isRecord(input.metadata) ? { ...input.metadata } : undefined;

    return {
        id,
        labels: normalizeLabels(input.labels),
        ...(title ? { title } : {}),
        ...(section ? { section } : {}),
        ...(normalizedPositionRange ? { normalizedPositionRange } : {}),
        ...(timeGap ? { timeGap } : {}),
        ...(comparison ? { comparison } : {}),
        ...(comparisonDiagnostics ? { comparisonDiagnostics } : {}),
        ...(metadata ? { metadata } : {}),
    };
};

export const normalizeAnalysisResultsData = (input: unknown): AnalysisResultsData => {
    const rawElements = Array.isArray(input)
        ? input
        : isRecord(input) && Array.isArray(input.elements)
            ? input.elements
            : isRecord(input) && Array.isArray(input.items)
                ? input.items
                : [];
    const usedIds = new Set<string>();
    const elements: AnalysisResultElement[] = [];

    rawElements.forEach((rawElement) => {
        const element = normalizeAnalysisResultElement(rawElement);
        if (!element) return;
        if (usedIds.has(element.id)) {
            element.id = createAnalysisResultElementId();
        }
        usedIds.add(element.id);
        elements.push(element);
    });

    return { elements };
};

export const appendAnalysisResultElement = (
    data: unknown,
    input: unknown,
): { result: AnalysisResultMutationResult; data: AnalysisResultsData } => {
    const current = normalizeAnalysisResultsData(data);
    if (!isRecord(input)) {
        return {
            data: current,
            result: { success: false, message: 'append_element requires an element object.' },
        };
    }

    const suppliedId = optionalText(input.id);
    if (suppliedId && current.elements.some((element) => element.id === suppliedId)) {
        return {
            data: current,
            result: { success: false, message: `Element '${suppliedId}' already exists.` },
        };
    }

    const element = normalizeAnalysisResultElement(input);
    if (!element) {
        return {
            data: current,
            result: { success: false, message: 'Unable to normalize the supplied element.' },
        };
    }
    const next = { elements: [...current.elements, element] };
    return {
        data: next,
        result: {
            success: true,
            message: `Appended element '${element.id}'.`,
            data: { element, count: next.elements.length },
        },
    };
};

export const updateAnalysisResultElement = (
    data: unknown,
    idValue: unknown,
    changesValue: unknown,
): { result: AnalysisResultMutationResult; data: AnalysisResultsData } => {
    const current = normalizeAnalysisResultsData(data);
    const id = optionalText(idValue);
    if (!id) {
        return {
            data: current,
            result: { success: false, message: 'update_element requires a non-empty id.' },
        };
    }
    const index = current.elements.findIndex((element) => element.id === id);
    if (index < 0) {
        return {
            data: current,
            result: { success: false, message: `Element '${id}' was not found.` },
        };
    }
    if (!isRecord(changesValue)) {
        return {
            data: current,
            result: { success: false, message: 'update_element requires a changes object.' },
        };
    }
    if ('id' in changesValue && changesValue.id !== id) {
        return {
            data: current,
            result: { success: false, message: 'Element IDs are immutable.' },
        };
    }

    const existing = current.elements[index];
    const merged: Record<string, unknown> = {
        ...existing,
        ...changesValue,
        id,
    };
    if (
        'normalized_position_range' in changesValue
        || 'start_position' in changesValue
        || 'end_position' in changesValue
        || 'startPosition' in changesValue
        || 'endPosition' in changesValue
    ) {
        delete merged.normalizedPositionRange;
    }
    if ('normalizedPositionRange' in changesValue || 'normalized_position_range' in changesValue) {
        delete merged.start_position;
        delete merged.end_position;
        delete merged.startPosition;
        delete merged.endPosition;
    }
    if (('track_section' in changesValue || 'trackSection' in changesValue) && !('section' in changesValue)) {
        delete merged.section;
    }
    if ('time_gap' in changesValue && !('timeGap' in changesValue)) {
        delete merged.timeGap;
    }
    const element = normalizeAnalysisResultElement(merged);
    if (!element) {
        return {
            data: current,
            result: { success: false, message: `Unable to normalize changes for element '${id}'.` },
        };
    }
    const elements = [...current.elements];
    elements[index] = element;
    return {
        data: { elements },
        result: {
            success: true,
            message: `Updated element '${id}'.`,
            data: { element, count: elements.length },
        },
    };
};

export const removeAnalysisResultElement = (
    data: unknown,
    idValue: unknown,
): { result: AnalysisResultMutationResult; data: AnalysisResultsData } => {
    const current = normalizeAnalysisResultsData(data);
    const id = optionalText(idValue);
    if (!id) {
        return {
            data: current,
            result: { success: false, message: 'remove_element requires a non-empty id.' },
        };
    }
    if (!current.elements.some((element) => element.id === id)) {
        return {
            data: current,
            result: { success: false, message: `Element '${id}' was not found.` },
        };
    }
    const elements = current.elements.filter((element) => element.id !== id);
    return {
        data: { elements },
        result: {
            success: true,
            message: `Removed element '${id}'.`,
            data: { id, count: elements.length },
        },
    };
};
