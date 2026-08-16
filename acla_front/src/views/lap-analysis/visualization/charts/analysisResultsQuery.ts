import jsonata from 'jsonata';
import {
    getAnalysisResultMistakeParentLabels,
    type AnalysisResultLabelResolver,
    type MistakeParentLabelId,
} from './analysisResultsModel';

export type JsonPrimitive = null | boolean | number | string;
export type JsonObject = { [key: string]: JsonValue };
export type JsonValue = JsonPrimitive | JsonObject | JsonValue[];

export type QueryAnalysisResultInput = {
    query: string;
};

export type QueryAnalysisResultOutput = {
    status: 'ready';
    data: JsonValue;
};

export type ApplyAnalysisResultQueryInput = {
    query: string;
    page_number?: number;
};

export type ApplyAnalysisResultQueryOutput = {
    status: 'ready';
    data: number;
    applied_query: string;
    applied_page_id: string | null;
    applied_page_number: number;
    requested_page_number: number | null;
    used_most_recent_fallback: boolean;
};

export type ActivePageQueryElement = {
    id: string;
    labels: string[];
    title?: string;
    section?: string;
    normalizedPositionRange?: {
        start: number;
        end: number;
    };
    timeGap?: JsonObject;
    comparison?: JsonObject;
    metadata?: JsonObject;
};

export type ActivePageQueryInput = {
    elements: ActivePageQueryElement[];
};

export type OverallTrendQueryPage = {
    id: string;
    createdAt: number;
    sourceIndex: number;
    baseline: {
        lap: number;
        lapTimeMs: number | null;
        track: string;
        car: string;
    };
    elements: ActivePageQueryElement[];
};

export type OverallTrendQueryInput = {
    pages: OverallTrendQueryPage[];
};

export interface OverallTrendTaxonomyValue {
    id: string;
    fallbackName: string;
    resolvedName?: string;
}

export interface OverallTrendQueryTaxonomy {
    parent: OverallTrendTaxonomyValue;
    categories: readonly OverallTrendTaxonomyValue[];
}

export interface OverallTrendQueryCategory {
    id: string;
    label: string;
    aliases: readonly string[];
}

export interface OverallTrendQueryDefinition {
    expression: string;
    parentAliases: readonly string[];
    categories: readonly OverallTrendQueryCategory[];
}

export interface OverallTrendQueryCategoryCount {
    id: string;
    label: string;
    count: number;
}

export interface OverallTrendQueryLapResult {
    pageId: string;
    label: string;
    lap: number;
    lapTimeMs: number | null;
    totalCount: number;
    categoryCounts: OverallTrendQueryCategoryCount[];
}

export interface OverallTrendQueryCategoryResult {
    id: string;
    label: string;
    occurrences: number;
}

export interface OverallTrendQueryResult {
    laps: OverallTrendQueryLapResult[];
    categories: OverallTrendQueryCategoryResult[];
}

export type AnalysisResultsQueryInput = ActivePageQueryInput | OverallTrendQueryInput;

export interface AnalysisResultsQueryErrorDetail {
    code: string;
    position?: number;
    token?: string;
    message: string;
}

export class AnalysisResultsQueryError extends Error {
    readonly code: string;
    readonly position?: number;
    readonly token?: string;
    readonly detail: AnalysisResultsQueryErrorDetail;

    constructor(detail: AnalysisResultsQueryErrorDetail) {
        super(detail.message);
        Object.setPrototypeOf(this, new.target.prototype);
        this.name = 'AnalysisResultsQueryError';
        this.code = detail.code;
        this.position = detail.position;
        this.token = detail.token;
        this.detail = { ...detail };
    }
}

export const ANALYSIS_RESULTS_QUERY_GUARDRAILS = Object.freeze({
    timeout: 100,
    stack: 64,
    sequence: 10000,
});

type JsonCloneOptions = {
    allowJsonataArrayMetadata: boolean;
};

const JSONATA_ARRAY_METADATA_KEYS = new Set([
    'cons',
    'keepSingleton',
    'outerWrapper',
    'push',
    'sequence',
    'tupleStream',
]);

const hasOwn = (value: object, key: PropertyKey): boolean => (
    Object.prototype.hasOwnProperty.call(value, key)
);

const invalidQueryError = (message: string): AnalysisResultsQueryError => (
    new AnalysisResultsQueryError({ code: 'INVALID_QUERY', message })
);

const invalidInputError = (message: string): AnalysisResultsQueryError => (
    new AnalysisResultsQueryError({ code: 'INVALID_QUERY_INPUT', message })
);

const invalidJsonValueError = (path: string, message: string): AnalysisResultsQueryError => (
    new AnalysisResultsQueryError({
        code: 'INVALID_JSON_VALUE',
        message: `${path} ${message}`,
    })
);

const isRecord = (value: unknown): value is Record<string, unknown> => (
    Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

const assertPlainRecord = (
    value: unknown,
    path: string,
): Record<string, unknown> => {
    if (!isRecord(value)) {
        throw invalidInputError(`${path} must be an object.`);
    }
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
        throw invalidInputError(`${path} must be a plain JSON object.`);
    }
    return value;
};

const readDataProperty = (
    value: object,
    key: string,
    path: string,
): { present: boolean; value: unknown } => {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (!descriptor) return { present: false, value: undefined };
    if (!('value' in descriptor) || !descriptor.enumerable) {
        throw invalidInputError(`${path}.${key} must be an enumerable data property.`);
    }
    return { present: true, value: descriptor.value };
};

const requireDataProperty = (
    value: object,
    key: string,
    path: string,
): unknown => {
    const property = readDataProperty(value, key, path);
    if (!property.present) {
        throw invalidInputError(`${path}.${key} is required.`);
    }
    if (property.value === undefined) {
        throw invalidInputError(`${path}.${key} cannot be undefined.`);
    }
    return property.value;
};

const assertExactRootKeys = (
    value: Record<string, unknown>,
    expectedKey: 'elements' | 'pages',
    path: string,
): void => {
    const keys = Reflect.ownKeys(value);
    if (keys.length !== 1 || keys[0] !== expectedKey) {
        throw invalidInputError(`${path} must contain only '${expectedKey}'.`);
    }
};

const requireString = (value: unknown, path: string): string => {
    if (typeof value !== 'string') {
        throw invalidInputError(`${path} must be a string.`);
    }
    return value;
};

const requireNonEmptyString = (value: unknown, path: string): string => {
    const text = requireString(value, path);
    if (!text.trim()) {
        throw invalidInputError(`${path} must not be empty.`);
    }
    return text;
};

const requireFiniteNumber = (value: unknown, path: string): number => {
    if (typeof value !== 'number' || !Number.isFinite(value)) {
        throw invalidInputError(`${path} must be a finite number.`);
    }
    return value;
};

const requireDenseArray = (value: unknown, path: string): unknown[] => {
    if (!Array.isArray(value)) {
        throw invalidInputError(`${path} must be an array.`);
    }
    for (let index = 0; index < value.length; index += 1) {
        if (!hasOwn(value, index)) {
            throw invalidInputError(`${path}[${index}] cannot be missing.`);
        }
        const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
        if (!descriptor || !('value' in descriptor) || !descriptor.enumerable) {
            throw invalidInputError(`${path}[${index}] must be an enumerable data property.`);
        }
        if (descriptor.value === undefined) {
            throw invalidInputError(`${path}[${index}] cannot be undefined.`);
        }
    }
    return value;
};

const isCanonicalArrayIndex = (key: string, length: number): boolean => {
    const index = Number(key);
    return Number.isInteger(index)
        && index >= 0
        && index < length
        && String(index) === key;
};

const cloneJsonValue = (
    value: unknown,
    path: string,
    ancestors: Set<object>,
    options: JsonCloneOptions,
): JsonValue => {
    if (value === null || typeof value === 'string' || typeof value === 'boolean') {
        return value;
    }
    if (typeof value === 'number') {
        if (!Number.isFinite(value)) {
            throw invalidJsonValueError(path, 'must be a finite number.');
        }
        return value;
    }
    if (value === undefined) {
        throw invalidJsonValueError(path, 'cannot be undefined.');
    }
    if (typeof value === 'bigint') {
        throw invalidJsonValueError(path, 'cannot be a bigint.');
    }
    if (typeof value === 'function') {
        throw invalidJsonValueError(path, 'cannot be a function.');
    }
    if (typeof value === 'symbol') {
        throw invalidJsonValueError(path, 'cannot be a symbol.');
    }
    if (typeof value !== 'object') {
        throw invalidJsonValueError(path, 'is not JSON-serializable.');
    }
    if (ancestors.has(value)) {
        throw invalidJsonValueError(path, 'cannot contain a circular reference.');
    }

    ancestors.add(value);
    try {
        if (Array.isArray(value)) {
            const jsonataSequence = options.allowJsonataArrayMetadata
                && hasOwn(value, 'sequence')
                && (value as unknown as Record<string, unknown>).sequence === true;
            const jsonataConstructedArray = options.allowJsonataArrayMetadata
                && hasOwn(value, 'cons')
                && (value as unknown as Record<string, unknown>).cons === true;

            for (const key of Reflect.ownKeys(value)) {
                if (typeof key === 'symbol') {
                    throw invalidJsonValueError(path, 'cannot contain symbol properties.');
                }
                if (key === 'length' || isCanonicalArrayIndex(key, value.length)) continue;
                if (
                    (jsonataSequence || jsonataConstructedArray)
                    && JSONATA_ARRAY_METADATA_KEYS.has(key)
                ) continue;
                throw invalidJsonValueError(path, `cannot contain the non-JSON array property '${key}'.`);
            }

            const result: JsonValue[] = [];
            for (let index = 0; index < value.length; index += 1) {
                if (!hasOwn(value, index)) {
                    throw invalidJsonValueError(`${path}[${index}]`, 'cannot be missing.');
                }
                const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
                if (!descriptor || !('value' in descriptor) || !descriptor.enumerable) {
                    throw invalidJsonValueError(
                        `${path}[${index}]`,
                        'must be an enumerable data property.',
                    );
                }
                result.push(cloneJsonValue(
                    descriptor.value,
                    `${path}[${index}]`,
                    ancestors,
                    options,
                ));
            }
            return result;
        }

        const prototype = Object.getPrototypeOf(value);
        if (prototype !== Object.prototype && prototype !== null) {
            throw invalidJsonValueError(path, 'must be a plain JSON object.');
        }

        const result: JsonObject = {};
        for (const key of Reflect.ownKeys(value)) {
            if (typeof key === 'symbol') {
                throw invalidJsonValueError(path, 'cannot contain symbol properties.');
            }
            const descriptor = Object.getOwnPropertyDescriptor(value, key);
            if (!descriptor || !('value' in descriptor) || !descriptor.enumerable) {
                throw invalidJsonValueError(
                    `${path}.${key}`,
                    'must be an enumerable data property.',
                );
            }
            Object.defineProperty(result, key, {
                configurable: true,
                enumerable: true,
                writable: true,
                value: cloneJsonValue(descriptor.value, `${path}.${key}`, ancestors, options),
            });
        }
        return result;
    } finally {
        ancestors.delete(value);
    }
};

export const cloneJsonSafeValue = (value: unknown, path = 'value'): JsonValue => (
    cloneJsonValue(value, path, new Set<object>(), { allowJsonataArrayMetadata: false })
);

const cloneJsonSafeObject = (value: unknown, path: string): JsonObject => {
    const record = assertPlainRecord(value, path);
    const cloned = cloneJsonSafeValue(record, path);
    if (!isRecord(cloned)) {
        throw invalidInputError(`${path} must be an object.`);
    }
    return cloned as JsonObject;
};

const normalizeOptionalObject = (
    input: Record<string, unknown>,
    key: 'timeGap' | 'comparison' | 'metadata',
    path: string,
): JsonObject | undefined => {
    const property = readDataProperty(input, key, path);
    if (!property.present) return undefined;
    if (property.value === undefined) {
        throw invalidInputError(`${path}.${key} cannot be undefined.`);
    }
    return cloneJsonSafeObject(property.value, `${path}.${key}`);
};

const normalizeActivePageElement = (
    value: unknown,
    path: string,
): ActivePageQueryElement => {
    const input = assertPlainRecord(value, path);
    const id = requireNonEmptyString(requireDataProperty(input, 'id', path), `${path}.id`);
    const rawLabels = requireDenseArray(requireDataProperty(input, 'labels', path), `${path}.labels`);
    const labels = rawLabels.map((label, index) => (
        requireString(label, `${path}.labels[${index}]`)
    ));
    const result: ActivePageQueryElement = { id, labels };

    for (const key of ['title', 'section'] as const) {
        const property = readDataProperty(input, key, path);
        if (!property.present) continue;
        if (property.value === undefined) {
            throw invalidInputError(`${path}.${key} cannot be undefined.`);
        }
        result[key] = requireString(property.value, `${path}.${key}`);
    }

    const positionProperty = readDataProperty(input, 'normalizedPositionRange', path);
    if (positionProperty.present) {
        if (positionProperty.value === undefined) {
            throw invalidInputError(`${path}.normalizedPositionRange cannot be undefined.`);
        }
        const position = assertPlainRecord(
            positionProperty.value,
            `${path}.normalizedPositionRange`,
        );
        result.normalizedPositionRange = {
            start: requireFiniteNumber(
                requireDataProperty(position, 'start', `${path}.normalizedPositionRange`),
                `${path}.normalizedPositionRange.start`,
            ),
            end: requireFiniteNumber(
                requireDataProperty(position, 'end', `${path}.normalizedPositionRange`),
                `${path}.normalizedPositionRange.end`,
            ),
        };
    }

    const timeGap = normalizeOptionalObject(input, 'timeGap', path);
    const comparison = normalizeOptionalObject(input, 'comparison', path);
    const metadata = normalizeOptionalObject(input, 'metadata', path);
    if (timeGap) result.timeGap = timeGap;
    if (comparison) result.comparison = comparison;
    if (metadata) result.metadata = metadata;

    return result;
};

const normalizeElements = (value: unknown, path: string): ActivePageQueryElement[] => {
    const rawElements = requireDenseArray(value, path);
    const ids = new Set<string>();
    return rawElements.map((element, index) => {
        const normalized = normalizeActivePageElement(element, `${path}[${index}]`);
        if (ids.has(normalized.id)) {
            throw invalidInputError(`${path} contains duplicate element id '${normalized.id}'.`);
        }
        ids.add(normalized.id);
        return normalized;
    });
};

export const normalizeActivePageQueryInput = (value: unknown): ActivePageQueryInput => {
    const input = assertPlainRecord(value, 'active-page query input');
    assertExactRootKeys(input, 'elements', 'active-page query input');
    return {
        elements: normalizeElements(
            requireDataProperty(input, 'elements', 'active-page query input'),
            'active-page query input.elements',
        ),
    };
};

const normalizeOverallTrendPage = (
    value: unknown,
    sourceIndex: number,
    path: string,
): OverallTrendQueryPage => {
    const input = assertPlainRecord(value, path);
    const baseline = assertPlainRecord(
        requireDataProperty(input, 'baseline', path),
        `${path}.baseline`,
    );
    const lap = requireFiniteNumber(
        requireDataProperty(baseline, 'lap', `${path}.baseline`),
        `${path}.baseline.lap`,
    );
    if (lap <= 0) {
        throw invalidInputError(`${path}.baseline.lap must be positive.`);
    }

    const camelLapTime = readDataProperty(baseline, 'lapTimeMs', `${path}.baseline`);
    const snakeLapTime = readDataProperty(baseline, 'lap_time_ms', `${path}.baseline`);
    if (camelLapTime.present && snakeLapTime.present) {
        throw invalidInputError(
            `${path}.baseline must not contain both 'lapTimeMs' and 'lap_time_ms'.`,
        );
    }
    if (!camelLapTime.present && !snakeLapTime.present) {
        throw invalidInputError(`${path}.baseline.lapTimeMs is required.`);
    }
    const rawLapTime = camelLapTime.present ? camelLapTime.value : snakeLapTime.value;
    if (rawLapTime === undefined) {
        throw invalidInputError(`${path}.baseline.lapTimeMs cannot be undefined.`);
    }
    const lapTimeMs = rawLapTime === null
        ? null
        : requireFiniteNumber(rawLapTime, `${path}.baseline.lapTimeMs`);
    if (lapTimeMs !== null && lapTimeMs < 0) {
        throw invalidInputError(`${path}.baseline.lapTimeMs cannot be negative.`);
    }

    return {
        id: requireNonEmptyString(requireDataProperty(input, 'id', path), `${path}.id`),
        createdAt: requireFiniteNumber(
            requireDataProperty(input, 'createdAt', path),
            `${path}.createdAt`,
        ),
        sourceIndex,
        baseline: {
            lap,
            lapTimeMs,
            track: requireString(
                requireDataProperty(baseline, 'track', `${path}.baseline`),
                `${path}.baseline.track`,
            ),
            car: requireString(
                requireDataProperty(baseline, 'car', `${path}.baseline`),
                `${path}.baseline.car`,
            ),
        },
        elements: normalizeElements(
            requireDataProperty(input, 'elements', path),
            `${path}.elements`,
        ),
    };
};

export const normalizeOverallTrendQueryInput = (value: unknown): OverallTrendQueryInput => {
    const input = assertPlainRecord(value, 'overall-trend query input');
    assertExactRootKeys(input, 'pages', 'overall-trend query input');
    const rawPages = requireDenseArray(
        requireDataProperty(input, 'pages', 'overall-trend query input'),
        'overall-trend query input.pages',
    );
    const ids = new Set<string>();
    const pages = rawPages.map((page, sourceIndex) => {
        const normalized = normalizeOverallTrendPage(
            page,
            sourceIndex,
            `overall-trend query input.pages[${sourceIndex}]`,
        );
        if (ids.has(normalized.id)) {
            throw invalidInputError(
                `overall-trend query input.pages contains duplicate page id '${normalized.id}'.`,
            );
        }
        ids.add(normalized.id);
        return normalized;
    });
    return { pages };
};

export const normalizeAnalysisResultsQueryInput = (value: unknown): AnalysisResultsQueryInput => {
    if (!isRecord(value)) {
        throw invalidInputError("query input must be either an { elements } or { pages } root.");
    }
    const hasElements = hasOwn(value, 'elements');
    const hasPages = hasOwn(value, 'pages');
    if (hasElements === hasPages) {
        throw invalidInputError("query input must contain exactly one of 'elements' or 'pages'.");
    }
    return hasElements
        ? normalizeActivePageQueryInput(value)
        : normalizeOverallTrendQueryInput(value);
};

const ownString = (value: object, key: string): string | undefined => {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    return descriptor && 'value' in descriptor && typeof descriptor.value === 'string'
        ? descriptor.value
        : undefined;
};

const ownPosition = (value: object): number | undefined => {
    const descriptor = Object.getOwnPropertyDescriptor(value, 'position');
    const position = descriptor && 'value' in descriptor ? descriptor.value : undefined;
    return typeof position === 'number' && Number.isInteger(position) && position >= 0
        ? position
        : undefined;
};

export const normalizeAnalysisResultsQueryError = (
    error: unknown,
): AnalysisResultsQueryErrorDetail => {
    if (error instanceof AnalysisResultsQueryError) return { ...error.detail };

    const value = error && (typeof error === 'object' || typeof error === 'function')
        ? error as object
        : null;
    const code = value ? ownString(value, 'code') : undefined;
    const position = value ? ownPosition(value) : undefined;
    const token = value ? ownString(value, 'token') : undefined;
    const rawMessage = value ? ownString(value, 'message') : undefined;
    const message = rawMessage
        ?? (typeof error === 'string' && error ? error : 'JSONata query evaluation failed.');

    return {
        code: code ?? 'JSONATA_ERROR',
        ...(position !== undefined ? { position } : {}),
        ...(token !== undefined ? { token } : {}),
        message,
    };
};

export const toAnalysisResultsQueryError = (error: unknown): AnalysisResultsQueryError => (
    error instanceof AnalysisResultsQueryError
        ? error
        : new AnalysisResultsQueryError(normalizeAnalysisResultsQueryError(error))
);

export const detachJsonataResult = (value: unknown): JsonValue => {
    if (value === undefined) return null;
    if (
        Array.isArray(value)
        && value.length === 0
        && hasOwn(value, 'sequence')
        && (value as unknown as Record<string, unknown>).sequence === true
    ) return null;
    return cloneJsonValue(
        value,
        'query result',
        new Set<object>(),
        { allowJsonataArrayMetadata: true },
    );
};

const requireQueryString = (query: unknown): string => {
    if (typeof query !== 'string') {
        throw invalidQueryError('JSONata query must be a string.');
    }
    if (!query.trim()) {
        throw invalidQueryError('JSONata query must not be empty.');
    }
    return query;
};

export const compileAnalysisResultsQuery = (query: unknown): jsonata.Expression => {
    const source = requireQueryString(query);
    try {
        return jsonata(source, ANALYSIS_RESULTS_QUERY_GUARDRAILS);
    } catch (error) {
        throw toAnalysisResultsQueryError(error);
    }
};

export const evaluateAnalysisResultsQuery = async (
    query: unknown,
    input: unknown,
): Promise<JsonValue> => {
    const source = requireQueryString(query);
    const root = normalizeAnalysisResultsQueryInput(input);
    try {
        const expression = compileAnalysisResultsQuery(source);
        const result = await expression.evaluate(root);
        return detachJsonataResult(result);
    } catch (error) {
        throw toAnalysisResultsQueryError(error);
    }
};

export type ActivePageQueryTemplateKey =
    | 'all-results'
    | 'mistakes'
    | 'common-label-mistakes'
    | 'time-lost-mistakes';

export interface ActivePageQueryTemplate {
    key: ActivePageQueryTemplateKey;
    label: string;
    expression: string;
}

export interface ActivePageQueryTemplateTaxonomy {
    getCategoryLabels: (parentId: MistakeParentLabelId) => readonly string[];
    getLabelName?: AnalysisResultLabelResolver;
}

const ACTIVE_PAGE_TEMPLATE_LABELS: Readonly<Record<ActivePageQueryTemplateKey, string>> = {
    'all-results': 'All results',
    mistakes: 'Mistakes',
    'common-label-mistakes': 'Most common label in mistakes',
    'time-lost-mistakes': 'Most time lost in mistakes',
};

const MISTAKE_PARENT_IDS: readonly MistakeParentLabelId[] = ['MSP', 'MSR'];

const uniqueStrings = (values: readonly string[]): string[] => {
    const result: string[] = [];
    const seen = new Set<string>();
    for (const value of values) {
        if (seen.has(value)) continue;
        seen.add(value);
        result.push(value);
    }
    return result;
};

const buildMistakeParentValues = (
    getLabelName?: AnalysisResultLabelResolver,
): string[] => uniqueStrings(MISTAKE_PARENT_IDS.flatMap((parentId) => (
    Array.from(getAnalysisResultMistakeParentLabels(parentId, getLabelName))
)));

type ActivePageQueryCategory = {
    id: string;
    label: string;
    aliases: string[];
};

const buildMistakeCategories = (
    taxonomy: ActivePageQueryTemplateTaxonomy,
    parentValues: readonly string[],
): ActivePageQueryCategory[] => {
    const parentValueSet = new Set(parentValues);
    const childIds = uniqueStrings(MISTAKE_PARENT_IDS.flatMap((parentId) => (
        Array.from(taxonomy.getCategoryLabels(parentId))
    )));

    return childIds.map((id) => {
        const resolvedName = taxonomy.getLabelName?.(id);
        const label = resolvedName || id;
        return {
            id,
            label,
            aliases: uniqueStrings([id, ...(resolvedName ? [resolvedName] : [])])
                .filter((alias) => !parentValueSet.has(alias)),
        };
    });
};

const buildMistakesExpression = (parentValues: readonly string[]): string => `(
  $parentLabels := ${JSON.stringify(parentValues)};
  elements[labels[$ in $parentLabels]]
)`;

const buildCommonLabelMistakesExpression = (
    parentValues: readonly string[],
    categories: readonly ActivePageQueryCategory[],
): string => `(
  $parentLabels := ${JSON.stringify(parentValues)};
  $categories := ${JSON.stringify(categories)};
  $matchesParent := function($element) {
    $exists($element.labels[$ in $parentLabels])
  };
  $matchesCategory := function($element, $category) {
    $exists($element.labels[$ in $category.aliases])
  };
  $indexedMistakes := $filter(
    $map(elements, function($element, $sourceIndex) {
      {"element": $element, "sourceIndex": $sourceIndex}
    }),
    function($item) { $matchesParent($item.element) }
  );
  $countedCategories := $map($categories, function($category) {
    {
      "id": $category.id,
      "label": $category.label,
      "foldedLabel": $lowercase($category.label),
      "aliases": $category.aliases,
      "count": $count($filter(
        $indexedMistakes,
        function($item) { $matchesCategory($item.element, $category) }
      ))
    }
  });
  $rankedMistakes := $map($indexedMistakes, function($item) {(
    $matchingCategories := $filter(
      $countedCategories,
      function($category) { $matchesCategory($item.element, $category) }
    );
    $bestCategory := ($matchingCategories^(
      >count,
      <foldedLabel,
      <label
    ))[0];
    {
      "element": $item.element,
      "sourceIndex": $item.sourceIndex,
      "count": $exists($bestCategory) ? $bestCategory.count : 0,
      "foldedLabel": $exists($bestCategory) ? $bestCategory.foldedLabel : "",
      "label": $exists($bestCategory) ? $bestCategory.label : ""
    }
  )});
  $rankedMistakes^(
    >count,
    <foldedLabel,
    <label,
    <sourceIndex
  ).element
)`;

const buildTimeLostMistakesExpression = (parentValues: readonly string[]): string => `(
  $parentLabels := ${JSON.stringify(parentValues)};
  $indexedMistakes := $filter(
    $map(elements, function($element, $sourceIndex) {
      {"element": $element, "sourceIndex": $sourceIndex}
    }),
    function($item) {
      $exists($item.element.labels[$ in $parentLabels])
    }
  );
  $rankedMistakes := $map($indexedMistakes, function($item) {(
    $hasFiniteDelta := $type($item.element.timeGap.deltaMs) = "number";
    {
      "element": $item.element,
      "sourceIndex": $item.sourceIndex,
      "missingDelta": $hasFiniteDelta ? 0 : 1,
      "deltaMs": $hasFiniteDelta ? $item.element.timeGap.deltaMs : 0
    }
  )});
  $rankedMistakes^(
    <missingDelta,
    >deltaMs,
    <sourceIndex
  ).element
)`;

export const buildActivePageQueryTemplates = (
    taxonomy: ActivePageQueryTemplateTaxonomy,
): readonly ActivePageQueryTemplate[] => {
    const parentValues = buildMistakeParentValues(taxonomy.getLabelName);
    const categories = buildMistakeCategories(taxonomy, parentValues);

    return [
        {
            key: 'all-results',
            label: ACTIVE_PAGE_TEMPLATE_LABELS['all-results'],
            expression: 'elements',
        },
        {
            key: 'mistakes',
            label: ACTIVE_PAGE_TEMPLATE_LABELS.mistakes,
            expression: buildMistakesExpression(parentValues),
        },
        {
            key: 'common-label-mistakes',
            label: ACTIVE_PAGE_TEMPLATE_LABELS['common-label-mistakes'],
            expression: buildCommonLabelMistakesExpression(parentValues, categories),
        },
        {
            key: 'time-lost-mistakes',
            label: ACTIVE_PAGE_TEMPLATE_LABELS['time-lost-mistakes'],
            expression: buildTimeLostMistakesExpression(parentValues),
        },
    ];
};

const invalidOverallTrendTaxonomyError = (message: string): AnalysisResultsQueryError => (
    new AnalysisResultsQueryError({
        code: 'INVALID_OVERALL_TREND_TAXONOMY',
        message,
    })
);

const invalidOverallTrendResultError = (message: string): AnalysisResultsQueryError => (
    new AnalysisResultsQueryError({
        code: 'INVALID_OVERALL_TREND_QUERY_RESULT',
        message,
    })
);

const requireTaxonomyString = (
    value: unknown,
    path: string,
    optional = false,
): string | undefined => {
    if (value === undefined && optional) return undefined;
    if (typeof value !== 'string' || !value.trim()) {
        throw invalidOverallTrendTaxonomyError(`${path} must be a non-empty string.`);
    }
    return value;
};

const normalizeOverallTrendTaxonomyValue = (
    value: OverallTrendTaxonomyValue,
    path: string,
): OverallTrendTaxonomyValue => {
    if (!isRecord(value)) {
        throw invalidOverallTrendTaxonomyError(`${path} must be an object.`);
    }
    const id = requireTaxonomyString(value.id, `${path}.id`)!;
    const fallbackName = requireTaxonomyString(
        value.fallbackName,
        `${path}.fallbackName`,
    )!;
    const resolvedName = requireTaxonomyString(
        value.resolvedName,
        `${path}.resolvedName`,
        true,
    );
    return {
        id,
        fallbackName,
        ...(resolvedName !== undefined ? { resolvedName } : {}),
    };
};

const compareOverallTrendCategoryLabels = (
    left: Pick<OverallTrendQueryCategory, 'id' | 'label'>,
    right: Pick<OverallTrendQueryCategory, 'id' | 'label'>,
): number => {
    const foldedLeft = left.label.toLowerCase();
    const foldedRight = right.label.toLowerCase();
    if (foldedLeft < foldedRight) return -1;
    if (foldedLeft > foldedRight) return 1;
    if (left.label < right.label) return -1;
    if (left.label > right.label) return 1;
    if (left.id < right.id) return -1;
    if (left.id > right.id) return 1;
    return 0;
};

export const buildOverallTrendQueryDefinition = (
    taxonomy: OverallTrendQueryTaxonomy,
): OverallTrendQueryDefinition => {
    if (!isRecord(taxonomy)) {
        throw invalidOverallTrendTaxonomyError('Overall Trends taxonomy must be an object.');
    }
    const parent = normalizeOverallTrendTaxonomyValue(
        taxonomy.parent,
        'Overall Trends taxonomy.parent',
    );
    if (!Array.isArray(taxonomy.categories)) {
        throw invalidOverallTrendTaxonomyError(
            'Overall Trends taxonomy.categories must be an array.',
        );
    }

    const parentAliases = uniqueStrings([
        parent.id,
        parent.fallbackName,
        ...(parent.resolvedName ? [parent.resolvedName] : []),
    ]);
    const parentAliasSet = new Set(parentAliases);
    const categoryIds = new Set<string>();
    const categories: OverallTrendQueryCategory[] = taxonomy.categories.map((value, index) => {
        const category = normalizeOverallTrendTaxonomyValue(
            value,
            `Overall Trends taxonomy.categories[${index}]`,
        );
        if (categoryIds.has(category.id)) {
            throw invalidOverallTrendTaxonomyError(
                `Overall Trends taxonomy contains duplicate category ID '${category.id}'.`,
            );
        }
        categoryIds.add(category.id);
        return {
            id: category.id,
            label: category.resolvedName ?? category.fallbackName,
            aliases: uniqueStrings([
                category.id,
                category.fallbackName,
                ...(category.resolvedName ? [category.resolvedName] : []),
            ]).filter((alias) => !parentAliasSet.has(alias)),
        };
    });

    const expression = `(
  $parentLabels := ${JSON.stringify(parentAliases)};
  $categories := ${JSON.stringify(categories)};
  $matchesParent := function($element) {
    $exists($element.labels[$ in $parentLabels])
  };
  $matchesCategory := function($element, $category) {
    $exists($element.labels[$ in $category.aliases])
  };
  $orderedPages := pages;
  $laps := [$map($orderedPages, function($page) {(
    $mistakes := $filter(
      $page.elements,
      function($element) { $matchesParent($element) }
    );
    $categoryCounts := [$map($categories, function($category) {
      {
        "id": $category.id,
        "label": $category.label,
        "count": $count($filter(
          $mistakes,
          function($element) { $matchesCategory($element, $category) }
        ))
      }
    })];
    {
      "pageId": $page.id,
      "lap": $page.baseline.lap,
      "lapTimeMs": $page.baseline.lapTimeMs,
      "totalCount": $count($mistakes),
      "categoryCounts": $categoryCounts
    }
  )})];
  $totals := [$map($categories, function($category) {(
    $occurrences := $sum([0, $laps.categoryCounts[id = $category.id].count]);
    {
      "id": $category.id,
      "label": $category.label,
      "foldedLabel": $lowercase($category.label),
      "occurrences": $occurrences
    }
  )})];
  $positiveTotals := $filter(
    $totals,
    function($category) { $category.occurrences > 0 }
  )^(<foldedLabel, <label, <id);
  {
    "laps": $laps,
    "categories": [$map($positiveTotals, function($category) {
      {
        "id": $category.id,
        "label": $category.label,
        "occurrences": $category.occurrences
      }
    })]
  }
)`;

    return { expression, parentAliases, categories };
};

export const buildOverallTrendQueryExpression = (
    taxonomy: OverallTrendQueryTaxonomy,
): string => buildOverallTrendQueryDefinition(taxonomy).expression;

const requireOverallTrendResultObject = (
    value: unknown,
    path: string,
    expectedKeys: readonly string[],
): Record<string, unknown> => {
    if (!isRecord(value)) {
        throw invalidOverallTrendResultError(`${path} must be an object.`);
    }
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
        throw invalidOverallTrendResultError(`${path} must be a plain JSON object.`);
    }
    const keys = Reflect.ownKeys(value);
    const expected = new Set(expectedKeys);
    if (
        keys.length !== expected.size
        || keys.some((key) => typeof key !== 'string' || !expected.has(key))
    ) {
        throw invalidOverallTrendResultError(
            `${path} must contain exactly ${expectedKeys.map((key) => `'${key}'`).join(', ')}.`,
        );
    }
    return value;
};

const requireOverallTrendResultProperty = (
    value: Record<string, unknown>,
    key: string,
    path: string,
): unknown => {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (!descriptor || !('value' in descriptor) || !descriptor.enumerable) {
        throw invalidOverallTrendResultError(
            `${path}.${key} must be an enumerable data property.`,
        );
    }
    if (descriptor.value === undefined) {
        throw invalidOverallTrendResultError(`${path}.${key} cannot be undefined.`);
    }
    return descriptor.value;
};

const requireOverallTrendResultArray = (value: unknown, path: string): unknown[] => {
    if (!Array.isArray(value)) {
        throw invalidOverallTrendResultError(`${path} must be an array.`);
    }
    for (const key of Reflect.ownKeys(value)) {
        if (typeof key === 'symbol') {
            throw invalidOverallTrendResultError(`${path} cannot contain symbol properties.`);
        }
        if (key === 'length' || isCanonicalArrayIndex(key, value.length)) continue;
        throw invalidOverallTrendResultError(
            `${path} cannot contain the non-JSON array property '${key}'.`,
        );
    }
    for (let index = 0; index < value.length; index += 1) {
        const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
        if (!descriptor || !('value' in descriptor) || !descriptor.enumerable) {
            throw invalidOverallTrendResultError(
                `${path}[${index}] must be an enumerable data property.`,
            );
        }
        if (descriptor.value === undefined) {
            throw invalidOverallTrendResultError(`${path}[${index}] cannot be undefined.`);
        }
    }
    return value;
};

const requireOverallTrendResultString = (value: unknown, path: string): string => {
    if (typeof value !== 'string') {
        throw invalidOverallTrendResultError(`${path} must be a string.`);
    }
    return value;
};

const requireOverallTrendCount = (value: unknown, path: string): number => {
    if (
        typeof value !== 'number'
        || !Number.isFinite(value)
        || !Number.isInteger(value)
        || value < 0
    ) {
        throw invalidOverallTrendResultError(
            `${path} must be a finite non-negative integer.`,
        );
    }
    return value;
};

const requireOverallTrendLap = (value: unknown, path: string): number => {
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
        throw invalidOverallTrendResultError(`${path} must be a finite positive number.`);
    }
    return value;
};

const requireOverallTrendLapTime = (value: unknown, path: string): number | null => {
    if (value === null) return null;
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
        throw invalidOverallTrendResultError(
            `${path} must be null or a finite non-negative number.`,
        );
    }
    return value;
};

export const resolveOverallTrendQueryResult = (
    value: unknown,
    input: OverallTrendQueryInput,
    taxonomy: OverallTrendQueryTaxonomy,
): OverallTrendQueryResult => {
    const normalizedInput = normalizeOverallTrendQueryInput(input);
    const definition = buildOverallTrendQueryDefinition(taxonomy);
    const root = requireOverallTrendResultObject(value, 'Overall Trends query result', [
        'laps',
        'categories',
    ]);
    const rawLaps = requireOverallTrendResultArray(
        requireOverallTrendResultProperty(root, 'laps', 'Overall Trends query result'),
        'Overall Trends query result.laps',
    );
    const rawCategories = requireOverallTrendResultArray(
        requireOverallTrendResultProperty(root, 'categories', 'Overall Trends query result'),
        'Overall Trends query result.categories',
    );
    const orderedPages = normalizedInput.pages;
    if (rawLaps.length !== orderedPages.length) {
        throw invalidOverallTrendResultError(
            `Overall Trends query result.laps must contain exactly ${orderedPages.length} page rows.`,
        );
    }

    const expectedCategories = definition.categories;
    const expectedCategoriesById = new Map(expectedCategories.map((category) => (
        [category.id, category] as const
    )));
    const categorySums = new Map(expectedCategories.map((category) => [category.id, 0]));
    const seenPageIds = new Set<string>();
    const laps: OverallTrendQueryLapResult[] = rawLaps.map((rawLap, lapIndex) => {
        const path = `Overall Trends query result.laps[${lapIndex}]`;
        const lapObject = requireOverallTrendResultObject(rawLap, path, [
            'pageId',
            'lap',
            'lapTimeMs',
            'totalCount',
            'categoryCounts',
        ]);
        const pageId = requireOverallTrendResultString(
            requireOverallTrendResultProperty(lapObject, 'pageId', path),
            `${path}.pageId`,
        );
        if (seenPageIds.has(pageId)) {
            throw invalidOverallTrendResultError(`${path}.pageId duplicates page '${pageId}'.`);
        }
        seenPageIds.add(pageId);
        const page = orderedPages[lapIndex];
        if (pageId !== page.id) {
            const knownPage = orderedPages.some((candidate) => candidate.id === pageId);
            throw invalidOverallTrendResultError(
                knownPage
                    ? `${path}.pageId is out of retained-page array order.`
                    : `${path}.pageId references unknown page '${pageId}'.`,
            );
        }

        const lap = requireOverallTrendLap(
            requireOverallTrendResultProperty(lapObject, 'lap', path),
            `${path}.lap`,
        );
        if (lap !== page.baseline.lap) {
            throw invalidOverallTrendResultError(
                `${path}.lap does not match canonical page '${pageId}'.`,
            );
        }
        const lapTimeMs = requireOverallTrendLapTime(
            requireOverallTrendResultProperty(lapObject, 'lapTimeMs', path),
            `${path}.lapTimeMs`,
        );
        if (lapTimeMs !== page.baseline.lapTimeMs) {
            throw invalidOverallTrendResultError(
                `${path}.lapTimeMs does not match canonical page '${pageId}'.`,
            );
        }
        const totalCount = requireOverallTrendCount(
            requireOverallTrendResultProperty(lapObject, 'totalCount', path),
            `${path}.totalCount`,
        );
        const rawCategoryCounts = requireOverallTrendResultArray(
            requireOverallTrendResultProperty(lapObject, 'categoryCounts', path),
            `${path}.categoryCounts`,
        );
        if (rawCategoryCounts.length !== expectedCategories.length) {
            throw invalidOverallTrendResultError(
                `${path}.categoryCounts must contain every canonical category exactly once.`,
            );
        }

        const categoryCountsById = new Map<string, OverallTrendQueryCategoryCount>();
        rawCategoryCounts.forEach((rawCount, countIndex) => {
            const countPath = `${path}.categoryCounts[${countIndex}]`;
            const countObject = requireOverallTrendResultObject(rawCount, countPath, [
                'id',
                'label',
                'count',
            ]);
            const id = requireOverallTrendResultString(
                requireOverallTrendResultProperty(countObject, 'id', countPath),
                `${countPath}.id`,
            );
            const expected = expectedCategoriesById.get(id);
            if (!expected) {
                throw invalidOverallTrendResultError(
                    `${countPath}.id references unknown category '${id}'.`,
                );
            }
            if (categoryCountsById.has(id)) {
                throw invalidOverallTrendResultError(
                    `${countPath}.id duplicates category '${id}'.`,
                );
            }
            const label = requireOverallTrendResultString(
                requireOverallTrendResultProperty(countObject, 'label', countPath),
                `${countPath}.label`,
            );
            if (label !== expected.label) {
                throw invalidOverallTrendResultError(
                    `${countPath}.label does not match canonical category '${id}'.`,
                );
            }
            const count = requireOverallTrendCount(
                requireOverallTrendResultProperty(countObject, 'count', countPath),
                `${countPath}.count`,
            );
            if (count > totalCount) {
                throw invalidOverallTrendResultError(
                    `${countPath}.count cannot exceed the lap's totalCount.`,
                );
            }
            const normalizedCount = { id, label: expected.label, count };
            categoryCountsById.set(id, normalizedCount);
            categorySums.set(id, categorySums.get(id)! + count);
        });

        const categoryCounts = expectedCategories.map((category) => {
            const count = categoryCountsById.get(category.id);
            if (!count) {
                throw invalidOverallTrendResultError(
                    `${path}.categoryCounts is missing category '${category.id}'.`,
                );
            }
            return count;
        });
        return {
            pageId: page.id,
            label: `Analysis ${lapIndex + 1} \u00b7 Lap ${page.baseline.lap}`,
            lap: page.baseline.lap,
            lapTimeMs: page.baseline.lapTimeMs,
            totalCount,
            categoryCounts,
        };
    });

    const expectedPositiveCategories = expectedCategories
        .filter((category) => categorySums.get(category.id)! > 0)
        .sort(compareOverallTrendCategoryLabels);
    if (rawCategories.length !== expectedPositiveCategories.length) {
        throw invalidOverallTrendResultError(
            'Overall Trends query result.categories must contain exactly the positive-occurrence categories.',
        );
    }
    const seenCategoryIds = new Set<string>();
    const categories = rawCategories.map((rawCategory, index) => {
        const path = `Overall Trends query result.categories[${index}]`;
        const categoryObject = requireOverallTrendResultObject(rawCategory, path, [
            'id',
            'label',
            'occurrences',
        ]);
        const id = requireOverallTrendResultString(
            requireOverallTrendResultProperty(categoryObject, 'id', path),
            `${path}.id`,
        );
        if (seenCategoryIds.has(id)) {
            throw invalidOverallTrendResultError(`${path}.id duplicates category '${id}'.`);
        }
        seenCategoryIds.add(id);
        const expected = expectedPositiveCategories[index];
        const known = expectedCategoriesById.get(id);
        if (!known) {
            throw invalidOverallTrendResultError(
                `${path}.id references unknown category '${id}'.`,
            );
        }
        if (id !== expected.id) {
            throw invalidOverallTrendResultError(
                `${path} is not in deterministic category-label order.`,
            );
        }
        const label = requireOverallTrendResultString(
            requireOverallTrendResultProperty(categoryObject, 'label', path),
            `${path}.label`,
        );
        if (label !== expected.label) {
            throw invalidOverallTrendResultError(
                `${path}.label does not match canonical category '${id}'.`,
            );
        }
        const occurrences = requireOverallTrendCount(
            requireOverallTrendResultProperty(categoryObject, 'occurrences', path),
            `${path}.occurrences`,
        );
        const expectedOccurrences = categorySums.get(id)!;
        if (occurrences !== expectedOccurrences || occurrences === 0) {
            throw invalidOverallTrendResultError(
                `${path}.occurrences does not equal its per-lap count sum.`,
            );
        }
        return { id, label: expected.label, occurrences };
    });

    return { laps, categories };
};

export const validateOverallTrendQueryResult = resolveOverallTrendQueryResult;

const invalidActivePageResultError = (message: string): AnalysisResultsQueryError => (
    new AnalysisResultsQueryError({
        code: 'INVALID_ACTIVE_PAGE_QUERY_RESULT',
        message,
    })
);

const readActivePageResultId = (value: unknown, path: string): string => {
    if (typeof value === 'string') return value;
    if (!isRecord(value)) {
        throw invalidActivePageResultError(
            `${path} must be an element ID or an object with a string 'id'.`,
        );
    }
    const descriptor = Object.getOwnPropertyDescriptor(value, 'id');
    if (
        !descriptor
        || !('value' in descriptor)
        || !descriptor.enumerable
        || typeof descriptor.value !== 'string'
    ) {
        throw invalidActivePageResultError(`${path}.id must be an enumerable string property.`);
    }
    return descriptor.value;
};

export const resolveActivePageQueryResult = <TElement extends { id: string }>(
    value: unknown,
    canonicalElements: readonly TElement[],
): TElement[] => {
    if (value === null) return [];

    const elementsById = new Map<string, TElement>();
    canonicalElements.forEach((element) => {
        if (elementsById.has(element.id)) {
            throw invalidActivePageResultError(
                `The active page contains duplicate element ID '${element.id}'.`,
            );
        }
        elementsById.set(element.id, element);
    });

    const values = Array.isArray(value) ? value : [value];
    const resolved: TElement[] = [];
    const seenIds = new Set<string>();
    for (let index = 0; index < values.length; index += 1) {
        if (!hasOwn(values, index)) {
            throw invalidActivePageResultError(`query result[${index}] cannot be missing.`);
        }
        const item = values[index];
        if (Array.isArray(item)) {
            throw invalidActivePageResultError(`query result[${index}] cannot be a nested array.`);
        }
        const path = Array.isArray(value) ? `query result[${index}]` : 'query result';
        const id = readActivePageResultId(item, path);
        const canonical = elementsById.get(id);
        if (!canonical) {
            throw invalidActivePageResultError(`${path} references unknown element ID '${id}'.`);
        }
        if (seenIds.has(id)) continue;
        seenIds.add(id);
        resolved.push(canonical);
    }
    return resolved;
};
