import jsonata from 'jsonata';
import {
    AnalysisResultsQueryError,
    evaluateAllAnalysisResultsQuery,
    evaluateAnalysisResultsQuery,
    normalizeAnalysisResultsQueryError,
    type JsonValue,
} from './analysisResultsQuery';
import { serializeError } from 'errors/OperationError';

jest.mock('jsonata', () => jest.fn());

beforeEach(() => {
    jest.mocked(jsonata).mockImplementation(jest.requireActual<typeof jsonata>('jsonata'));
});

const bytes = (value: unknown) => Buffer.byteLength(JSON.stringify(value), 'utf8');
const readyBytes = (data: unknown) => bytes({ status: 'ready', data });
const inputWithValue = (value: JsonValue) => ({ analyses: [{
    id: 'analysis', createdAt: null, baseline: null,
    elements: [{ id: 'element', labels: [], metadata: { value } }],
}] });
const readValue = (value: JsonValue) => evaluateAllAnalysisResultsQuery(
    'analyses.elements.metadata.value', inputWithValue(value),
);
const limitError = { code: 'QUERY_RESULT_LIMIT_EXCEEDED' };
const diagnosticLimitError = { code: 'QUERY_ERROR_DETAILS_LIMIT_EXCEEDED' };

describe('AI analysis query result limits', () => {
    it.each<[string, (padding: string) => JsonValue]>([
        ['ASCII', (padding: string) => padding],
        ['Unicode', (padding: string) => `é中😀${padding}`],
        ['escaped strings', (padding: string) => `"\\\b\t\n\f\r\u0000\u001f${padding}`],
        ['lone surrogates', (padding: string) => `\ud800x\udfff${padding}`],
        ['object keys', (padding: string) => ({ [`é中😀"\\\n\u0000\ud800${padding}`]: -1.25e-7 })],
        ['nested content', (padding: string) => ({ list: [null, true, false, -0, 1e30, { value: padding }] })],
    ])('accepts exactly 8192 bytes and rejects 8193 bytes: %s', async (_name, makeValue) => {
        const padding = 'x'.repeat(8192 - readyBytes(makeValue('')));
        const exact = makeValue(padding);
        const over = makeValue(`${padding}x`);
        expect(readyBytes(exact)).toBe(8192);
        expect(readyBytes(over)).toBe(8193);
        const result = await readValue(exact);
        expect(result).toEqual(exact);
        expect(readyBytes(result)).toBe(8192);
        await expect(readValue(over)).rejects.toMatchObject(limitError);
    });

    it.each<[string, (count: number) => string]>([
        ['root', (count: number) => `[1..${count}]`],
        ['object', (count: number) => `{"items":[1..${count}]}`],
        ['nested array', (count: number) => `[{"nested":{"items":[1..${count}]}}]`],
    ])('allows 50 items and rejects 51 items at %s', async (_name, query) => {
        const input = { analyses: [] };
        const result = await evaluateAllAnalysisResultsQuery(query(50), input);
        expect(readyBytes(result)).toBeLessThan(8192);
        await expect(evaluateAllAnalysisResultsQuery(query(51), input)).rejects.toMatchObject(limitError);
    });

    it('counts commas and delimiters across many object keys', async () => {
        const value: Record<string, JsonValue> = Object.fromEntries(
            Array.from({ length: 500 }, (_, index) => [`key${index}`, index % 2 === 0]),
        );
        value.padding = '';
        value.padding = 'x'.repeat(8192 - readyBytes(value));
        expect(readyBytes(value)).toBe(8192);
        await expect(readValue(value)).resolves.toEqual(value);
        value.extra = null;
        await expect(readValue(value)).rejects.toMatchObject(limitError);
    });

    it('rejects bulk results and stringified datasets but still aggregates all large input data', async () => {
        const input = { analyses: Array.from({ length: 100 }, (_, index) => ({
            id: `analysis-${index}`, createdAt: null, baseline: null,
            elements: Array.from({ length: 60 }, (_, element) => ({
                id: `element-${element}`, labels: [{ label_name: 'MSP', start_index: 0, end_index: 1 }], title: 'private-telemetry '.repeat(20),
            })),
        })) };
        for (const query of ['analyses', '$', 'analyses.elements', '{"all":analyses}', '$string(analyses)', '{"text":$string(analyses)}']) {
            await expect(evaluateAllAnalysisResultsQuery(query, input)).rejects.toMatchObject(limitError);
        }
        await expect(evaluateAllAnalysisResultsQuery(
            '{"analyses":$count(analyses),"elements":$count(analyses.elements)}', input,
        )).resolves.toEqual({ analyses: 100, elements: 6000 });
        await expect(evaluateAllAnalysisResultsQuery(
            '$count(analyses.elements[labels[label_name = "MSP"]])', input,
        )).resolves.toBe(6000);
    });

    it('allows a small complete dataset and independent repeated bounded queries', async () => {
        const input = inputWithValue('complete data');
        const result = await evaluateAllAnalysisResultsQuery('analyses', input);
        expect(result).toEqual([{ ...input.analyses[0], sourceIndex: 0 }]);
        expect(result).not.toBe(input.analyses);
        const value = 'x'.repeat(8192 - readyBytes(''));
        for (let index = 0; index < 3; index += 1) {
            await expect(readValue(value)).resolves.toBe(value);
        }
    });

    it('leaves manual chart and Overall Trends queries unrestricted by the AI output limits', async () => {
        const elements = Array.from({ length: 60 }, (_, index) => ({
            id: `element-${index}`, labels: [], title: 'x'.repeat(200),
        }));
        await expect(evaluateAnalysisResultsQuery('elements', { elements })).resolves.toEqual(elements);
        const pages = Array.from({ length: 60 }, (_, index) => ({
            id: `page-${index}`, createdAt: index, sourceIndex: index,
            baseline: { lap_id: index, lapTimeMs: null, track: 'Spa', car: 'GT3' }, elements,
        }));
        const result = await evaluateAnalysisResultsQuery('pages', { pages });
        expect(result).toHaveLength(60);
        expect(readyBytes(result)).toBeGreaterThan(8192);
    });

    it('rejects oversized containers before visiting later values or detaching the whole result', async () => {
        const visited = jest.fn(() => { throw new Error('Should not read this value'); });
        const hugeArray = new Array(100000);
        Object.defineProperty(hugeArray, '0', { get: visited, enumerable: true });
        const hugeObject: Record<string, unknown> = {};
        for (let index = 0; index < 10000; index += 1) hugeObject[`key${index}`] = 'x'.repeat(100);
        Object.defineProperty(hugeObject, 'later', { get: visited, enumerable: true });
        const descriptors = jest.spyOn(Object, 'getOwnPropertyDescriptor');
        for (const result of [hugeArray, hugeObject]) {
            jest.mocked(jsonata).mockReturnValueOnce({ evaluate: async () => result } as any);
            await expect(evaluateAllAnalysisResultsQuery('analyses', { analyses: [] })).rejects.toMatchObject(limitError);
        }
        expect(visited).not.toHaveBeenCalled();
        expect(descriptors.mock.calls.filter(([value]) => value === hugeObject).length).toBeLessThan(100);
        descriptors.mockRestore();
    });

    it('rejects deeply nested oversized results with the limit error rather than overflowing the call stack', async () => {
        let result: JsonValue = null;
        for (let index = 0; index < 10000; index += 1) result = [result];
        jest.mocked(jsonata).mockReturnValueOnce({ evaluate: async () => result } as any);
        await expect(evaluateAllAnalysisResultsQuery('analyses', { analyses: [] })).rejects.toMatchObject(limitError);
    });

    it('preserves JSON safety checks on the bounded path without executing getters or toJSON', async () => {
        const getter = jest.fn(() => 'unsafe');
        const cyclic: Record<string, unknown> = {};
        cyclic.self = cyclic;
        const unsafe = [cyclic, new Date(), new Array(1), NaN, Infinity, BigInt(1), Symbol('bad'), () => 1,
            { nested: undefined }, { [Symbol('key')]: 1 }, { toJSON: getter },
            Object.defineProperty({}, 'hidden', { value: 1 }),
            Object.defineProperty({}, 'accessor', { get: getter, enumerable: true }),
            Object.assign([], { extra: 1 }),
        ];
        for (const result of unsafe) {
            jest.mocked(jsonata).mockReturnValueOnce({ evaluate: async () => result } as any);
            await expect(evaluateAllAnalysisResultsQuery('analyses', { analyses: [] }))
                .rejects.toMatchObject({ code: 'INVALID_JSON_VALUE' });
        }
        expect(getter).not.toHaveBeenCalled();
    });
});

describe('bounded query error diagnostics', () => {
    it.each(['message', 'token', 'code'] as const)('bounds the complete detail including %s at 1024 serialized bytes', (field) => {
        const detail = { code: 'D3137', position: 12, token: 'error', message: 'é中😀"\\\n\ud800' };
        detail[field] += 'x'.repeat(1024 - bytes(detail));
        expect(bytes(detail)).toBe(1024);
        expect(normalizeAnalysisResultsQueryError(detail)).toEqual(detail);
        const exact = new AnalysisResultsQueryError(detail);
        expect(bytes(exact.detail)).toBe(1024);
        detail[field] += 'x';
        expect(bytes(detail)).toBe(1025);
        expect(normalizeAnalysisResultsQueryError(detail)).toMatchObject(diagnosticLimitError);
        const error = new AnalysisResultsQueryError(detail);
        expect(error).toMatchObject(diagnosticLimitError);
        expect(bytes(error.detail)).toBeLessThanOrEqual(1024);
        expect(error).not.toHaveProperty('cause');
        expect(JSON.stringify(serializeError(error))).not.toContain('xxxxxxxxxx');
        expect(error.stack).not.toContain('xxxxxxxxxx');
    });

    it('replaces $error($string(analyses)) diagnostics without retaining the original message or cause', async () => {
        const marker = 'private-analysis-content';
        const input = inputWithValue(marker.repeat(1000));
        const error = await evaluateAllAnalysisResultsQuery('$error($string(analyses))', input).catch((failure) => failure);
        expect(error).toBeInstanceOf(AnalysisResultsQueryError);
        expect(error).toMatchObject(diagnosticLimitError);
        expect(bytes(error.detail)).toBeLessThanOrEqual(1024);
        expect(error).not.toHaveProperty('cause');
        expect(error.message).not.toContain(marker);
        expect(error.stack).not.toContain(marker);
        expect(JSON.stringify(serializeError(error))).not.toContain(marker);
    });

    it('bounds syntax tokens, thrown strings, and normalization failures too', async () => {
        const marker = 'private-data-'.repeat(1000);
        expect(normalizeAnalysisResultsQueryError(marker)).toMatchObject(diagnosticLimitError);
        await expect(evaluateAllAnalysisResultsQuery(`$${marker}(`, { analyses: [] })).rejects.toBeInstanceOf(AnalysisResultsQueryError);
        const input = inputWithValue(null);
        input.analyses[0].id = marker;
        input.analyses.push(input.analyses[0]);
        const error = await evaluateAllAnalysisResultsQuery('analyses', input).catch((failure) => failure);
        expect(error).toMatchObject(diagnosticLimitError);
        expect(JSON.stringify(serializeError(error))).not.toContain('private-data-');
    });
});
