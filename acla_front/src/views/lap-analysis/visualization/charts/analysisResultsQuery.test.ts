import {
    AnalysisResultsQueryError,
    buildActivePageQueryTemplates,
    buildOverallTrendQueryDefinition,
    buildOverallTrendQueryExpression,
    detachJsonataResult,
    evaluateAllAnalysisResultsQuery,
    evaluateAnalysisResultsQuery,
    normalizeActivePageQueryInput,
    normalizeOverallTrendQueryInput,
    resolveActivePageQueryResult,
    resolveOverallTrendQueryResult,
    type ActivePageQueryElement,
    type ActivePageQueryTemplate,
    type ActivePageQueryTemplateTaxonomy,
    type OverallTrendQueryInput,
    type OverallTrendQueryResult,
    type OverallTrendQueryTaxonomy,
} from './analysisResultsQuery';

const activeElements: ActivePageQueryElement[] = [
    {
        id: 'a',
        labels: ['MSP', 'Lockup'],
        title: 'First',
        section: 'Turn 1',
        normalizedPositionRange: { start: 0.1, end: 0.2 },
        timeGap: { deltaMs: 125 },
        comparison: { score: 1 },
        metadata: { source: 'fixture' },
    },
    {
        id: 'b',
        labels: ['Informational'],
        title: 'Second',
        section: 'Turn 2',
    },
];

const activeInput = { elements: activeElements };

const getTemplate = (
    templates: readonly ActivePageQueryTemplate[],
    key: ActivePageQueryTemplate['key'],
): ActivePageQueryTemplate => {
    const template = templates.find((candidate) => candidate.key === key);
    if (!template) throw new Error(`Missing template '${key}'.`);
    return template;
};

describe('analysisResultsQuery evaluator', () => {
    it('evaluates scalars, objects, arrays, projections, aggregations, filters, and comments', async () => {
        await expect(evaluateAnalysisResultsQuery('$count(elements)', activeInput)).resolves.toBe(2);
        await expect(evaluateAnalysisResultsQuery(
            '{"id": elements[0].id, "title": elements[0].title}',
            activeInput,
        )).resolves.toEqual({ id: 'a', title: 'First' });
        await expect(evaluateAnalysisResultsQuery('[elements.id]', activeInput)).resolves.toEqual(['a', 'b']);
        await expect(evaluateAnalysisResultsQuery(
            'elements.{"id": id, "section": section}',
            activeInput,
        )).resolves.toEqual([
            { id: 'a', section: 'Turn 1' },
            { id: 'b', section: 'Turn 2' },
        ]);
        await expect(evaluateAnalysisResultsQuery(
            '{"count": $count(elements), "sections": $distinct(elements.section)}',
            activeInput,
        )).resolves.toEqual({ count: 2, sections: ['Turn 1', 'Turn 2'] });
        await expect(evaluateAnalysisResultsQuery(
            'elements[labels[$ = "Lockup"]].id',
            activeInput,
        )).resolves.toBe('a');
        await expect(evaluateAnalysisResultsQuery(
            '/* active page count */ $count(elements)',
            activeInput,
        )).resolves.toBe(2);
    });

    it('normalizes singleton collapse and an empty sequence without changing constructed arrays', async () => {
        await expect(evaluateAnalysisResultsQuery('elements[id = "a"].id', activeInput)).resolves.toBe('a');
        await expect(evaluateAnalysisResultsQuery('elements[id = "missing"]', activeInput)).resolves.toBeNull();
        await expect(evaluateAnalysisResultsQuery('[elements[id = "missing"]]', activeInput)).resolves.toEqual([]);
    });

    it('normalizes and isolates active-page and overall-trend roots', async () => {
        const normalizedActive = normalizeActivePageQueryInput(activeInput);
        expect(normalizedActive).toEqual(activeInput);
        expect(normalizedActive).not.toBe(activeInput);
        expect(normalizedActive.elements[0]).not.toBe(activeElements[0]);

        const normalizedTrend = normalizeOverallTrendQueryInput({
            pages: [
                {
                    id: 'later',
                    createdAt: 20,
                    sourceIndex: 77,
                    baseline: {
                        lap: 3,
                        lap_time_ms: 61_000,
                        track: 'Spa',
                        car: 'GT3',
                    },
                    elements: [],
                },
                {
                    id: 'same-time',
                    createdAt: 20,
                    sourceIndex: -5,
                    baseline: {
                        lap: 4,
                        lapTimeMs: null,
                        track: 'Spa',
                        car: 'GT3',
                    },
                    elements: [],
                },
            ],
        });
        expect(normalizedTrend.pages.map((page) => ({
            id: page.id,
            sourceIndex: page.sourceIndex,
            lapTimeMs: page.baseline.lapTimeMs,
        }))).toEqual([
            { id: 'later', sourceIndex: 0, lapTimeMs: 61_000 },
            { id: 'same-time', sourceIndex: 1, lapTimeMs: null },
        ]);

        await expect(evaluateAnalysisResultsQuery('pages', normalizedActive)).resolves.toBeNull();
        await expect(evaluateAnalysisResultsQuery('elements', normalizedTrend)).resolves.toBeNull();
        await expect(evaluateAnalysisResultsQuery('elements', {
            elements: [],
            pages: [],
        })).rejects.toMatchObject({ code: 'INVALID_QUERY_INPUT' });
        await expect(evaluateAnalysisResultsQuery('elements', { unexpected: [] })).rejects.toMatchObject({
            code: 'INVALID_QUERY_INPUT',
        });
    });

    it('evaluates one root containing every analysis result', async () => {
        const input = {
            analyses: [
                {
                    id: 'lap-one',
                    createdAt: 10,
                    baseline: {
                        lap: 1,
                        lap_time_ms: 61_000,
                        track: 'Spa',
                        car: 'GT3',
                    },
                    elements: activeElements,
                },
                {
                    id: 'recorded-analysis',
                    createdAt: null,
                    baseline: null,
                    elements: [],
                },
            ],
        };

        await expect(evaluateAllAnalysisResultsQuery(
            '{"analysisCount": $count(analyses), "segmentCount": $count(analyses.elements)}',
            input,
        )).resolves.toEqual({ analysisCount: 2, segmentCount: 2 });
        await expect(evaluateAllAnalysisResultsQuery(
            'analyses.{"id": id, "lap": baseline.lap, "segmentCount": $count(elements)}',
            input,
        )).resolves.toEqual([
            { id: 'lap-one', lap: 1, segmentCount: 2 },
            { id: 'recorded-analysis', segmentCount: 0 },
        ]);
        await expect(evaluateAllAnalysisResultsQuery('$count(analyses)', {
            ...input,
            elements: [],
        })).rejects.toMatchObject({ code: 'INVALID_QUERY_INPUT' });
    });

    it('counts a retained analysis whose raw telemetry lap is zero', async () => {
        const analysis = {
            id: 'zero-lap-analysis',
            createdAt: 10,
            baseline: {
                lap: 0,
                lap_time_ms: 61_000,
                track: 'Spa',
                car: 'GT3',
            },
            elements: [],
        };

        await expect(evaluateAllAnalysisResultsQuery(
            '$count(analyses)',
            { analyses: [analysis] },
        )).resolves.toBe(1);
        await expect(evaluateAllAnalysisResultsQuery(
            'analyses[0].baseline.lap',
            { analyses: [analysis] },
        )).resolves.toBe(0);
        expect(() => normalizeOverallTrendQueryInput({ pages: [analysis] })).toThrow(
            expect.objectContaining({ code: 'INVALID_QUERY_INPUT' }),
        );
    });

    it('rejects malformed all-analysis arrays, duplicate IDs, and unsafe element values', async () => {
        const analysis = {
            id: 'retained-analysis',
            createdAt: null,
            baseline: null,
            elements: [],
        };
        const sparseAnalyses = new Array(1);

        await expect(evaluateAllAnalysisResultsQuery('$count(analyses)', {
            analyses: sparseAnalyses,
        })).rejects.toMatchObject({ code: 'INVALID_QUERY_INPUT' });
        await expect(evaluateAllAnalysisResultsQuery('$count(analyses)', {
            analyses: [analysis, analysis],
        })).rejects.toMatchObject({ code: 'INVALID_QUERY_INPUT' });
        await expect(evaluateAllAnalysisResultsQuery('$count(analyses)', {
            analyses: [{
                ...analysis,
                elements: [{ id: 'unsafe', labels: [], metadata: { value: undefined } }],
            }],
        })).rejects.toMatchObject({ code: 'INVALID_JSON_VALUE' });
    });

    it('rejects cyclic and non-JSON-safe input before evaluation', async () => {
        const cyclicMetadata: Record<string, unknown> = {};
        cyclicMetadata.self = cyclicMetadata;

        const invalidValues: unknown[] = [
            cyclicMetadata,
            { nested: undefined },
            { value: () => true },
            { value: BigInt(1) },
            { value: Symbol('unsafe') },
            { value: Number.POSITIVE_INFINITY },
        ];

        for (const metadata of invalidValues) {
            await expect(evaluateAnalysisResultsQuery('elements', {
                elements: [{ id: 'unsafe', labels: [], metadata }],
            })).rejects.toMatchObject({ code: 'INVALID_JSON_VALUE' });
        }
    });

    it('rejects functions and every other unsafe output shape', async () => {
        await expect(evaluateAnalysisResultsQuery('function($value) { $value }', activeInput))
            .rejects.toMatchObject({ code: 'INVALID_JSON_VALUE' });

        const cyclic: Record<string, unknown> = {};
        cyclic.self = cyclic;
        const invalidResults: unknown[] = [
            cyclic,
            () => true,
            BigInt(1),
            Symbol('unsafe'),
            Number.NaN,
            Number.NEGATIVE_INFINITY,
            { nested: undefined },
        ];
        invalidResults.forEach((value) => {
            expect(() => detachJsonataResult(value)).toThrow(AnalysisResultsQueryError);
        });
    });

    it('preserves structured syntax and runtime error details', async () => {
        for (const query of ['elements[', '$error("expected runtime failure")']) {
            try {
                await evaluateAnalysisResultsQuery(query, activeInput);
                throw new Error('Expected the query to fail.');
            } catch (error) {
                expect(error).toBeInstanceOf(AnalysisResultsQueryError);
                expect(error).toMatchObject({
                    code: expect.any(String),
                    position: expect.any(Number),
                    token: expect.any(String),
                    message: expect.any(String),
                });
                expect((error as AnalysisResultsQueryError).position).toBeGreaterThan(0);
            }
        }
    });

    it('enforces timeout, stack, and sequence guardrails', async () => {
        await expect(evaluateAnalysisResultsQuery(
            '($loop := function($value) { $loop($value + 1) }; $loop(0))',
            activeInput,
        )).rejects.toMatchObject({
            message: expect.stringMatching(/time|timeout|duration/i),
        });

        await expect(evaluateAnalysisResultsQuery(
            '($recurse := function($value) { $value = 0 ? 0 : 1 + $recurse($value - 1) }; $recurse(1000))',
            activeInput,
        )).rejects.toMatchObject({
            message: expect.stringMatching(/stack|depth/i),
        });

        await expect(evaluateAnalysisResultsQuery('[1..10001]', activeInput)).rejects.toMatchObject({
            message: expect.stringMatching(/sequence|limit|items/i),
        });
    }, 10_000);

    it('returns detached arrays without JSONata sequence metadata', async () => {
        const result = await evaluateAnalysisResultsQuery('elements.id', activeInput);
        expect(result).toEqual(['a', 'b']);
        expect(Array.isArray(result)).toBe(true);
        expect(Reflect.ownKeys(result as unknown[])).toEqual(['0', '1', 'length']);
        expect(result).not.toBe(activeElements.map(({ id }) => id));
    });

    it('treats the removed enum names as ordinary missing field lookups', async () => {
        await expect(evaluateAnalysisResultsQuery('result_count', activeInput)).resolves.toBeNull();
        await expect(evaluateAnalysisResultsQuery('mistake_count', activeInput)).resolves.toBeNull();
    });
});

describe('resolveActivePageQueryResult', () => {
    const canonical = [
        { id: 'a', title: 'Canonical A' },
        { id: 'b', title: 'Canonical B' },
        { id: 'c', title: 'Canonical C' },
    ];

    it('accepts IDs, projections, singleton results, arrays, and null', () => {
        expect(resolveActivePageQueryResult(null, canonical)).toEqual([]);
        expect(resolveActivePageQueryResult('b', canonical)).toEqual([canonical[1]]);
        expect(resolveActivePageQueryResult({ id: 'a', title: 'Projected' }, canonical))
            .toEqual([canonical[0]]);
        expect(resolveActivePageQueryResult(['c', { id: 'a' }], canonical))
            .toEqual([canonical[2], canonical[0]]);
    });

    it('removes duplicate IDs while preserving the first-result order and canonical identity', () => {
        const resolved = resolveActivePageQueryResult(
            [{ id: 'b' }, 'a', 'b', { id: 'c' }, { id: 'a' }],
            canonical,
        );
        expect(resolved).toEqual([canonical[1], canonical[0], canonical[2]]);
        expect(resolved[0]).toBe(canonical[1]);
    });

    it.each([
        ['unknown ID', 'missing'],
        ['nested array', [['a']]],
        ['mixed invalid member', ['a', 2]],
        ['boolean scalar', true],
        ['number scalar', 1],
        ['undefined scalar', undefined],
        ['null array member', ['a', null]],
        ['missing projected ID', { title: 'No ID' }],
        ['non-string projected ID', { id: 2 }],
    ])('rejects the entire result for a %s', (_label, result) => {
        expect(() => resolveActivePageQueryResult(result, canonical)).toThrow(
            expect.objectContaining({ code: 'INVALID_ACTIVE_PAGE_QUERY_RESULT' }),
        );
    });

    it('rejects an ambiguous canonical element map', () => {
        expect(() => resolveActivePageQueryResult('a', [canonical[0], { id: 'a' }])).toThrow(
            expect.objectContaining({ code: 'INVALID_ACTIVE_PAGE_QUERY_RESULT' }),
        );
    });
});

describe('active-page query templates', () => {
    const labelNames: Record<string, string> = {
        MSP: 'Training mistake current',
        MSR: 'Racing mistake current',
        P1: 'Lockup',
        P2: 'Wide exit',
        R1: 'Spin',
    };
    const taxonomy: ActivePageQueryTemplateTaxonomy = {
        getCategoryLabels: (parentId) => (
            parentId === 'MSP' ? ['P1', 'P2'] : ['R1']
        ),
        getLabelName: (id) => labelNames[id],
    };
    const templates = buildActivePageQueryTemplates(taxonomy);
    const elements: ActivePageQueryElement[] = [
        { id: 'p1', labels: ['MSP', 'P1', 'Lockup'], timeGap: { deltaMs: 5 } },
        { id: 'p2', labels: ['Mistake (Practice)', 'Lockup'], timeGap: { deltaMs: 9 } },
        { id: 'wide', labels: ['Training mistake current', 'P2'], timeGap: { deltaMs: 'invalid' } },
        { id: 'spin', labels: ['MSR', 'Spin'], timeGap: { deltaMs: 5 } },
        { id: 'plain', labels: ['P1'], timeGap: { deltaMs: 500 } },
        { id: 'unlabelled', labels: ['Racing mistake current'], timeGap: {} },
        { id: 'negative', labels: ['Mistake (Racing)', 'R1'], timeGap: { deltaMs: -1 } },
    ];

    const evaluateIds = async (key: ActivePageQueryTemplate['key']): Promise<string[]> => {
        const result = await evaluateAnalysisResultsQuery(
            getTemplate(templates, key).expression,
            { elements },
        );
        return resolveActivePageQueryResult(result, elements).map(({ id }) => id);
    };

    it('publishes exactly the four stable keys and display labels', () => {
        expect(templates.map(({ key, label }) => ({ key, label }))).toEqual([
            { key: 'all-results', label: 'All results' },
            { key: 'mistakes', label: 'Mistakes' },
            { key: 'common-label-mistakes', label: 'Most common label in mistakes' },
            { key: 'time-lost-mistakes', label: 'Most time lost in mistakes' },
        ]);
    });

    it('returns all results in source order', async () => {
        await expect(evaluateIds('all-results')).resolves.toEqual(elements.map(({ id }) => id));
    });

    it('returns recognized Training and Racing mistakes in source order', async () => {
        await expect(evaluateIds('mistakes')).resolves.toEqual([
            'p1',
            'p2',
            'wide',
            'spin',
            'unlabelled',
            'negative',
        ]);
    });

    it('orders mistakes by deduplicated page-wide child-label frequency', async () => {
        await expect(evaluateIds('common-label-mistakes')).resolves.toEqual([
            'p1',
            'p2',
            'spin',
            'negative',
            'wide',
            'unlabelled',
        ]);
    });

    it('uses case-insensitive, exact-text, and source-order common-label ties', async () => {
        const tiedTemplates = buildActivePageQueryTemplates({
            getCategoryLabels: () => ['lower', 'upper'],
            getLabelName: (id) => ({
                lower: 'alpha',
                upper: 'Alpha',
            }[id]),
        });
        const tiedElements: ActivePageQueryElement[] = [
            { id: 'lower-first', labels: ['MSP', 'lower'] },
            { id: 'upper-first', labels: ['MSP', 'upper'] },
            { id: 'upper-second', labels: ['MSP', 'upper'] },
            { id: 'lower-second', labels: ['MSP', 'lower'] },
        ];
        const result = await evaluateAnalysisResultsQuery(
            getTemplate(tiedTemplates, 'common-label-mistakes').expression,
            { elements: tiedElements },
        );
        expect(resolveActivePageQueryResult(result, tiedElements).map(({ id }) => id)).toEqual([
            'upper-first',
            'upper-second',
            'lower-first',
            'lower-second',
        ]);
    });

    it('orders finite time loss first, descending, with source-order ties and invalid values last', async () => {
        await expect(evaluateIds('time-lost-mistakes')).resolves.toEqual([
            'p2',
            'p1',
            'spin',
            'negative',
            'wide',
            'unlabelled',
        ]);
    });

    it('escapes special-character and Unicode taxonomy values', async () => {
        const specialParent = 'Training "mistake"\\line\n雪';
        const specialChildId = 'child"\\\nΩ';
        const specialChildName = 'Locked\n"輪"';
        const specialTemplates = buildActivePageQueryTemplates({
            getCategoryLabels: (parentId) => parentId === 'MSP' ? [specialChildId] : [],
            getLabelName: (id) => ({
                MSP: specialParent,
                [specialChildId]: specialChildName,
            }[id]),
        });
        const specialElements: ActivePageQueryElement[] = [
            { id: 'special', labels: [specialParent, specialChildName] },
        ];

        for (const key of ['mistakes', 'common-label-mistakes'] as const) {
            const result = await evaluateAnalysisResultsQuery(
                getTemplate(specialTemplates, key).expression,
                { elements: specialElements },
            );
            expect(resolveActivePageQueryResult(result, specialElements)).toEqual(specialElements);
        }
    });

    it('deduplicates exact taxonomy values while preserving case-distinct values', () => {
        const duplicated = buildActivePageQueryTemplates({
            getCategoryLabels: () => ['child', 'child', 'Child'],
            getLabelName: (id) => id === 'MSP' ? 'MSP' : undefined,
        });
        const deduplicated = buildActivePageQueryTemplates({
            getCategoryLabels: () => ['child', 'Child'],
        });
        const withoutCaseDistinct = buildActivePageQueryTemplates({
            getCategoryLabels: () => ['child'],
        });

        expect(getTemplate(duplicated, 'common-label-mistakes').expression)
            .toBe(getTemplate(deduplicated, 'common-label-mistakes').expression);
        expect(getTemplate(duplicated, 'common-label-mistakes').expression)
            .not.toBe(getTemplate(withoutCaseDistinct, 'common-label-mistakes').expression);
    });
});

describe('Overall Trends query path', () => {
    const trainingTaxonomy: OverallTrendQueryTaxonomy = {
        parent: {
            id: 'MSP',
            fallbackName: 'Mistake (Practice)',
            resolvedName: 'Training "mistake" 雪',
        },
        categories: [
            {
                id: 'LOCK',
                fallbackName: 'Lock "up"',
                resolvedName: 'Locked "輪"',
            },
            {
                id: 'WIDE',
                fallbackName: 'Wide\\exit',
            },
            {
                id: 'ZERO',
                fallbackName: '雪 absent',
            },
        ],
    };
    const pages: OverallTrendQueryInput = {
        pages: [
            {
                id: 'same-time-first',
                createdAt: 20,
                sourceIndex: 91,
                baseline: {
                    lap: 2,
                    lapTimeMs: null,
                    track: 'Spa',
                    car: 'GT3',
                },
                elements: [
                    {
                        id: 'resolved-parent-and-child',
                        labels: ['Training "mistake" 雪', 'Locked "輪"'],
                    },
                ],
            },
            {
                id: 'earliest',
                createdAt: 10,
                sourceIndex: -10,
                baseline: {
                    lap: 1,
                    lapTimeMs: 60_000,
                    track: 'Spa',
                    car: 'GT3',
                },
                elements: [
                    {
                        id: 'deduplicated-labels',
                        labels: ['MSP', 'LOCK', 'LOCK', 'Locked "輪"'],
                    },
                    {
                        id: 'two-categories',
                        labels: ['Mistake (Practice)', 'Lock "up"', 'WIDE'],
                    },
                    {
                        id: 'child-without-parent',
                        labels: ['LOCK'],
                    },
                ],
            },
            {
                id: 'same-time-second',
                createdAt: 20,
                sourceIndex: -100,
                baseline: {
                    lap: 3,
                    lapTimeMs: 59_500,
                    track: 'Spa',
                    car: 'GT3',
                },
                elements: [
                    { id: 'wide-one', labels: ['Mistake (Practice)', 'Wide\\exit'] },
                    { id: 'wide-two', labels: ['MSP', 'WIDE'] },
                ],
            },
            {
                id: 'empty',
                createdAt: 30,
                sourceIndex: 0,
                baseline: {
                    lap: 4,
                    lapTimeMs: 0,
                    track: 'Spa',
                    car: 'GT3',
                },
                elements: [],
            },
        ],
    };

    const evaluateAndResolve = async (
        taxonomy = trainingTaxonomy,
        input = pages,
    ): Promise<OverallTrendQueryResult> => {
        const raw = await evaluateAnalysisResultsQuery(
            buildOverallTrendQueryExpression(taxonomy),
            input,
        );
        return resolveOverallTrendQueryResult(raw, input, taxonomy);
    };

    it('generates a complete Training expression in retained-array order regardless of timestamps', async () => {
        const result = await evaluateAndResolve();

        expect(result.laps.map((lap) => ({
            pageId: lap.pageId,
            label: lap.label,
            lap: lap.lap,
            lapTimeMs: lap.lapTimeMs,
            totalCount: lap.totalCount,
            counts: lap.categoryCounts.map(({ id, count }) => [id, count]),
        }))).toEqual([
            {
                pageId: 'same-time-first',
                label: 'Analysis 1 · Lap 2',
                lap: 2,
                lapTimeMs: null,
                totalCount: 1,
                counts: [['LOCK', 1], ['WIDE', 0], ['ZERO', 0]],
            },
            {
                pageId: 'earliest',
                label: 'Analysis 2 · Lap 1',
                lap: 1,
                lapTimeMs: 60_000,
                totalCount: 2,
                counts: [['LOCK', 2], ['WIDE', 1], ['ZERO', 0]],
            },
            {
                pageId: 'same-time-second',
                label: 'Analysis 3 · Lap 3',
                lap: 3,
                lapTimeMs: 59_500,
                totalCount: 2,
                counts: [['LOCK', 0], ['WIDE', 2], ['ZERO', 0]],
            },
            {
                pageId: 'empty',
                label: 'Analysis 4 · Lap 4',
                lap: 4,
                lapTimeMs: 0,
                totalCount: 0,
                counts: [['LOCK', 0], ['WIDE', 0], ['ZERO', 0]],
            },
        ]);
        expect(result.categories).toEqual([
            { id: 'LOCK', label: 'Locked "輪"', occurrences: 3 },
            { id: 'WIDE', label: 'Wide\\exit', occurrences: 3 },
        ]);
    });

    it('builds an independent Racing expression from canonical, fallback, and resolved values', async () => {
        const racingTaxonomy: OverallTrendQueryTaxonomy = {
            parent: {
                id: 'MSR',
                fallbackName: 'Mistake (Racing)',
                resolvedName: 'Race Ω',
            },
            categories: [
                {
                    id: 'SPIN',
                    fallbackName: 'Spin fallback',
                    resolvedName: 'Spin "resolved"',
                },
            ],
        };
        const racingInput: OverallTrendQueryInput = {
            pages: [{
                id: 'race',
                createdAt: 1,
                sourceIndex: 0,
                baseline: {
                    lap: 7,
                    lapTimeMs: 70_000,
                    track: '鈴鹿',
                    car: 'GT4',
                },
                elements: [
                    { id: 'canonical', labels: ['MSR', 'SPIN'] },
                    { id: 'fallback', labels: ['Mistake (Racing)', 'Spin fallback'] },
                    { id: 'resolved', labels: ['Race Ω', 'Spin "resolved"'] },
                ],
            }],
        };

        await expect(evaluateAndResolve(racingTaxonomy, racingInput)).resolves.toMatchObject({
            laps: [{
                pageId: 'race',
                totalCount: 3,
                categoryCounts: [{ id: 'SPIN', count: 3 }],
            }],
            categories: [{ id: 'SPIN', label: 'Spin "resolved"', occurrences: 3 }],
        });
    });

    it('preserves array shapes when no page has a matching mistake', async () => {
        const input: OverallTrendQueryInput = {
            pages: [{
                id: 'no-matches',
                createdAt: 1,
                sourceIndex: 0,
                baseline: { lap: 1, lapTimeMs: null, track: '', car: '' },
                elements: [{ id: 'informational', labels: ['Informational'] }],
            }],
        };

        await expect(evaluateAndResolve(trainingTaxonomy, input)).resolves.toEqual({
            laps: [{
                pageId: 'no-matches',
                label: 'Analysis 1 · Lap 1',
                lap: 1,
                lapTimeMs: null,
                totalCount: 0,
                categoryCounts: [
                    { id: 'LOCK', label: 'Locked "輪"', count: 0 },
                    { id: 'WIDE', label: 'Wide\\exit', count: 0 },
                    { id: 'ZERO', label: '雪 absent', count: 0 },
                ],
            }],
            categories: [],
        });
    });

    it('deduplicates exact aliases and safely escapes quotes, slashes, newlines, and Unicode', async () => {
        const duplicated: OverallTrendQueryTaxonomy = {
            parent: { id: 'P', fallbackName: 'P', resolvedName: 'Parent\n雪' },
            categories: [{ id: 'C', fallbackName: 'C', resolvedName: 'Child "\\輪"\n' }],
        };
        const definition = buildOverallTrendQueryDefinition(duplicated);
        expect(definition.parentAliases).toEqual(['P', 'Parent\n雪']);
        expect(definition.categories).toEqual([{
            id: 'C',
            label: 'Child "\\輪"\n',
            aliases: ['C', 'Child "\\輪"\n'],
        }]);

        const input: OverallTrendQueryInput = {
            pages: [{
                id: 'special',
                createdAt: 1,
                sourceIndex: 0,
                baseline: { lap: 1, lapTimeMs: null, track: '', car: '' },
                elements: [{ id: 'special-element', labels: ['Parent\n雪', 'Child "\\輪"\n'] }],
            }],
        };
        await expect(evaluateAndResolve(duplicated, input)).resolves.toMatchObject({
            categories: [{ id: 'C', occurrences: 1 }],
        });
    });

    describe('strict result validation', () => {
        type RawOverallTrendResult = {
            laps: Array<Omit<OverallTrendQueryResult['laps'][number], 'label'>>;
            categories: OverallTrendQueryResult['categories'];
        };

        let valid: RawOverallTrendResult;

        beforeAll(async () => {
            valid = await evaluateAnalysisResultsQuery(
                buildOverallTrendQueryExpression(trainingTaxonomy),
                pages,
            ) as unknown as RawOverallTrendResult;
            expect(() => resolveOverallTrendQueryResult(valid, pages, trainingTaxonomy)).not.toThrow();
        });

        const cloneValid = (): RawOverallTrendResult => JSON.parse(JSON.stringify(valid));
        const expectInvalid = (value: unknown): void => {
            expect(() => resolveOverallTrendQueryResult(value, pages, trainingTaxonomy)).toThrow(
                expect.objectContaining({ code: 'INVALID_OVERALL_TREND_QUERY_RESULT' }),
            );
        };

        it('rejects unknown, missing, duplicate, and out-of-order pages', () => {
            const unknown = cloneValid();
            unknown.laps[0].pageId = 'unknown';
            expectInvalid(unknown);

            const missing = cloneValid();
            missing.laps.pop();
            expectInvalid(missing);

            const duplicate = cloneValid();
            duplicate.laps[1].pageId = duplicate.laps[0].pageId;
            expectInvalid(duplicate);

            const reordered = cloneValid();
            [reordered.laps[0], reordered.laps[1]] = [reordered.laps[1], reordered.laps[0]];
            expectInvalid(reordered);
        });

        it('rejects unknown, missing, duplicate, mislabeled, and misordered categories', () => {
            const unknown = cloneValid();
            unknown.laps[0].categoryCounts[0].id = 'unknown';
            expectInvalid(unknown);

            const missing = cloneValid();
            missing.laps[0].categoryCounts.pop();
            expectInvalid(missing);

            const duplicate = cloneValid();
            duplicate.laps[0].categoryCounts[1].id = duplicate.laps[0].categoryCounts[0].id;
            expectInvalid(duplicate);

            const mislabeled = cloneValid();
            mislabeled.laps[0].categoryCounts[0].label = 'Projected label';
            expectInvalid(mislabeled);

            const topUnknown = cloneValid();
            topUnknown.categories[0].id = 'unknown';
            expectInvalid(topUnknown);

            const topDuplicate = cloneValid();
            topDuplicate.categories[1].id = topDuplicate.categories[0].id;
            expectInvalid(topDuplicate);

            const topMissing = cloneValid();
            topMissing.categories.pop();
            expectInvalid(topMissing);

            const topReordered = cloneValid();
            [topReordered.categories[0], topReordered.categories[1]] = [
                topReordered.categories[1],
                topReordered.categories[0],
            ];
            expectInvalid(topReordered);
        });

        it('rejects negative, fractional, non-finite, excessive, and inconsistent counts', () => {
            for (const count of [-1, 0.5, Number.POSITIVE_INFINITY, Number.NaN]) {
                const invalid = cloneValid();
                invalid.laps[0].categoryCounts[0].count = count;
                expectInvalid(invalid);
            }

            const excessive = cloneValid();
            excessive.laps[0].categoryCounts[0].count = excessive.laps[0].totalCount + 1;
            expectInvalid(excessive);

            const badTotal = cloneValid();
            badTotal.laps[0].totalCount = -1;
            expectInvalid(badTotal);

            const inconsistent = cloneValid();
            inconsistent.categories[0].occurrences += 1;
            expectInvalid(inconsistent);
        });

        it('rejects invalid or non-canonical lap values and lap times', () => {
            for (const lap of [0, -1, Number.POSITIVE_INFINITY, Number.NaN, 1.5]) {
                const invalid = cloneValid();
                invalid.laps[0].lap = lap;
                expectInvalid(invalid);
            }
            for (const lapTimeMs of [-1, Number.POSITIVE_INFINITY, Number.NaN, 123]) {
                const invalid = cloneValid();
                invalid.laps[1].lapTimeMs = lapTimeMs;
                expectInvalid(invalid);
            }
        });

        it('rejects malformed or unexpectedly nested shapes', () => {
            expectInvalid(null);
            expectInvalid([]);
            expectInvalid({ laps: valid.laps, categories: valid.categories, extra: true });

            const nestedCounts = cloneValid() as unknown as {
                laps: Array<{ categoryCounts: unknown }>;
                categories: unknown;
            };
            nestedCounts.laps[0].categoryCounts = { unexpected: [] };
            expectInvalid(nestedCounts);

            const nestedCategory = cloneValid() as unknown as {
                laps: unknown;
                categories: unknown[];
            };
            nestedCategory.categories[0] = [nestedCategory.categories[0]];
            expectInvalid(nestedCategory);
        });
    });
});
