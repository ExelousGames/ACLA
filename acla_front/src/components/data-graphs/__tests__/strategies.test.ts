import {
    buildHistogramBins,
    getHistogramBinCount,
    GRAPH_STRATEGY_TYPES,
    resolveGraphSpec,
} from '../strategies';

const readyOption = (spec: unknown): any => {
    const resolution = resolveGraphSpec(spec);
    expect(resolution.status).toBe('ready');
    if (resolution.status !== 'ready') throw new Error('Expected a ready graph option.');
    return resolution.option;
};

describe('data graph strategy registry', () => {
    it('exhaustively registers the supported graph discriminants', () => {
        expect(GRAPH_STRATEGY_TYPES).toEqual(['bar', 'line', 'histogram']);
    });

    it('builds a horizontal stacked multi-series bar option', () => {
        const option = readyOption({
            type: 'bar',
            data: [
                { team: 'A', wins: 3, podiums: 5 },
                { team: 'B', wins: 2, podiums: 4 },
            ],
            categoryKey: 'team',
            series: [
                { key: 'wins', label: 'Wins' },
                { key: 'podiums', label: 'Podiums' },
            ],
            orientation: 'horizontal',
            stacked: true,
            xAxisLabel: 'Results',
            yAxisLabel: 'Team',
        });

        expect(option.xAxis).toMatchObject({ type: 'value', name: 'Results', minInterval: 1 });
        expect(option.yAxis).toMatchObject({ type: 'category', name: 'Team', inverse: true });
        expect(option.series).toHaveLength(2);
        expect(option.series[0]).toMatchObject({
            type: 'bar',
            name: 'Wins',
            stack: 'total',
            encode: { x: 'wins', y: 'team' },
        });
        expect(option.series[1]).toMatchObject({ stack: 'total' });
    });

    it('builds a vertical unstacked bar option and omits non-finite values', () => {
        const option = readyOption({
            type: 'bar',
            data: [
                { category: 'kept', value: 1, second: Number.NaN },
                { category: 'also kept', value: Number.POSITIVE_INFINITY, second: 2.5 },
                { category: 'dropped', value: Number.NaN, second: Number.NEGATIVE_INFINITY },
            ],
            categoryKey: 'category',
            series: [{ key: 'value' }, { key: 'second' }],
            orientation: 'vertical',
        });

        expect(option.xAxis).toMatchObject({ type: 'category', inverse: false });
        expect(option.yAxis).toMatchObject({ type: 'value' });
        expect(option.yAxis).not.toHaveProperty('minInterval');
        expect(option.series[0]).not.toHaveProperty('stack');
        expect(option.dataset.source).toEqual([
            { category: 'kept', value: 1 },
            { category: 'also kept', second: 2.5 },
        ]);
    });

    it.each(['category', 'value', 'time'] as const)(
        'configures a %s line axis with smoothing and point markers',
        (xAxisType) => {
            const xValue = xAxisType === 'category'
                ? 'Lap 1'
                : xAxisType === 'time'
                    ? '2026-08-02T12:00:00Z'
                    : 1;
            const option = readyOption({
                type: 'line',
                data: [{ x: xValue, speed: 100, delta: 2 }],
                xKey: 'x',
                xAxisType,
                series: [{ key: 'speed' }, { key: 'delta' }],
                smooth: true,
                showPoints: false,
            });

            expect(option.xAxis.type).toBe(xAxisType);
            expect(option.series).toHaveLength(2);
            expect(option.series[0]).toMatchObject({
                type: 'line',
                smooth: true,
                showSymbol: false,
                symbol: 'circle',
                connectNulls: false,
            });
        },
    );

    it('drops invalid line x values and non-finite series values', () => {
        const option = readyOption({
            type: 'line',
            data: [
                { x: 1, value: 2 },
                { x: Number.NaN, value: 4 },
                { x: 2, value: Number.POSITIVE_INFINITY, other: 5 },
            ],
            xKey: 'x',
            xAxisType: 'value',
            series: [{ key: 'value' }, { key: 'other' }],
        });

        expect(option.dataset.source).toEqual([
            { x: 1, value: 2 },
            { x: 2, other: 5 },
        ]);
    });

    it('returns empty for valid specs without usable numeric data', () => {
        expect(resolveGraphSpec({
            type: 'bar',
            data: [{ category: 'A', value: Number.NaN }],
            categoryKey: 'category',
            series: [{ key: 'value' }],
        })).toEqual({ status: 'empty' });
    });

    it('returns unsupported for an invalid runtime discriminant or shape', () => {
        expect(resolveGraphSpec({ type: 'pie', data: [] })).toEqual({ status: 'unsupported' });
        expect(resolveGraphSpec({ type: 'bar', data: [], categoryKey: 'x', series: [] }))
            .toEqual({ status: 'unsupported' });
    });
});

describe('histogram strategy', () => {
    it('uses the square-root rule clamped to one through twenty bins', () => {
        expect(getHistogramBinCount(1)).toBe(1);
        expect(getHistogramBinCount(10)).toBe(4);
        expect(getHistogramBinCount(10_000)).toBe(20);
        expect(getHistogramBinCount(10, 100)).toBe(100);
    });

    it('creates one conserving bin for an equal-value dataset', () => {
        expect(buildHistogramBins([4, 4, 4], 10)).toEqual([
            { label: '4', start: 4, end: 4, count: 3 },
        ]);
    });

    it('uses contiguous boundaries, includes the maximum, and conserves the sample count', () => {
        const bins = buildHistogramBins([0, 1, 2, 3, 4, Number.NaN], 4);

        expect(bins).toHaveLength(4);
        expect(bins[0]).toMatchObject({ start: 0, end: 1 });
        expect(bins[3]).toMatchObject({ start: 3, end: 4 });
        expect(bins.slice(1).every((bin, index) => bin.start === bins[index].end)).toBe(true);
        expect(bins.reduce((total, bin) => total + bin.count, 0)).toBe(5);
        expect(bins[3].count).toBe(2);
    });

    it('renders histogram bins through the bar strategy with integer counts', () => {
        const option = readyOption({
            type: 'histogram',
            values: [0, 1, 2, 3, Number.POSITIVE_INFINITY],
            binCount: 2,
            xAxisLabel: 'Seconds',
            yAxisLabel: 'Count',
        });

        expect(option.xAxis).toMatchObject({ type: 'category', name: 'Seconds' });
        expect(option.yAxis).toMatchObject({ type: 'value', name: 'Count', minInterval: 1 });
        expect(option.series).toEqual([expect.objectContaining({ type: 'bar', name: 'Count' })]);
        expect(option.dataset.source.reduce((total: number, bin: any) => total + bin.count, 0)).toBe(4);
    });
});
