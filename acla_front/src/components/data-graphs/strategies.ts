import {
    BarGraphSpec,
    GraphRecord,
    GraphSeries,
    GraphSpec,
    HistogramGraphSpec,
    LineGraphSpec,
    XYLineGraphSeries,
    XYLineGraphSpec,
} from './types';
import { ACLA_GRAPH_THEME, getSeriesColor, getSpecColors } from './theme';

type VendorGraphOption = Record<string, unknown>;
type GraphType = GraphSpec['type'];

interface GraphSpecByType {
    bar: BarGraphSpec;
    line: LineGraphSpec;
    'xy-line': XYLineGraphSpec;
    histogram: HistogramGraphSpec;
}

interface GraphStrategy<K extends GraphType> {
    isValid: (spec: unknown) => spec is GraphSpecByType[K];
    buildOption: (spec: GraphSpecByType[K]) => VendorGraphOption | null;
}

export interface HistogramBin {
    label: string;
    start: number;
    end: number;
    count: number;
}

export type GraphResolution =
    | { status: 'ready'; option: VendorGraphOption }
    | { status: 'empty' }
    | { status: 'unsupported' };

const isRecord = (value: unknown): value is GraphRecord => (
    Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

const isNonEmptyString = (value: unknown): value is string => (
    typeof value === 'string' && value.trim().length > 0
);

const isFiniteNumber = (value: unknown): value is number => (
    typeof value === 'number' && Number.isFinite(value)
);

const isGraphSeries = (value: unknown): value is GraphSeries => {
    if (!isRecord(value) || !isNonEmptyString(value.key)) return false;
    return (value.label === undefined || typeof value.label === 'string')
        && (value.color === undefined || typeof value.color === 'string');
};

const hasTabularShape = (spec: GraphRecord): boolean => (
    Array.isArray(spec.data)
    && spec.data.every(isRecord)
    && Array.isArray(spec.series)
    && spec.series.length > 0
    && spec.series.every(isGraphSeries)
);

const isCommonShape = (spec: GraphRecord): boolean => (
    (spec.height === undefined || (
        (typeof spec.height === 'number' && Number.isFinite(spec.height) && spec.height > 0)
        || isNonEmptyString(spec.height)
    ))
    && (spec.colors === undefined || (
        Array.isArray(spec.colors) && spec.colors.every(isNonEmptyString)
    ))
    && (spec.seriesColors === undefined || (
        Array.isArray(spec.seriesColors) && spec.seriesColors.every(isNonEmptyString)
    ))
);

const isBarGraphSpec = (value: unknown): value is BarGraphSpec => {
    if (!isRecord(value) || value.type !== 'bar' || !isCommonShape(value)) return false;
    return hasTabularShape(value)
        && isNonEmptyString(value.categoryKey)
        && (value.orientation === undefined || value.orientation === 'horizontal' || value.orientation === 'vertical')
        && (value.stacked === undefined || typeof value.stacked === 'boolean');
};

const isLineGraphSpec = (value: unknown): value is LineGraphSpec => {
    if (!isRecord(value) || value.type !== 'line' || !isCommonShape(value)) return false;
    return hasTabularShape(value)
        && isNonEmptyString(value.xKey)
        && (value.xAxisType === 'category' || value.xAxisType === 'value' || value.xAxisType === 'time')
        && (value.smooth === undefined || typeof value.smooth === 'boolean')
        && (value.showPoints === undefined || typeof value.showPoints === 'boolean')
        && (value.step === undefined
            || typeof value.step === 'boolean'
            || value.step === 'start'
            || value.step === 'middle'
            || value.step === 'end');
};

const isXYLineSeries = (value: unknown): value is XYLineGraphSeries => (
    isGraphSeries(value)
    && isRecord(value)
    && isNonEmptyString(value.xKey)
    && isNonEmptyString(value.yKey)
);

const isXYLineGraphSpec = (value: unknown): value is XYLineGraphSpec => {
    if (!isRecord(value) || value.type !== 'xy-line' || !isCommonShape(value)) return false;
    return Array.isArray(value.data)
        && value.data.every(isRecord)
        && Array.isArray(value.series)
        && value.series.length > 0
        && value.series.every(isXYLineSeries)
        && (value.smooth === undefined || typeof value.smooth === 'boolean')
        && (value.showPoints === undefined || typeof value.showPoints === 'boolean');
};

const isHistogramGraphSpec = (value: unknown): value is HistogramGraphSpec => {
    if (!isRecord(value) || value.type !== 'histogram' || !isCommonShape(value)) return false;
    return Array.isArray(value.values)
        && (value.binCount === undefined || (
            isFiniteNumber(value.binCount) && Number.isInteger(value.binCount) && value.binCount > 0
        ));
};

const formatBinValue = (value: number): string => {
    if (Number.isInteger(value)) return String(value);
    const magnitude = Math.abs(value);
    const precision = magnitude > 0 && magnitude < 0.01 ? 4 : 2;
    return Number(value.toFixed(precision)).toString();
};

export const getHistogramBinCount = (sampleCount: number, requested?: number): number => {
    if (sampleCount <= 0) return 0;
    if (requested !== undefined) return Math.max(1, Math.floor(requested));
    return Math.max(1, Math.min(20, Math.ceil(Math.sqrt(sampleCount))));
};

export const buildHistogramBins = (
    inputValues: readonly number[],
    requestedBinCount?: number,
): HistogramBin[] => {
    const values = inputValues.filter(isFiniteNumber);
    if (values.length === 0) return [];

    const minimum = Math.min(...values);
    const maximum = Math.max(...values);
    if (minimum === maximum) {
        return [{
            label: formatBinValue(minimum),
            start: minimum,
            end: maximum,
            count: values.length,
        }];
    }

    const binCount = getHistogramBinCount(values.length, requestedBinCount);
    const width = (maximum - minimum) / binCount;
    const bins = Array.from({ length: binCount }, (_, index) => {
        const start = minimum + (width * index);
        const end = index === binCount - 1 ? maximum : minimum + (width * (index + 1));
        return {
            label: `${formatBinValue(start)}–${formatBinValue(end)}`,
            start,
            end,
            count: 0,
        };
    });

    values.forEach((value) => {
        const index = value === maximum
            ? binCount - 1
            : Math.min(binCount - 1, Math.floor((value - minimum) / width));
        bins[index].count += 1;
    });
    return bins;
};

const axisLabelName = (spec: GraphSpec, axis: 'x' | 'y'): string | undefined => (
    spec.axisLabels?.[axis] ?? (axis === 'x' ? spec.xAxisLabel : spec.yAxisLabel)
);

const categoryValueIsUsable = (value: unknown): boolean => (
    isNonEmptyString(value) || isFiniteNumber(value)
);

const lineXValueIsUsable = (value: unknown, axisType: LineGraphSpec['xAxisType']): boolean => {
    if (axisType === 'value') return isFiniteNumber(value);
    if (axisType === 'category') return categoryValueIsUsable(value);
    if (isFiniteNumber(value)) return true;
    if (!isNonEmptyString(value)) return false;
    return Number.isFinite(Date.parse(value));
};

const sanitizeRows = (
    data: readonly GraphRecord[],
    dimensionKey: string,
    series: readonly GraphSeries[],
    dimensionIsUsable: (value: unknown) => boolean,
): GraphRecord[] => data.flatMap((row) => {
    if (!dimensionIsUsable(row[dimensionKey])) return [];
    const sanitized: GraphRecord = { [dimensionKey]: row[dimensionKey] };
    let hasNumericValue = false;
    series.forEach(({ key }) => {
        const value = row[key];
        if (isFiniteNumber(value)) {
            sanitized[key] = value;
            hasNumericValue = true;
        }
    });
    return hasNumericValue ? [sanitized] : [];
});

const sanitizeXYRows = (
    data: readonly GraphRecord[],
    series: readonly XYLineGraphSeries[],
): GraphRecord[] => data.flatMap((row) => {
    const sanitized: GraphRecord = {};
    let hasPair = false;
    series.forEach(({ xKey, yKey }) => {
        const x = row[xKey];
        const y = row[yKey];
        if (!isFiniteNumber(x) || !isFiniteNumber(y)) return;
        sanitized[xKey] = x;
        sanitized[yKey] = y;
        hasPair = true;
    });
    return hasPair ? [sanitized] : [];
});

const allSeriesValuesAreIntegers = (
    rows: readonly GraphRecord[],
    series: readonly GraphSeries[],
): boolean => {
    const values = rows.flatMap((row) => series.map(({ key }) => row[key]).filter(isFiniteNumber));
    return values.length > 0 && values.every(Number.isInteger);
};

const makeValueAxis = (name: string | undefined, integerValues: boolean): VendorGraphOption => ({
    type: 'value',
    name,
    nameTextStyle: { color: ACLA_GRAPH_THEME.mutedText },
    axisLabel: { color: ACLA_GRAPH_THEME.mutedText },
    axisLine: { lineStyle: { color: ACLA_GRAPH_THEME.axis } },
    splitLine: { lineStyle: { color: ACLA_GRAPH_THEME.splitLine } },
    ...(integerValues ? { minInterval: 1 } : {}),
});

const makeCategoryAxis = (
    name: string | undefined,
    inverse = false,
): VendorGraphOption => ({
    type: 'category',
    name,
    inverse,
    nameTextStyle: { color: ACLA_GRAPH_THEME.mutedText },
    axisLabel: { color: ACLA_GRAPH_THEME.mutedText },
    axisLine: { lineStyle: { color: ACLA_GRAPH_THEME.axis } },
    axisTick: { alignWithLabel: true },
});

const makeBaseOption = (spec: GraphSpec): VendorGraphOption => ({
    color: [...getSpecColors(spec)],
    backgroundColor: 'transparent',
    textStyle: { color: ACLA_GRAPH_THEME.text },
    animationDuration: 250,
    grid: {
        top: 16,
        right: 20,
        bottom: 18,
        left: 20,
        containLabel: true,
    },
    tooltip: {
        trigger: 'axis',
        confine: true,
        backgroundColor: ACLA_GRAPH_THEME.tooltipBackground,
        borderColor: ACLA_GRAPH_THEME.tooltipBorder,
        textStyle: { color: ACLA_GRAPH_THEME.text },
    },
    aria: {
        enabled: true,
        label: {
            enabled: true,
            description: spec.accessibleLabel ?? spec.ariaLabel ?? spec.title,
        },
    },
});

const buildBarOption = (spec: BarGraphSpec): VendorGraphOption | null => {
    const rows = sanitizeRows(spec.data, spec.categoryKey, spec.series, categoryValueIsUsable);
    if (rows.length === 0) return null;
    const horizontal = (spec.orientation ?? 'vertical') === 'horizontal';
    const integerValues = allSeriesValuesAreIntegers(rows, spec.series);
    const categoryAxis = makeCategoryAxis(
        axisLabelName(spec, horizontal ? 'y' : 'x'),
        horizontal,
    );
    const valueAxis = makeValueAxis(axisLabelName(spec, horizontal ? 'x' : 'y'), integerValues);

    return {
        ...makeBaseOption(spec),
        dataset: { source: rows },
        xAxis: horizontal ? valueAxis : categoryAxis,
        yAxis: horizontal ? categoryAxis : valueAxis,
        series: spec.series.map((series, index) => ({
            type: 'bar',
            name: series.label ?? series.key,
            encode: horizontal
                ? { x: series.key, y: spec.categoryKey }
                : { x: spec.categoryKey, y: series.key },
            itemStyle: { color: getSeriesColor(spec, series, index) },
            emphasis: { focus: 'series' },
            ...(spec.stacked ? { stack: 'total' } : {}),
        })),
    };
};

const buildLineOption = (spec: LineGraphSpec): VendorGraphOption | null => {
    const rows = sanitizeRows(
        spec.data,
        spec.xKey,
        spec.series,
        (value) => lineXValueIsUsable(value, spec.xAxisType),
    );
    if (rows.length === 0) return null;

    return {
        ...makeBaseOption(spec),
        dataset: { source: rows },
        xAxis: {
            ...(spec.xAxisType === 'category'
                ? makeCategoryAxis(axisLabelName(spec, 'x'))
                : makeValueAxis(axisLabelName(spec, 'x'), false)),
            type: spec.xAxisType,
        },
        yAxis: makeValueAxis(
            axisLabelName(spec, 'y'),
            allSeriesValuesAreIntegers(rows, spec.series),
        ),
        series: spec.series.map((series, index) => ({
            type: 'line',
            name: series.label ?? series.key,
            encode: { x: spec.xKey, y: series.key },
            smooth: spec.smooth ?? false,
            step: spec.step ?? false,
            showSymbol: spec.showPoints ?? true,
            symbol: 'circle',
            itemStyle: { color: getSeriesColor(spec, series, index) },
            lineStyle: { color: getSeriesColor(spec, series, index), width: 2 },
            connectNulls: false,
        })),
    };
};

const buildXYLineOption = (spec: XYLineGraphSpec): VendorGraphOption | null => {
    const rows = sanitizeXYRows(spec.data, spec.series);
    if (rows.length === 0) return null;

    return {
        ...makeBaseOption(spec),
        dataset: { source: rows },
        xAxis: {
            ...makeValueAxis(axisLabelName(spec, 'x'), false),
            scale: true,
        },
        yAxis: {
            ...makeValueAxis(axisLabelName(spec, 'y'), false),
            scale: true,
        },
        tooltip: {
            ...(makeBaseOption(spec).tooltip as VendorGraphOption),
            trigger: 'item',
        },
        series: spec.series.map((series, index) => ({
            type: 'line',
            name: series.label ?? series.key,
            encode: { x: series.xKey, y: series.yKey },
            smooth: spec.smooth ?? false,
            showSymbol: spec.showPoints ?? true,
            symbol: 'circle',
            itemStyle: { color: getSeriesColor(spec, series, index) },
            lineStyle: { color: getSeriesColor(spec, series, index), width: 2 },
            connectNulls: false,
        })),
    };
};

const buildHistogramOption = (spec: HistogramGraphSpec): VendorGraphOption | null => {
    const bins = buildHistogramBins(spec.values, spec.binCount);
    if (bins.length === 0) return null;
    const rows = bins.map(({ label, start, end, count }) => ({ label, start, end, count }));
    const histogramSeries: GraphSeries = { key: 'count', label: 'Count' };

    return {
        ...makeBaseOption(spec),
        dataset: { source: rows },
        xAxis: makeCategoryAxis(axisLabelName(spec, 'x')),
        yAxis: makeValueAxis(axisLabelName(spec, 'y'), true),
        series: [{
            type: 'bar',
            name: 'Count',
            encode: { x: 'label', y: 'count', tooltip: ['start', 'end', 'count'] },
            itemStyle: { color: getSeriesColor(spec, histogramSeries, 0) },
            barGap: '0%',
            barCategoryGap: '4%',
        }],
    };
};

const strategyRegistry: { [K in GraphType]: GraphStrategy<K> } = {
    bar: { isValid: isBarGraphSpec, buildOption: buildBarOption },
    line: { isValid: isLineGraphSpec, buildOption: buildLineOption },
    'xy-line': { isValid: isXYLineGraphSpec, buildOption: buildXYLineOption },
    histogram: { isValid: isHistogramGraphSpec, buildOption: buildHistogramOption },
};

export const GRAPH_STRATEGY_TYPES = Object.freeze(
    Object.keys(strategyRegistry) as GraphType[],
);

export const resolveGraphSpec = (value: unknown): GraphResolution => {
    if (!isRecord(value) || !isNonEmptyString(value.type)) return { status: 'unsupported' };

    try {
        switch (value.type) {
            case 'bar': {
                const strategy = strategyRegistry.bar;
                if (!strategy.isValid(value)) return { status: 'unsupported' };
                const option = strategy.buildOption(value);
                return option ? { status: 'ready', option } : { status: 'empty' };
            }
            case 'line': {
                const strategy = strategyRegistry.line;
                if (!strategy.isValid(value)) return { status: 'unsupported' };
                const option = strategy.buildOption(value);
                return option ? { status: 'ready', option } : { status: 'empty' };
            }
            case 'xy-line': {
                const strategy = strategyRegistry['xy-line'];
                if (!strategy.isValid(value)) return { status: 'unsupported' };
                const option = strategy.buildOption(value);
                return option ? { status: 'ready', option } : { status: 'empty' };
            }
            case 'histogram': {
                const strategy = strategyRegistry.histogram;
                if (!strategy.isValid(value)) return { status: 'unsupported' };
                const option = strategy.buildOption(value);
                return option ? { status: 'ready', option } : { status: 'empty' };
            }
            default:
                return { status: 'unsupported' };
        }
    } catch {
        return { status: 'unsupported' };
    }
};
