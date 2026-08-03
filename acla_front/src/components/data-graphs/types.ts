export type GraphRecord = Record<string, unknown>;

export interface GraphSeries {
    key: string;
    label?: string;
    color?: string;
}

export interface GraphAxisLabels {
    x?: string;
    y?: string;
}

interface CommonGraphSpec {
    title?: string;
    axisLabels?: GraphAxisLabels;
    xAxisLabel?: string;
    yAxisLabel?: string;
    showLegend?: boolean;
    colors?: readonly string[];
    seriesColors?: readonly string[];
    accessibleLabel?: string;
    ariaLabel?: string;
    height?: number | string;
    emptyStateText?: string;
}

export interface BarGraphSpec extends CommonGraphSpec {
    type: 'bar';
    data: readonly GraphRecord[];
    categoryKey: string;
    series: readonly GraphSeries[];
    orientation?: 'horizontal' | 'vertical';
    stacked?: boolean;
}

export interface LineGraphSpec extends CommonGraphSpec {
    type: 'line';
    data: readonly GraphRecord[];
    xKey: string;
    xAxisType: 'category' | 'value' | 'time';
    series: readonly GraphSeries[];
    smooth?: boolean;
    showPoints?: boolean;
}

export interface HistogramGraphSpec extends CommonGraphSpec {
    type: 'histogram';
    values: readonly number[];
    binCount?: number;
}

export type GraphSpec = BarGraphSpec | LineGraphSpec | HistogramGraphSpec;

export interface DataGraphProps {
    spec: GraphSpec;
    className?: string;
}
