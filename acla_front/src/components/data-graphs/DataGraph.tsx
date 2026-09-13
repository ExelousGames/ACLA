import React from 'react';
import { BarChart, LineChart } from 'echarts/charts';
import {
    AriaComponent,
    DatasetComponent,
    GridComponent,
    TooltipComponent,
} from 'echarts/components';
import { init, use } from 'echarts/core';
import type { EChartsCoreOption } from 'echarts/core';
import { CanvasRenderer } from 'echarts/renderers';
import { resolveGraphSpec } from './strategies';
import { DataGraphProps, GraphSeries, GraphSpec } from './types';
import { getSeriesColor } from './theme';
import styles from './DataGraph.module.css';

use([
    BarChart,
    LineChart,
    GridComponent,
    TooltipComponent,
    DatasetComponent,
    AriaComponent,
    CanvasRenderer,
]);

const DEFAULT_GRAPH_HEIGHT = 300;
const UNSUPPORTED_STATE_TEXT = 'This graph configuration is unsupported.';

const toCssHeight = (height: number | string | undefined): number | string => {
    if (typeof height === 'number') return `${height}px`;
    return height ?? `${DEFAULT_GRAPH_HEIGHT}px`;
};

const getLegendSeries = (spec: GraphSpec): readonly GraphSeries[] => {
    if (spec.type === 'histogram') return [{ key: 'count', label: 'Count' }];
    return Array.isArray(spec.series) ? spec.series : [];
};

const getAccessibleLabel = (spec: GraphSpec): string => (
    spec.accessibleLabel ?? spec.ariaLabel ?? spec.title ?? `${spec.type} graph`
);

export const DataGraph: React.FC<DataGraphProps> = ({ spec, className }) => {
    const safeSpec = spec && typeof spec === 'object' ? spec : ({} as GraphSpec);
    const chartElementRef = React.useRef<HTMLDivElement | null>(null);
    const chartRef = React.useRef<ReturnType<typeof init> | null>(null);
    const resolution = React.useMemo(() => resolveGraphSpec(spec), [spec]);
    const ready = resolution.status === 'ready';

    React.useEffect(() => {
        if (!ready || !chartElementRef.current) return undefined;

        const chart = init(chartElementRef.current, undefined, { renderer: 'canvas' });
        chartRef.current = chart;
        const resizeObserver = typeof ResizeObserver === 'undefined'
            ? null
            : new ResizeObserver(() => chart.resize());
        resizeObserver?.observe(chartElementRef.current);

        return () => {
            resizeObserver?.disconnect();
            chart.dispose();
            chartRef.current = null;
        };
    }, [ready]);

    React.useEffect(() => {
        if (resolution.status !== 'ready' || !chartRef.current) return;
        chartRef.current.setOption(resolution.option as EChartsCoreOption, {
            notMerge: true,
            lazyUpdate: false,
        });
    }, [resolution]);

    const legendSeries = getLegendSeries(safeSpec);
    const showLegend = (safeSpec.showLegend ?? legendSeries.length > 1) && resolution.status === 'ready';
    const rootClassName = [styles.root, className].filter(Boolean).join(' ');

    return (
        <section
            className={rootClassName}
            style={{ height: toCssHeight(safeSpec.height) }}
            data-testid="data-graph"
        >
            {(safeSpec.title || showLegend) && (
                <header className={styles.header}>
                    {safeSpec.title && <h3 className={styles.title}>{safeSpec.title}</h3>}
                    {showLegend && (
                        <div className={styles.legend} aria-label="Graph legend">
                            {legendSeries.map((series, index) => (
                                <span className={styles.legendItem} key={`${series.key}-${index}`}>
                                    <span
                                        className={styles.legendSwatch}
                                        style={{ backgroundColor: getSeriesColor(safeSpec, series, index) }}
                                    />
                                    {series.label ?? series.key}
                                </span>
                            ))}
                        </div>
                    )}
                </header>
            )}
            {resolution.status === 'ready' ? (
                <div
                    ref={chartElementRef}
                    className={styles.chart}
                    role="img"
                    aria-label={getAccessibleLabel(safeSpec)}
                    data-testid="data-graph-canvas"
                />
            ) : (
                <div
                    className={styles.state}
                    role="status"
                    data-testid={resolution.status === 'empty'
                        ? 'data-graph-empty-state'
                        : 'data-graph-unsupported-state'}
                >
                    {resolution.status === 'empty'
                        ? safeSpec.emptyStateText ?? 'No data available.'
                        : UNSUPPORTED_STATE_TEXT}
                </div>
            )}
        </section>
    );
};
