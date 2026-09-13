import { GraphSeries, GraphSpec } from './types';

export const ACLA_DARK_PALETTE = [
    '#00e676',
    '#448aff',
    '#ffb300',
    '#ff1744',
    '#b388ff',
    '#26c6da',
    '#ff80ab',
    '#c6ff00',
] as const;

export const ACLA_GRAPH_THEME = {
    text: '#e8e8f0',
    mutedText: '#8080a0',
    axis: 'rgba(255, 255, 255, 0.18)',
    splitLine: 'rgba(255, 255, 255, 0.07)',
    tooltipBackground: '#161628',
    tooltipBorder: 'rgba(255, 255, 255, 0.12)',
} as const;

export const getSpecColors = (spec: GraphSpec): readonly string[] => (
    spec.colors?.length
        ? spec.colors
        : spec.seriesColors?.length
            ? spec.seriesColors
            : ACLA_DARK_PALETTE
);

export const getSeriesColor = (
    spec: GraphSpec,
    series: GraphSeries,
    index: number,
): string => series.color ?? getSpecColors(spec)[index % getSpecColors(spec).length];
