import { resolveGraphSpec } from './strategies';

describe('graph strategies', () => {
    it('uses ECharts 6 outer bounds to contain axis labels', () => {
        const resolution = resolveGraphSpec({
            type: 'bar',
            data: [{ label: 'Turn 1', count: 2 }],
            categoryKey: 'label',
            series: [{ key: 'count' }],
        });

        expect(resolution.status).toBe('ready');
        if (resolution.status !== 'ready') return;
        expect(resolution.option.grid).toMatchObject({
            outerBounds: { top: 16, right: 20, bottom: 18, left: 20 },
        });
        expect(resolution.option.grid).not.toHaveProperty('containLabel');
    });
});
