jest.mock('./charts/TelemetryOverview', () => () => null);
jest.mock('./charts/MapVisualization', () => () => null);
jest.mock('./charts/ImitationGuidanceChart', () => () => null);
jest.mock('./charts/EventLogChart', () => () => null);
jest.mock('./charts/AnalysisResultsChart', () => () => null);

import { visualizationController, visualizationRegistry } from './VisualizationRegistry';

describe('analysis-results visualization controls', () => {
    beforeEach(() => {
        visualizationController.setUpdateCallback(() => undefined);
        visualizationController.setCurrentInstances([]);
    });

    it('exposes append, update, and remove through the generic control interface', async () => {
        const opened = visualizationController.openVisualization('analysis-results', {
            elements: [{ id: 'initial', labels: ['Mistake'] }],
        });

        expect(opened.success).toBe(true);
        expect(visualizationController.getVisualizationAssistantContext()).toEqual(expect.objectContaining({
            openInstances: expect.arrayContaining([
                expect.objectContaining({
                    id: opened.chartId,
                    controls: expect.arrayContaining([
                        expect.objectContaining({ name: 'append_element' }),
                        expect.objectContaining({ name: 'update_element' }),
                        expect.objectContaining({ name: 'remove_element' }),
                    ]),
                }),
            ]),
        }));

        const appended = await visualizationController.invokeVisualizationControl({
            id: opened.chartId,
            control: 'append_element',
            args: { element: { labels: ['Future label'] } },
        });
        expect(appended).toMatchObject({ success: true, data: { count: 2 } });
        const generatedId = appended.data.element.id;

        const updated = await visualizationController.invokeVisualizationControl({
            type: 'analysis-results',
            control: 'update_element',
            args: { id: generatedId, changes: { title: 'Updated' } },
        });
        expect(updated).toMatchObject({
            success: true,
            data: { element: { id: generatedId, title: 'Updated' }, count: 2 },
        });

        const removed = await visualizationController.invokeVisualizationControl({
            id: opened.chartId,
            control: 'remove_element',
            args: { id: 'initial' },
        });
        expect(removed).toMatchObject({ success: true, data: { id: 'initial', count: 1 } });
    });

    it('returns failed control results without changing chart data', async () => {
        const opened = visualizationController.openVisualization('analysis-results', {
            elements: [{ id: 'duplicate', labels: [] }],
        });
        const before = visualizationController.getCurrentInstances()[0].data;

        const result = await visualizationController.invokeVisualizationControl({
            id: opened.chartId,
            control: 'append_element',
            args: { element: { id: 'duplicate', labels: ['Other'] } },
        });

        expect(result).toMatchObject({
            success: false,
            message: expect.stringContaining('already exists'),
        });
        expect(visualizationController.getCurrentInstances()[0].data).toEqual(before);
    });

    it('keeps Analysis Results out of the recorded workspace menu', () => {
        const recordedTypes = visualizationRegistry.getRecordedWorkspaceTypes();

        expect(recordedTypes).not.toContain('analysis-results');
        expect(recordedTypes).toEqual(expect.arrayContaining([
            'telemetry-overview',
            'map-visualization',
            'imitation-guidance-chart',
            'event-log',
        ]));
    });
});
