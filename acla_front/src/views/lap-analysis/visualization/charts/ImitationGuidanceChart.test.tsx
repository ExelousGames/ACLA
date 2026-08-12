import React, { createRef } from 'react';
import { act, render, screen } from '@testing-library/react';
import apiService from 'services/api.service';
import { AiToolComponentRefProvider } from 'contexts/AiToolComponentRefContext';
import { VisualizationControlFailedError } from 'contexts/AiToolComponentError';
import ImitationGuidanceChart, { ImitationGuidanceChartHandle } from './ImitationGuidanceChart';

jest.mock('@radix-ui/themes', () => {
    const React = require('react');
    const Element = ({ children }: { children?: React.ReactNode }) => React.createElement('div', null, children);
    return {
        Box: Element,
        Button: Element,
        Card: Element,
        Flex: Element,
        IconButton: Element,
        Text: Element,
        TextField: { Root: Element },
        Table: {
            Root: Element,
            Header: Element,
            Row: Element,
            ColumnHeaderCell: Element,
            Body: Element,
            Cell: Element,
        },
    };
});
jest.mock('@radix-ui/react-icons', () => ({ Cross2Icon: () => null }));
jest.mock('services/api.service', () => ({
    __esModule: true,
    default: { post: jest.fn() },
}));

const mockPost = apiService.post as jest.Mock;
const componentName = 'visualization:imitation-guidance';

const renderChart = (data?: any) => {
    const ref = createRef<ImitationGuidanceChartHandle>();
    render(
        <AiToolComponentRefProvider>
            <ImitationGuidanceChart
                ref={ref}
                id="guidance-1"
                name={componentName}
                data={data}
                onUpdate={() => true}
                onDisable={() => true}
            />
        </AiToolComponentRefProvider>,
    );
    return ref;
};

describe('ImitationGuidanceChart component failures', () => {
    beforeEach(() => {
        mockPost.mockReset();
    });

    it('throws a typed control error and renders it when telemetry is missing', async () => {
        const ref = renderChart();
        let thrown: unknown;

        await act(async () => {
            try {
                await ref.current!.refreshGuidanceOnce();
            } catch (error) {
                thrown = error;
            }
        });

        expect(thrown).toBeInstanceOf(VisualizationControlFailedError);
        expect(thrown).toMatchObject({
            name: 'VisualizationControlFailedError',
            componentName,
            message: 'Guidance refresh requires telemetry data.',
        });
        expect(screen.getByText('Guidance refresh requires telemetry data.')).toBeInTheDocument();
        expect(mockPost).not.toHaveBeenCalled();
    });

    it('wraps guidance request failures and retains the rendered error', async () => {
        const consoleError = jest.spyOn(console, 'error').mockImplementation(() => undefined);
        const ref = renderChart({ telemetry: { Physics_speed_kmh: 180 } });
        const cause = new Error('Guidance service unavailable.');
        mockPost.mockRejectedValueOnce(cause);
        let thrown: unknown;

        await act(async () => {
            try {
                await ref.current!.refreshGuidanceOnce();
            } catch (error) {
                thrown = error;
            }
        });

        expect(thrown).toBeInstanceOf(VisualizationControlFailedError);
        expect(thrown).toMatchObject({
            name: 'VisualizationControlFailedError',
            componentName,
            message: 'Guidance service unavailable.',
            cause,
        });
        expect(screen.getByText('Guidance service unavailable.')).toBeInTheDocument();
        consoleError.mockRestore();
    });
});
