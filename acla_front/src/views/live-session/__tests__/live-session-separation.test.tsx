import React, { useContext, useState } from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { LiveSessionContext, LiveSessionProvider } from '../LiveSessionContext';
import { AnalysisContext } from 'views/lap-analysis/analysis-context';

const RecordedSelectionProvider = ({ children }: { children: React.ReactNode }) => {
    const [mapSelected, setMap] = useState<string | null>(null);
    return (
        <AnalysisContext.Provider value={{ mapSelected, setMap } as any}>
            {children}
        </AnalysisContext.Provider>
    );
};

const SeparationHarness = () => {
    const recorded = useContext(AnalysisContext);
    const live = useContext(LiveSessionContext);
    const [showLivePanel, setShowLivePanel] = useState(true);

    return (
        <>
            <button onClick={() => recorded.setMap('Recorded Spa')}>Select recorded map</button>
            <button onClick={() => live.setRecordingMetadata({ sessionName: 'Live Run', mapName: 'Monza', carName: 'GT3' })}>Start live metadata</button>
            <button onClick={live.clearRecordingSession}>Reset live recording</button>
            <button onClick={() => setShowLivePanel((visible) => !visible)}>Toggle live panel</button>
            <output data-testid="recorded-map">{recorded.mapSelected || 'none'}</output>
            {showLivePanel ? <output data-testid="live-session-name">{live.recordingMetadata?.sessionName || 'none'}</output> : null}
        </>
    );
};

describe('live session state separation', () => {
    it('never changes recorded Analysis selection when live metadata starts or resets', () => {
        render(
            <LiveSessionProvider>
                <RecordedSelectionProvider>
                    <SeparationHarness />
                </RecordedSelectionProvider>
            </LiveSessionProvider>,
        );

        fireEvent.click(screen.getByText('Select recorded map'));
        fireEvent.click(screen.getByText('Start live metadata'));
        expect(screen.getByTestId('recorded-map')).toHaveTextContent('Recorded Spa');
        expect(screen.getByTestId('live-session-name')).toHaveTextContent('Live Run');

        fireEvent.click(screen.getByText('Reset live recording'));
        expect(screen.getByTestId('recorded-map')).toHaveTextContent('Recorded Spa');
        expect(screen.getByTestId('live-session-name')).toHaveTextContent('none');
    });

    it('retains live runtime state while the visible live panel is unmounted', () => {
        render(
            <LiveSessionProvider>
                <RecordedSelectionProvider>
                    <SeparationHarness />
                </RecordedSelectionProvider>
            </LiveSessionProvider>,
        );

        fireEvent.click(screen.getByText('Start live metadata'));
        fireEvent.click(screen.getByText('Toggle live panel'));
        expect(screen.queryByTestId('live-session-name')).not.toBeInTheDocument();
        fireEvent.click(screen.getByText('Toggle live panel'));
        expect(screen.getByTestId('live-session-name')).toHaveTextContent('Live Run');
    });
});
