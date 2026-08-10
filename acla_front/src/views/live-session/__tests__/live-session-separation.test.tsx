import React, { useContext, useState } from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { LiveSessionContext, LiveSessionProvider } from '../LiveSessionContext';
import { AnalysisContext } from 'views/lap-analysis/analysis-context';
import { ACC_STATUS } from 'data/live-analysis/live-map-data';

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
            <button onClick={() => live.startLiveSession('acc')}>Capture ACC</button>
            <button onClick={() => live.startLiveSession('iracing')}>Capture iRacing</button>
            <button onClick={() => live.setRecordingMetadata({ sessionName: 'Live Run', mapName: 'Monza', carName: 'GT3', gameRecordedFrom: 'acc' })}>Start live metadata</button>
            <button onClick={() => {
                live.setCurrentTelemetry({ Graphics_status: ACC_STATUS.ACC_LIVE, speed: 120 });
                live.setStaticData({ track: 'Monza', car_model: 'GT3' });
                live.setRecordingMetadata({ sessionName: 'Live Run', mapName: 'Monza', carName: 'GT3', gameRecordedFrom: 'acc' });
                live.publishLiveRangeTodoListSnapshot({ items: [{ id: 'todo-1' }] } as any);
                live.transitionRecordingState({ type: 'sessionAvailable' });
            }}>Populate live state</button>
            <button onClick={live.endLiveSession}>Reset live recording</button>
            <button onClick={() => {
                const pageNumber = live.analysisResultPages.length + 1;
                live.appendAnalysisResultPage({
                    baseline: {
                        id: `baseline-${pageNumber}`,
                        lap: pageNumber,
                        lap_time_ms: pageNumber * 90_000,
                        captured_at: pageNumber,
                        track: 'Monza',
                        car: 'GT3',
                        sample_count: pageNumber,
                    },
                    elements: [{ id: `result-${pageNumber}`, labels: ['MSP'] }],
                });
            }}>Append result page</button>
            <button onClick={() => {
                const lastPage = live.analysisResultPages[live.analysisResultPages.length - 1];
                if (lastPage) live.selectAnalysisResultPage(lastPage.id);
            }}>Select last page</button>
            <button onClick={() => setShowLivePanel((visible) => !visible)}>Toggle live panel</button>
            <output data-testid="recorded-map">{recorded.mapSelected || 'none'}</output>
            <output data-testid="session-game">{live.sessionGame || 'none'}</output>
            {showLivePanel ? (
                <>
                    <output data-testid="live-session-name">{live.recordingMetadata?.sessionName || 'none'}</output>
                    <output data-testid="analysis-page-count">{live.analysisResultPages.length}</output>
                    <output data-testid="active-analysis-result">{
                        live.analysisResultPages
                            .find((page) => page.id === live.activeAnalysisResultPageId)
                            ?.elements[0]?.id ?? 'none'
                    }</output>
                </>
            ) : null}
            <output data-testid="telemetry-speed">{live.currentTelemetry.speed || 'none'}</output>
            <output data-testid="static-track">{live.staticData.track || 'none'}</output>
            <output data-testid="todo-snapshot">{live.liveRangeTodoListSnapshot ? 'present' : 'none'}</output>
            <output data-testid="recording-state">{live.recordingState}</output>
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
        fireEvent.click(screen.getByText('Capture ACC'));
        fireEvent.click(screen.getByText('Start live metadata'));
        expect(screen.getByTestId('recorded-map')).toHaveTextContent('Recorded Spa');
        expect(screen.getByTestId('live-session-name')).toHaveTextContent('Live Run');

        fireEvent.click(screen.getByText('Reset live recording'));
        expect(screen.getByTestId('recorded-map')).toHaveTextContent('Recorded Spa');
        expect(screen.getByTestId('live-session-name')).toHaveTextContent('none');
        expect(screen.getByTestId('session-game')).toHaveTextContent('none');
    });

    it('retains live runtime state while the visible live panel is unmounted', () => {
        render(
            <LiveSessionProvider>
                <RecordedSelectionProvider>
                    <SeparationHarness />
                </RecordedSelectionProvider>
            </LiveSessionProvider>,
        );

        fireEvent.click(screen.getByText('Capture ACC'));
        fireEvent.click(screen.getByText('Start live metadata'));
        fireEvent.click(screen.getByText('Toggle live panel'));
        expect(screen.queryByTestId('live-session-name')).not.toBeInTheDocument();
        fireEvent.click(screen.getByText('Toggle live panel'));
        expect(screen.getByTestId('live-session-name')).toHaveTextContent('Live Run');
        expect(screen.getByTestId('session-game')).toHaveTextContent('acc');
    });

    it('appends pages chronologically, retains selection across remounts, and clears them on reset', () => {
        render(
            <LiveSessionProvider>
                <RecordedSelectionProvider>
                    <SeparationHarness />
                </RecordedSelectionProvider>
            </LiveSessionProvider>,
        );

        fireEvent.click(screen.getByText('Capture ACC'));
        fireEvent.click(screen.getByText('Append result page'));
        expect(screen.getByTestId('analysis-page-count')).toHaveTextContent('1');
        expect(screen.getByTestId('active-analysis-result')).toHaveTextContent('result-1');

        fireEvent.click(screen.getByText('Append result page'));
        expect(screen.getByTestId('analysis-page-count')).toHaveTextContent('2');
        expect(screen.getByTestId('active-analysis-result')).toHaveTextContent('result-1');

        fireEvent.click(screen.getByText('Select last page'));
        expect(screen.getByTestId('active-analysis-result')).toHaveTextContent('result-2');
        fireEvent.click(screen.getByText('Toggle live panel'));
        fireEvent.click(screen.getByText('Toggle live panel'));
        expect(screen.getByTestId('analysis-page-count')).toHaveTextContent('2');
        expect(screen.getByTestId('active-analysis-result')).toHaveTextContent('result-2');

        fireEvent.click(screen.getByText('Reset live recording'));
        expect(screen.getByTestId('analysis-page-count')).toHaveTextContent('0');
        expect(screen.getByTestId('active-analysis-result')).toHaveTextContent('none');
    });

    it('does not let a later start action replace the captured game', () => {
        render(
            <LiveSessionProvider>
                <RecordedSelectionProvider>
                    <SeparationHarness />
                </RecordedSelectionProvider>
            </LiveSessionProvider>,
        );

        fireEvent.click(screen.getByText('Capture ACC'));
        fireEvent.click(screen.getByText('Capture iRacing'));

        expect(screen.getByTestId('session-game')).toHaveTextContent('acc');
    });

    it('clears telemetry, static data, metadata, UI snapshots, and recording state on full reset', () => {
        jest.useFakeTimers();
        render(
            <LiveSessionProvider>
                <RecordedSelectionProvider>
                    <SeparationHarness />
                </RecordedSelectionProvider>
            </LiveSessionProvider>,
        );

        fireEvent.click(screen.getByText('Capture ACC'));
        fireEvent.click(screen.getByText('Populate live state'));
        act(() => jest.advanceTimersByTime(150));
        expect(screen.getByTestId('telemetry-speed')).toHaveTextContent('120');
        expect(screen.getByTestId('static-track')).toHaveTextContent('Monza');
        expect(screen.getByTestId('todo-snapshot')).toHaveTextContent('present');

        fireEvent.click(screen.getByText('Reset live recording'));

        expect(screen.getByTestId('session-game')).toHaveTextContent('none');
        expect(screen.getByTestId('live-session-name')).toHaveTextContent('none');
        expect(screen.getByTestId('telemetry-speed')).toHaveTextContent('none');
        expect(screen.getByTestId('static-track')).toHaveTextContent('none');
        expect(screen.getByTestId('todo-snapshot')).toHaveTextContent('none');
        expect(screen.getByTestId('recording-state')).toHaveTextContent('CHECKING');
        jest.useRealTimers();
    });

    it('does not persist the selected game after the provider unmounts', () => {
        const first = render(
            <LiveSessionProvider>
                <RecordedSelectionProvider>
                    <SeparationHarness />
                </RecordedSelectionProvider>
            </LiveSessionProvider>,
        );
        fireEvent.click(screen.getByText('Capture ACC'));
        fireEvent.click(screen.getByText('Append result page'));
        expect(screen.getByTestId('session-game')).toHaveTextContent('acc');
        expect(screen.getByTestId('analysis-page-count')).toHaveTextContent('1');
        first.unmount();

        render(
            <LiveSessionProvider>
                <RecordedSelectionProvider>
                    <SeparationHarness />
                </RecordedSelectionProvider>
            </LiveSessionProvider>,
        );
        expect(screen.getByTestId('session-game')).toHaveTextContent('none');
        expect(screen.getByTestId('analysis-page-count')).toHaveTextContent('0');
    });
});
