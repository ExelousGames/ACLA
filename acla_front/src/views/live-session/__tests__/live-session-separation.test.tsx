import React, { useContext, useState } from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { LiveSessionContext, LiveSessionProvider } from '../LiveSessionContext';
import { AnalysisContext } from 'views/lap-analysis/analysis-context';
import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import type { RecordingViewUpdate, StandardTelemetrySample } from '../live-session-types';

let recordingViewHandler: ((update: RecordingViewUpdate) => void) | null = null;
let recordingViewSequence = 0;

const publishRecordingViewSample = (sample: StandardTelemetrySample) => {
    recordingViewSequence += 1;
    recordingViewHandler?.({
        type: 'frame',
        game: 'acc',
        sample,
        sequence: recordingViewSequence,
        committedSequence: recordingViewSequence,
        committedCount: recordingViewSequence,
    });
};

const RecordedSelectionProvider = ({ children }: { children: React.ReactNode }) => {
    const [mapSelected, setMap] = useState<string | null>(null);
    return (
        <AnalysisContext.Provider value={{ mapSelected, setMap } as any}>
            {children}
        </AnalysisContext.Provider>
    );
};

let capturedLiveSession: React.ContextType<typeof LiveSessionContext>;
const LiveSessionCapture = () => {
    capturedLiveSession = useContext(LiveSessionContext);
    return null;
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
                publishRecordingViewSample({
                    Graphics_status: ACC_STATUS.ACC_LIVE,
                    Graphics_normalized_car_position: 0.1,
                    Static_track: 'Monza',
                    Static_car_model: 'GT3',
                    speed: 120,
                });
                live.setRecordingMetadata({ sessionName: 'Live Run', mapName: 'Monza', carName: 'GT3', gameRecordedFrom: 'acc' });
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
            <output data-testid="telemetry-speed">{String(live.currentTelemetry.speed || 'none')}</output>
            <output data-testid="next-corner">{live.getNextCorner()?.name || 'none'}</output>
            <output data-testid="static-track">{live.staticData.Static_track || 'none'}</output>
            <output data-testid="recording-state">{live.recordingState}</output>
        </>
    );
};

describe('live session state separation', () => {
    beforeEach(() => {
        recordingViewHandler = null;
        recordingViewSequence = 0;
        Object.defineProperty(window, 'electronAPI', {
            configurable: true,
            value: {
                onRecordingViewUpdate: jest.fn((handler) => {
                    recordingViewHandler = handler;
                    return jest.fn();
                }),
                onRecordingSessionEnded: jest.fn().mockReturnValue(jest.fn()),
            },
        });
    });

    it('derives snapshot and corner data from only the latest telemetry sample', () => {
        render(
            <LiveSessionProvider>
                <LiveSessionCapture />
            </LiveSessionProvider>,
        );

        act(() => {
            capturedLiveSession.startLiveSession('acc');
            [1, 2, 3].forEach((lap) => publishRecordingViewSample({
                Static_track: 'monza',
                Static_car_model: 'Ferrari 296',
                Static_num_cars: 1,
                Graphics_completed_laps: lap,
                Graphics_normalized_car_position: 0.1,
            }));
        });

        expect('telemetry' in capturedLiveSession).toBe(false);
        expect(capturedLiveSession.getLiveSessionSnapshot()).toEqual({
            status: 'ready',
            track: 'monza',
            car: 'Ferrari 296',
            current_lap: 3,
            completed_laps: 3,
            normalized_position: 0.1,
            sample_count: 3,
            live_session_type: 'solo_practice',
            completed_lap_count: 3,
        });
        expect(capturedLiveSession.getNextCorner()).toMatchObject({
            name: 'T2 Seconda Variante',
            trackPosition: 0.18,
        });
    });

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

    it('clears telemetry, static data, metadata, and recording state on full reset', () => {
        render(
            <LiveSessionProvider>
                <RecordedSelectionProvider>
                    <SeparationHarness />
                </RecordedSelectionProvider>
            </LiveSessionProvider>,
        );

        fireEvent.click(screen.getByText('Capture ACC'));
        fireEvent.click(screen.getByText('Populate live state'));
        expect(screen.getByTestId('telemetry-speed')).toHaveTextContent('120');
        expect(screen.getByTestId('next-corner')).toHaveTextContent('T2 Seconda Variante');
        expect(screen.getByTestId('static-track')).toHaveTextContent('Monza');

        fireEvent.click(screen.getByText('Reset live recording'));

        expect(screen.getByTestId('session-game')).toHaveTextContent('none');
        expect(screen.getByTestId('live-session-name')).toHaveTextContent('none');
        expect(screen.getByTestId('telemetry-speed')).toHaveTextContent('none');
        expect(screen.getByTestId('next-corner')).toHaveTextContent('none');
        expect(screen.getByTestId('static-track')).toHaveTextContent('none');
        expect(screen.getByTestId('recording-state')).toHaveTextContent('CHECKING');
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
