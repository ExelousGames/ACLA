import { useContext, useState } from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Theme } from '@radix-ui/themes';
import { AnalysisContext } from '../analysis-context';
import type { RacingSessionDetailedInfoDto } from 'data/live-analysis/live-analysis-type';
import type { RecordedFileReadEvent } from 'views/live-session/live-session-types';
import LocalIRacingTelemetry from './LocalIRacingTelemetry';
import { readLocalTelemetry } from './read-local-telemetry';

jest.mock('radix-ui/internal', () => jest.requireActual('radix-ui/dist/internal.js'), { virtual: true });
const mockEnvironment = jest.fn(() => 'electron');
jest.mock('contexts/EnvironmentContext', () => ({ useEnvironment: () => mockEnvironment() }));
jest.mock('../sessionAnalysis/session-analysis-split', () => () => <div>Local playback workspace</div>);

const imported = { filePath: 'C:\\temp\\local.jsonl', fileName: 'spa.ibt', track: 'Spa', car: 'GT3', rowCount: 2 };
const rows = [{ Physics_speed_kmh: 180 }, { Physics_speed_kmh: 190 }];
let listener: (event: RecordedFileReadEvent) => void;
let selected: RacingSessionDetailedInfoDto | null;
const unsubscribe = jest.fn();
const Harness = () => {
    const defaults = useContext(AnalysisContext);
    const [sessionSelected, setSession] = useState<RacingSessionDetailedInfoDto | null>(null);
    selected = sessionSelected;
    return <Theme><AnalysisContext.Provider value={{ ...defaults, sessionSelected, setSession, setMap: jest.fn() }}>
        <LocalIRacingTelemetry />
    </AnalysisContext.Provider></Theme>;
};

beforeEach(() => {
    mockEnvironment.mockReturnValue('electron');
    unsubscribe.mockClear();
    window.electronAPI = {
        importLocalIRacingTelemetry: jest.fn().mockResolvedValue(imported),
        onRecordedFileReadEvent: jest.fn((callback) => { listener = callback; return unsubscribe; }),
        startRecordedFileRead: jest.fn().mockResolvedValue({ readId: 'read-1' }),
        cancelRecordedFileRead: jest.fn().mockResolvedValue(undefined),
        deleteTempFile: jest.fn().mockResolvedValue({ success: true }),
    } as any;
});

function completeRead() {
    listener({ type: 'chunk', readId: 'read-1', rows });
    listener({ type: 'complete', readId: 'read-1', game: 'iracing', format: 'standard-flat', rowCount: 2, totalBytes: 200 });
}

it('opens local samples in the recorded workspace and removes only the converted temporary file', async () => {
    render(<Harness />);
    fireEvent.click(screen.getByRole('button', { name: 'Open .ibt file' }));
    await waitFor(() => expect(window.electronAPI.startRecordedFileRead).toHaveBeenCalledWith({ filePath: imported.filePath, game: 'iracing', purpose: 'consume' }));
    await act(async () => completeRead());
    expect(screen.getByText('Local playback workspace')).toBeInTheDocument();
    expect(selected).toMatchObject({ storage: 'local', game_recorded_from: 'iracing_recorded', session_name: 'spa.ibt', data: rows });
    expect(window.electronAPI.deleteTempFile).toHaveBeenCalledTimes(1);
    expect(window.electronAPI.deleteTempFile).toHaveBeenCalledWith(imported.filePath);
    expect(unsubscribe).toHaveBeenCalled();
});

it('leaves the workspace empty when the file picker is cancelled', async () => {
    (window.electronAPI.importLocalIRacingTelemetry as jest.Mock).mockResolvedValue(null);
    render(<Harness />);
    fireEvent.click(screen.getByRole('button', { name: 'Open .ibt file' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Open .ibt file' })).toBeEnabled());
    expect(selected).toBeNull();
    expect(window.electronAPI.startRecordedFileRead).not.toHaveBeenCalled();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
});

it('reports invalid files and allows retrying', async () => {
    (window.electronAPI.importLocalIRacingTelemetry as jest.Mock).mockRejectedValueOnce(new Error('Invalid iRacing .ibt header.'));
    render(<Harness />);
    fireEvent.click(screen.getByRole('button', { name: 'Open .ibt file' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Invalid iRacing .ibt header.');
    expect(screen.getByRole('button', { name: 'Open .ibt file' })).toBeEnabled();
});

it('explains desktop availability on the web', () => {
    mockEnvironment.mockReturnValue('web');
    render(<Harness />);
    expect(screen.getByText('Open the desktop app to analyze local iRacing .ibt files.')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Open .ibt file' })).not.toBeInTheDocument();
});

it('cleans up an import that finishes after leaving the subsection', async () => {
    let finishImport!: (value: typeof imported) => void;
    (window.electronAPI.importLocalIRacingTelemetry as jest.Mock).mockImplementation(() => new Promise((resolve) => { finishImport = resolve; }));
    const view = render(<Harness />);
    fireEvent.click(screen.getByRole('button', { name: 'Open .ibt file' }));
    view.unmount();
    await act(async () => { finishImport(imported); });
    expect(window.electronAPI.startRecordedFileRead).not.toHaveBeenCalled();
    expect(window.electronAPI.deleteTempFile).toHaveBeenCalledWith(imported.filePath);
});

it('accepts early read events and ignores unrelated reads', async () => {
    (window.electronAPI.startRecordedFileRead as jest.Mock).mockImplementation(async () => {
        listener({ type: 'chunk', readId: 'unrelated', rows: [{ Physics_speed_kmh: 1 }] });
        completeRead();
        return { readId: 'read-1' };
    });
    await expect(readLocalTelemetry(imported.filePath, new AbortController().signal, jest.fn())).resolves.toEqual(rows);
    expect(unsubscribe).toHaveBeenCalledTimes(1);
});

it('cancels an active read when leaving local analysis', async () => {
    const controller = new AbortController();
    const pending = readLocalTelemetry(imported.filePath, controller.signal, jest.fn());
    await Promise.resolve();
    controller.abort();
    await expect(pending).rejects.toThrow('cancelled');
    expect(window.electronAPI.cancelRecordedFileRead).toHaveBeenCalledWith('read-1');
    expect(unsubscribe).toHaveBeenCalledTimes(1);
});
