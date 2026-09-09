import React, { useContext } from 'react';
import { act, render } from '@testing-library/react';
import { LiveSessionContext, LiveSessionProvider } from '../LiveSessionContext';
import { createLiveSessionAnalysisResultPage } from '../live-session-analysis-results';
import {
    getPersistedLiveSessionAnalysis,
    LIVE_SESSION_ANALYSIS_STORAGE_KEY,
    savePersistedLiveSessionAnalysis,
} from '../live-session-analysis-storage';

const ownerEmail = 'driver@example.com';
const storageKey = `${LIVE_SESSION_ANALYSIS_STORAGE_KEY}:${ownerEmail}`;
const pageInput = {
    baseline: {
        id: 'baseline-1',
        lap_id: 1,
        lap_time_ms: null,
        captured_at: 123,
        track: 'Monza',
        car: 'GT3',
        sample_count: 42,
    },
    elements: [{
        id: 'result-1',
        labels: ['MSP'],
        normalizedPositionRange: { start: 0.2, end: 0.3 },
        timeGap: { deltaMs: 250 },
        comparison: { samples: [{
            driverTimeMs: 500,
            expertTimeMs: 450,
            driverTrackPosition: 0.2,
            expertTrackPosition: 0.2,
            driverTrajectory: { x: 1, z: 2 },
            expertTrajectory: { x: 2, z: 3 },
            driverBrake: 0.8,
            expertBrake: 0.9,
        }] },
        metadata: { source: 'ai_classifier', start_index: 1, end_index: 5 },
    }],
};

let runtime: React.ContextType<typeof LiveSessionContext>;
const Probe = () => {
    runtime = useContext(LiveSessionContext);
    return null;
};
const provider = (owner: string | null = ownerEmail) => (
    <LiveSessionProvider ownerEmail={owner}><Probe /></LiveSessionProvider>
);

describe('local analysis history', () => {
    beforeEach(() => {
        localStorage.clear();
        Object.defineProperty(window, 'electronAPI', { configurable: true, value: undefined });
    });

    afterEach(() => jest.restoreAllMocks());

    it('saves immediately and restores ordered pages, metadata, edits, and selection after a full remount', () => {
        const { unmount } = render(provider(' Driver@Example.com '));
        let selectedId = '';
        act(() => {
            runtime.appendAnalysisResultPage(pageInput);
            selectedId = runtime.appendAnalysisResultPage({
                ...pageInput,
                baseline: { ...pageInput.baseline, id: 'baseline-2', lap_id: 2 },
            }).pageId;
            runtime.selectAnalysisResultPage(selectedId);
            runtime.updateActiveAnalysisResultPage({
                elements: [{ ...pageInput.elements[0], id: 'edited-result', title: 'Brake earlier' }],
            });
            expect(getPersistedLiveSessionAnalysis(ownerEmail).activePageId).toBe(selectedId);
        });
        const expectedPages = runtime.analysisResultPages;
        unmount();

        render(<React.StrictMode>{provider()}</React.StrictMode>);
        expect(runtime.analysisResultPages).toEqual(expectedPages);
        expect(runtime.analysisResultPages.map((page) => page.baseline.lap_id)).toEqual([1, 2]);
        expect(runtime.analysisResultPages[0].baseline.lap_time_ms).toBeNull();
        expect(runtime.analysisResultPages[1].elements[0].title).toBe('Brake earlier');
        expect(runtime.analysisResultPages[0].elements[0].comparison).toEqual(pageInput.elements[0].comparison);
        expect(runtime.activeAnalysisResultPageId).toBe(selectedId);
    });

    it('retains history through recording cleanup and clears it when another session starts', () => {
        const { unmount } = render(provider());
        act(() => {
            runtime.startLiveSession('acc');
            runtime.appendAnalysisResultPage(pageInput);
            const { pageId } = runtime.appendAnalysisResultPage({
                ...pageInput,
                baseline: { ...pageInput.baseline, id: 'baseline-2', lap_id: 2 },
            });
            runtime.selectAnalysisResultPage(pageId);
        });
        const expectedPages = runtime.analysisResultPages;
        const selectedId = runtime.activeAnalysisResultPageId;
        act(() => { runtime.startLiveSession('acc'); });
        expect(runtime.analysisResultPages).toEqual(expectedPages);
        expect(runtime.activeAnalysisResultPageId).toBe(selectedId);

        act(() => {
            runtime.clearPersistedDraft();
            runtime.clearRecordingSession();
            runtime.endLiveSession();
        });
        expect(runtime.analysisResultPages).toEqual(expectedPages);
        expect(runtime.activeAnalysisResultPageId).toBe(selectedId);

        act(() => { runtime.startLiveSession('acc'); });
        expect(runtime.analysisResultPages).toEqual([]);
        expect(runtime.activeAnalysisResultPageId).toBeNull();
        expect(getPersistedLiveSessionAnalysis(ownerEmail)).toEqual({ pages: [], activePageId: null });
        unmount();

        render(provider());
        expect(runtime.analysisResultPages).toEqual([]);
        expect(runtime.activeAnalysisResultPageId).toBeNull();
    });

    it('clears restored analysis only for the account starting a new session', () => {
        const page = createLiveSessionAnalysisResultPage(pageInput);
        const saved = { pages: [page], activePageId: page.id };
        savePersistedLiveSessionAnalysis(ownerEmail, saved);
        savePersistedLiveSessionAnalysis('second@example.com', saved);
        render(provider());
        expect(runtime.analysisResultPages).toEqual([page]);

        act(() => { runtime.startLiveSession('iracing'); });
        expect(runtime.analysisResultPages).toEqual([]);
        expect(runtime.activeAnalysisResultPageId).toBeNull();
        expect(getPersistedLiveSessionAnalysis(ownerEmail)).toEqual({ pages: [], activePageId: null });
        expect(getPersistedLiveSessionAnalysis('second@example.com')).toEqual(saved);

        let newPageId = '';
        act(() => {
            const result = runtime.appendAnalysisResultPage(pageInput);
            newPageId = result.pageId;
            expect(result.pageCount).toBe(1);
        });
        expect(runtime.analysisResultPages).toHaveLength(1);
        expect(runtime.activeAnalysisResultPageId).toBe(newPageId);
    });

    it('isolates accounts and restores each history after logout and switching accounts', () => {
        const view = render(provider());
        act(() => { runtime.appendAnalysisResultPage(pageInput); });
        const firstPages = runtime.analysisResultPages;

        view.rerender(provider('second@example.com'));
        expect(runtime.analysisResultPages).toEqual([]);
        act(() => {
            runtime.appendAnalysisResultPage({ ...pageInput, elements: [{ id: 'second-user', labels: [] }] });
        });
        const secondPages = runtime.analysisResultPages;
        expect(getPersistedLiveSessionAnalysis(ownerEmail).pages).toEqual(firstPages);

        view.rerender(provider(null));
        expect(runtime.analysisResultPages).toEqual([]);
        expect(runtime.activeAnalysisResultPageId).toBeNull();
        view.rerender(provider('DRIVER@example.com'));
        expect(runtime.analysisResultPages).toEqual(firstPages);
        view.rerender(provider('second@example.com'));
        expect(runtime.analysisResultPages).toEqual(secondPages);
    });

    it.each(['invalid json', 'null', '{"version":99,"pages":[]}', '{"version":1,"pages":{}}'])(
        'opens safely with invalid stored history: %s', (raw) => {
            localStorage.setItem(storageKey, raw);
            render(provider());
            expect(runtime.analysisResultPages).toEqual([]);
            expect(runtime.activeAnalysisResultPageId).toBeNull();
            // Loading must not overwrite data before the user changes anything.
            expect(localStorage.getItem(storageKey)).toBe(raw);
        },
    );

    it.each([null, undefined, 'missing-page', 'older-page'])('restores the selected/latest valid page when stored selection is %s', (activePageId) => {
        const olderPage = { ...createLiveSessionAnalysisResultPage(pageInput), id: 'older-page' };
        const page = createLiveSessionAnalysisResultPage(pageInput);
        localStorage.setItem(storageKey, JSON.stringify({
            version: 1,
            pages: [null, olderPage, { id: 'broken' }, page, page, { id: 'invalid-latest' }],
            activePageId,
        }));
        render(provider());
        const expectedPageId = activePageId === olderPage.id ? olderPage.id : page.id;
        expect(runtime.analysisResultPages).toEqual([olderPage, page]);
        expect(runtime.activeAnalysisResultPageId).toBe(expectedPageId);
        act(() => { expect(runtime.selectAnalysisResultPage('unknown')).toBe(false); });
        expect(runtime.activeAnalysisResultPageId).toBe(expectedPageId);
    });

    it('keeps the last saved history and usable in-memory results if storage is full', () => {
        render(provider());
        act(() => { runtime.appendAnalysisResultPage(pageInput); });
        const saved = localStorage.getItem(storageKey);
        const warning = jest.spyOn(console, 'warn').mockImplementation(() => undefined);
        jest.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
            throw new DOMException('Storage full', 'QuotaExceededError');
        });
        act(() => { runtime.appendAnalysisResultPage(pageInput); });
        expect(runtime.analysisResultPages).toHaveLength(2);
        expect(localStorage.getItem(storageKey)).toBe(saved);
        expect(warning).toHaveBeenCalledWith('Unable to save analysis results locally', expect.any(Error));
    });

    it('keeps working if access to local storage is disabled', () => {
        const warning = jest.spyOn(console, 'warn').mockImplementation(() => undefined);
        jest.spyOn(window, 'localStorage', 'get').mockImplementation(() => {
            throw new DOMException('Storage disabled', 'SecurityError');
        });
        expect(getPersistedLiveSessionAnalysis(ownerEmail)).toEqual({ pages: [], activePageId: null });
        expect(() => savePersistedLiveSessionAnalysis(ownerEmail, { pages: [], activePageId: null })).not.toThrow();
        expect(warning).toHaveBeenCalled();
    });
});
