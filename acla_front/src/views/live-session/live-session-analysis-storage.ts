import { normalizeAnalysisResultsData } from 'views/lap-analysis/visualization/charts/analysisResultsModel';
import { LiveSessionAnalysisResultPage } from './live-session-analysis-results';
import { normalizeLiveSessionOwnerEmail } from './live-session-draft-storage';

export const LIVE_SESSION_ANALYSIS_STORAGE_KEY = 'acla.live-session-analysis';
const STORAGE_VERSION = 1;

export interface PersistedLiveSessionAnalysis {
    pages: LiveSessionAnalysisResultPage[];
    activePageId: string | null;
}

const emptyAnalysis = (): PersistedLiveSessionAnalysis => ({ pages: [], activePageId: null });

const isRecord = (value: unknown): value is Record<string, unknown> => (
    Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

const isPage = (value: unknown): value is LiveSessionAnalysisResultPage => {
    if (!isRecord(value) || !isRecord(value.baseline)) return false;
    const baseline = value.baseline;
    return typeof value.id === 'string' && value.id.length > 0
        && typeof value.createdAt === 'number' && Number.isFinite(value.createdAt)
        && Array.isArray(value.elements)
        && typeof baseline.id === 'string'
        && typeof baseline.lap_id === 'number' && Number.isFinite(baseline.lap_id)
        && (baseline.lap_time_ms === null || (
            typeof baseline.lap_time_ms === 'number' && Number.isFinite(baseline.lap_time_ms)
        ))
        && typeof baseline.captured_at === 'number' && Number.isFinite(baseline.captured_at)
        && typeof baseline.track === 'string'
        && typeof baseline.car === 'string'
        && typeof baseline.sample_count === 'number' && Number.isFinite(baseline.sample_count);
};

export const getPersistedLiveSessionAnalysis = (
    ownerEmail?: string | null,
): PersistedLiveSessionAnalysis => {
    const owner = normalizeLiveSessionOwnerEmail(ownerEmail);
    if (!owner) return emptyAnalysis();
    try {
        const raw = window.localStorage.getItem(`${LIVE_SESSION_ANALYSIS_STORAGE_KEY}:${owner}`);
        if (!raw) return emptyAnalysis();
        const parsed: unknown = JSON.parse(raw);
        if (!isRecord(parsed) || parsed.version !== STORAGE_VERSION || !Array.isArray(parsed.pages)) {
            return emptyAnalysis();
        }
        const seenIds = new Set<string>();
        const pages = parsed.pages.filter(isPage).filter((page) => {
            if (seenIds.has(page.id)) return false;
            seenIds.add(page.id);
            return true;
        }).map((page) => ({
            ...page,
            elements: normalizeAnalysisResultsData(page.elements).elements,
        }));
        const activePageId = pages.find((page) => page.id === parsed.activePageId)?.id
            ?? pages[pages.length - 1]?.id ?? null;
        return { pages, activePageId };
    } catch {
        return emptyAnalysis();
    }
};

export const savePersistedLiveSessionAnalysis = (
    ownerEmail: string | null | undefined,
    analysis: PersistedLiveSessionAnalysis,
): void => {
    const owner = normalizeLiveSessionOwnerEmail(ownerEmail);
    if (!owner) return;
    try {
        window.localStorage.setItem(
            `${LIVE_SESSION_ANALYSIS_STORAGE_KEY}:${owner}`,
            JSON.stringify({ version: STORAGE_VERSION, ...analysis }),
        );
    } catch (error) {
        console.warn('Unable to save analysis results locally', error);
    }
};
