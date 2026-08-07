import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import '../lap-analysis/ai-chat/ai-chat.css';
import './floating-chat.css';
import AiMapToolDisplay, { type AiMapDisplayPayload } from 'views/lap-analysis/ai-chat/AiMapToolDisplay';
import BaselineProgressDisplay from 'views/lap-analysis/ai-chat/BaselineProgressDisplay';
import ProcedurePlanDisplay from 'views/lap-analysis/ai-chat/ProcedurePlanDisplay';
import ToolMessageDisplay, { type ToolMessageDisplayData } from 'views/lap-analysis/ai-chat/ToolMessageDisplay';
import {
    FLOATING_PILL_COMPARISON_COMPLETION_PAUSE_MS,
    FLOATING_PILL_RICH_CONTENT_HOLD_MS,
    FLOATING_PILL_STORAGE_KEY,
    type FloatingPillPayloadKind,
} from './floating-pill-bridge';
import {
    DriverExpertComparisonData,
    DriverExpertComparisonGraph,
    getDriverExpertReplayDurationMs,
    normalizeDriverExpertComparisonData,
} from 'components/driver-expert-comparison';
import { LiveRangeTodoListDisplay } from 'views/live-session/LiveRangeTodoList';
import type { LiveRangeTodoListSnapshot } from 'views/live-session/live-range-todo-list-types';
import type { BaselineCollectionTag } from 'views/lap-analysis/ai-chat/BaselineCollectionTracker';
import type { ProcedurePlan } from 'views/lap-analysis/ai-chat/ai-chat-plan';

/**
 * AI Chat Pill — ambient overlay for the always-on-top Electron window.
 *
 * Idle: a small pulsing circle showing the "AI" avatar.
 * Active: expands horizontally and types out Kestrel's latest reply, then
 *         auto-collapses back to the circle.
 *
 * Voice is started from the main application — this window is read-only.
 * The pill subscribes to assistant transcripts via a shared localStorage key
 * (`storage` events fire across all same-origin BrowserWindows).
 */

const EMOTION_GIFS_KEY = 'acla-emotion-gifs';
const TYPE_INTERVAL_MS = 28;
const EMOTE_HOLD_MS = 3000;
const MIN_W = 220;
const MAX_W = 620;
const RICH_W = 420;
const COMPARISON_W = 760;
const MIN_H = 72;
// Match the source prototype's measurement: pill-height (72 = left-pad +
// avatar + right-pad at idle) + body left margin (16) + open right padding
// (26) + a little buffer (8) = 122.
const CHROME = 72 + 16 + 26 + 8;

interface PillPayload {
    kind: FloatingPillPayloadKind;
    text: string;
    ts: number;
    /** Optional override label for the name line; defaults to "Kestrel". */
    name?: string;
    /** Emotion tag emitted by the AI (e.g. "vibing", "sad"). */
    emotion?: string;
    /** Active agent tags to display beside the assistant label. */
    tags?: string[];
    data?: unknown;
}

interface ComparisonPillData {
    title?: string;
    comparison: DriverExpertComparisonData;
}

const parsePayload = (raw: string | null): PillPayload | null => {
    if (!raw) return null;
    try {
        const obj = JSON.parse(raw);
        const text = typeof obj?.text === 'string' ? obj.text.trim() : '';
        const kind = typeof obj?.kind === 'string' ? obj.kind : 'message';
        const tags = Array.isArray(obj.tags)
            ? obj.tags.filter((tag: unknown): tag is string => typeof tag === 'string' && !!tag.trim())
            : typeof obj.tag === 'string' && obj.tag.trim()
                ? [obj.tag]
                : undefined;
        if (text || tags || obj?.data) {
            return {
                kind: [
                    'tool',
                    'baseline',
                    'map',
                    'plan',
                    'live_range_todo_list',
                    'driver_expert_comparison',
                ].includes(kind) ? kind as PillPayload['kind'] : 'message',
                text,
                ts: Number(obj.ts) || Date.now(),
                name: typeof obj.name === 'string' ? obj.name : undefined,
                emotion: typeof obj.emotion === 'string' ? obj.emotion : undefined,
                tags,
                data: obj.data,
            };
        }
    } catch {
        /* ignore — stale or malformed payload */
    }
    return null;
};

const readEmotionGifs = (): Record<string, string> => {
    try { return JSON.parse(localStorage.getItem(EMOTION_GIFS_KEY) || '{}'); }
    catch { return {}; }
};

const getRichPayloadHeight = (payload: PillPayload): number => {
    if (payload.kind === 'driver_expert_comparison') return 500;
    if (payload.kind === 'map') return 260;
    if (payload.kind === 'plan') return 220;
    if (payload.kind === 'live_range_todo_list') return 210;
    if (payload.kind === 'baseline') return 136;
    if (payload.kind === 'tool') return 118;
    return MIN_H;
};

const getRichPayloadWidth = (payload: PillPayload): number => (
    payload.kind === 'driver_expert_comparison' ? COMPARISON_W : RICH_W
);

const normalizeComparisonPayload = (payload: PillPayload): PillPayload | null => {
    if (payload.kind !== 'driver_expert_comparison') return payload;
    const source = payload.data;
    if (!source || typeof source !== 'object' || Array.isArray(source)) return null;
    const comparison = normalizeDriverExpertComparisonData(
        (source as Record<string, unknown>).comparison,
    );
    if (!comparison) return null;
    return {
        ...payload,
        data: {
            ...source,
            comparison,
        },
    };
};

const getRichPayloadHoldDurationMs = (payload: PillPayload): number => {
    if (payload.kind !== 'driver_expert_comparison') {
        return FLOATING_PILL_RICH_CONTENT_HOLD_MS;
    }
    const display = payload.data as ComparisonPillData;
    return Math.max(
        FLOATING_PILL_RICH_CONTENT_HOLD_MS,
        getDriverExpertReplayDurationMs(display.comparison)
            + FLOATING_PILL_COMPARISON_COMPLETION_PAUSE_MS,
    );
};

const FloatingChat: React.FC = () => {
    const [open, setOpen] = useState(false);
    const [displayText, setDisplayText] = useState('');
    const [richPayload, setRichPayload] = useState<PillPayload | null>(null);
    const [showCaret, setShowCaret] = useState(false);
    const [name, setName] = useState('Kestrel');
    const [targetWidth, setTargetWidth] = useState<number>(MIN_W);
    const [targetHeight, setTargetHeight] = useState<number>(MIN_H);
    const [currentEmotion, setCurrentEmotion] = useState<string | null>(null);
    const [agentTags, setAgentTags] = useState<string[]>([]);
    const [emotionGifs, setEmotionGifs] = useState<Record<string, string>>(readEmotionGifs);

    const sizerRef = useRef<HTMLSpanElement>(null);
    const msgRef = useRef<HTMLDivElement>(null);
    const msgInnerRef = useRef<HTMLSpanElement>(null);
    const hideTimerRef = useRef<number | null>(null);
    const typeTimerRef = useRef<number | null>(null);
    const caretTimerRef = useRef<number | null>(null);
    const emoteRevertTimerRef = useRef<number | null>(null);
    const lastTsRef = useRef<number>(0);
    const latestAgentTagsRef = useRef<string[]>([]);

    const clearTimers = useCallback(() => {
        if (hideTimerRef.current !== null) {
            window.clearTimeout(hideTimerRef.current);
            hideTimerRef.current = null;
        }
        if (typeTimerRef.current !== null) {
            window.clearInterval(typeTimerRef.current);
            typeTimerRef.current = null;
        }
        if (caretTimerRef.current !== null) {
            window.clearTimeout(caretTimerRef.current);
            caretTimerRef.current = null;
        }
        if (emoteRevertTimerRef.current !== null) {
            window.clearTimeout(emoteRevertTimerRef.current);
            emoteRevertTimerRef.current = null;
        }
    }, []);

    const measure = useCallback((text: string, tags: string[] = []): number => {
        const sizer = sizerRef.current;
        if (!sizer) return MIN_W;
        sizer.textContent = `${tags.join(' ')} ${text}`.trim();
        const textW = sizer.getBoundingClientRect().width;
        return Math.max(MIN_W, Math.min(MAX_W, Math.ceil(textW + CHROME)));
    }, []);

    const resetScroll = useCallback(() => {
        const inner = msgInnerRef.current;
        if (inner) inner.style.transform = 'translateX(0)';
    }, []);

    /** Scroll the text so the newest character is always visible. Called
     *  after each typed-text update; the CSS transition smooths the shift
     *  so the roll matches the typing cadence. */
    const updateScroll = () => {
        const outer = msgRef.current;
        const inner = msgInnerRef.current;
        if (!outer || !inner) return;
        const overflow = inner.scrollWidth - outer.clientWidth;
        inner.style.transform = `translateX(${overflow > 0 ? -overflow : 0}px)`;
    };

    const shrink = useCallback(() => {
        clearTimers();
        const tags = latestAgentTagsRef.current;
        setTargetWidth(tags.length ? measure('', tags) : MIN_W);
        setTargetHeight(MIN_H);
        setOpen(false);
        setShowCaret(false);
        // Wait for the collapse transition to finish before clearing text so
        // it doesn't peek through the avatar.
        window.setTimeout(() => {
            setDisplayText('');
            setRichPayload(null);
            setCurrentEmotion(null);
            resetScroll();
        }, 400);
    }, [clearTimers, measure, resetScroll]);

    const setPersistentTags = useCallback((tags: string[] = []) => {
        latestAgentTagsRef.current = tags;
        setAgentTags(tags);
        setTargetWidth(tags.length ? measure('', tags) : MIN_W);
    }, [measure]);

    const speak = useCallback((text: string, displayName?: string, emotion?: string, tags: string[] = []) => {
        clearTimers();
        setRichPayload(null);
        setTargetHeight(MIN_H);
        setName(displayName || 'Kestrel');
        setCurrentEmotion(emotion ?? null);
        setPersistentTags(tags);
        const cleanText = text.trim();
        if (!cleanText) {
            setShowCaret(false);
            setDisplayText('');
            setOpen(false);
            return;
        }
        if (emotion && emotion !== 'idle') {
            emoteRevertTimerRef.current = window.setTimeout(() => {
                setCurrentEmotion(null);
                emoteRevertTimerRef.current = null;
            }, EMOTE_HOLD_MS);
        }
        setTargetWidth(measure(text, tags));
        setOpen(true);
        setDisplayText('');
        setShowCaret(true);

        let i = 0;
        typeTimerRef.current = window.setInterval(() => {
            i++;
            setDisplayText(text.slice(0, i));
            if (i >= text.length) {
                if (typeTimerRef.current !== null) {
                    window.clearInterval(typeTimerRef.current);
                    typeTimerRef.current = null;
                }
                // Caret disappears, then the shrink countdown starts from
                // *here* — not from speak() start — so the hold duration is
                // independent of typing length and never gets cut short.
                caretTimerRef.current = window.setTimeout(() => setShowCaret(false), 600);
                hideTimerRef.current = window.setTimeout(shrink, FLOATING_PILL_RICH_CONTENT_HOLD_MS);
            }
        }, TYPE_INTERVAL_MS);
    }, [clearTimers, measure, setPersistentTags, shrink]);

    const showPayload = useCallback((payload: PillPayload) => {
        if (payload.kind === 'message') {
            speak(payload.text, payload.name, payload.emotion, payload.tags);
            return;
        }

        const normalizedPayload = normalizeComparisonPayload(payload);
        if (!normalizedPayload) return;

        clearTimers();
        setName(normalizedPayload.name || 'Kestrel');
        setCurrentEmotion(normalizedPayload.emotion ?? null);
        setPersistentTags(normalizedPayload.tags);
        setDisplayText(normalizedPayload.text.trim());
        setShowCaret(false);
        setRichPayload(normalizedPayload);
        setTargetWidth(getRichPayloadWidth(normalizedPayload));
        setTargetHeight(getRichPayloadHeight(normalizedPayload));
        setOpen(true);
        hideTimerRef.current = window.setTimeout(
            shrink,
            getRichPayloadHoldDurationMs(normalizedPayload),
        );
    }, [clearTimers, setPersistentTags, shrink, speak]);

    // Subscribe to cross-window messages. The 'storage' event only fires in
    // OTHER windows that share the same origin/partition — perfect for the
    // main app → pill broadcast.
    //
    // We deliberately do NOT replay a payload that was already in
    // localStorage at mount time. The pill is for live messages; replaying
    // a stale message on overlay open also conflicts with StrictMode's
    // double-mount (the cleanup would clear the typing/shrink timers from
    // the first run, leaving the pill stuck open with no timer to close it).
    useEffect(() => {
        // Seed only lastTsRef from storage so the FIRST genuine new event is
        // always strictly greater. Do not seed tags here: localStorage is
        // stale across app/overlay restarts, while agent activation is live
        // in-memory state owned by the main chat window.
        const seed = parsePayload(localStorage.getItem(FLOATING_PILL_STORAGE_KEY));
        if (seed) {
            lastTsRef.current = seed.ts;
        }

        const onStorage = (event: StorageEvent) => {
            if (event.key === EMOTION_GIFS_KEY) {
                setEmotionGifs(readEmotionGifs());
                return;
            }
            if (event.key !== FLOATING_PILL_STORAGE_KEY) return;
            const payload = parsePayload(event.newValue);
            if (!payload) return;
            if (payload.ts <= lastTsRef.current) return;
            lastTsRef.current = payload.ts;
            showPayload(payload);
        };
        window.addEventListener('storage', onStorage);
        return () => {
            window.removeEventListener('storage', onStorage);
            clearTimers();
        };
    }, [clearTimers, measure, showPayload]);

    // Roll the typed text after every paint so the caret stays visible.
    // useLayoutEffect runs synchronously post-DOM mutation, so we measure
    // and translate before the browser commits the next frame — no flicker.
    useLayoutEffect(() => {
        updateScroll();
    }, [displayText, open]);

    const renderRichPayload = () => {
        if (!richPayload || richPayload.kind === 'message') return null;
        if (!richPayload.data) return null;

        if (richPayload.kind === 'baseline') {
            return <BaselineProgressDisplay tag={richPayload.data as BaselineCollectionTag} surface="pill" />;
        }
        if (richPayload.kind === 'map') {
            return <AiMapToolDisplay display={richPayload.data as AiMapDisplayPayload} surface="pill" />;
        }
        if (richPayload.kind === 'plan') {
            return <ProcedurePlanDisplay plan={richPayload.data as ProcedurePlan} surface="pill" />;
        }
        if (richPayload.kind === 'live_range_todo_list') {
            return <LiveRangeTodoListDisplay snapshot={richPayload.data as LiveRangeTodoListSnapshot} surface="pill" />;
        }
        if (richPayload.kind === 'driver_expert_comparison') {
            const display = richPayload.data as ComparisonPillData;
            return (
                <DriverExpertComparisonGraph
                    className="floating-pill-comparison"
                    data={display.comparison}
                    title={display.title || 'Driver vs Expert'}
                    layout={{
                        chartHeight: 150,
                        trajectoryHeight: 180,
                        minColumnWidth: 260,
                    }}
                />
            );
        }
        if (richPayload.kind === 'tool') {
            return <ToolMessageDisplay tool={richPayload.data as ToolMessageDisplayData} surface="pill" />;
        }

        return null;
    };

    const tagged = agentTags.length > 0;
    const rich = Boolean(richPayload && richPayload.kind !== 'message');

    // Track the OS window size to the pill so there's no transparent area
    // outside the pill (which would show the title bar of whatever sits
    // underneath as a white frame). When opening, grow immediately so the
    // pill has room to expand into; when closing, wait for the CSS shrink
    // transition (700ms) before snapping the window back, so the pill
    // isn't clipped mid-animation.
    useEffect(() => {
        const api = (window as unknown as { electronAPI?: { resizeFloatingChat?: (w: number, h: number) => void } }).electronAPI;
        const resize = api?.resizeFloatingChat;
        if (!resize) return;
        if (open || tagged) {
            resize(targetWidth, targetHeight);
            return;
        }
        const t = window.setTimeout(() => resize(72, MIN_H), 720);
        return () => window.clearTimeout(t);
    }, [open, tagged, targetHeight, targetWidth]);

    // Click the pill itself to dismiss when it's open.
    const handlePillClick = () => {
        if (open) shrink();
    };

    const pillStyle: React.CSSProperties = {
        ['--target-w' as any]: `${targetWidth}px`,
        ['--target-h' as any]: `${targetHeight}px`,
    };

    return (
        <div className="floating-pill-stage">
            <div
                className={`pill${open ? ' open' : ''}${tagged ? ' tagged' : ''}${rich ? ' rich' : ''}`}
                style={pillStyle}
                onClick={handlePillClick}
                aria-live="polite"
            >
                <div className="avatar" aria-hidden="true">
                    {(() => {
                        const key = currentEmotion ?? 'idle';
                        const gif = emotionGifs[key] ?? emotionGifs['idle'];
                        return gif ? <img src={gif} alt={key} /> : 'AI';
                    })()}
                </div>
                <div className="body">
                    <div className="name-row">
                        <span className="name">{name}</span>
                        {agentTags.map((tag) => (
                            <span key={tag} className="agent-tag">{tag}</span>
                        ))}
                    </div>
                    <div className="msg" ref={msgRef}>
                        <span className="msg-inner" ref={msgInnerRef}>
                            {displayText}
                            {showCaret && <span className="caret" />}
                        </span>
                    </div>
                    {rich && (
                        <div className="rich-body">
                            {renderRichPayload()}
                        </div>
                    )}
                </div>
            </div>
            <span className="sizer" ref={sizerRef} aria-hidden="true" />
        </div>
    );
};

export default FloatingChat;
