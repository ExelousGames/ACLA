import React, { useEffect, useLayoutEffect, useRef, useState } from 'react';
import './floating-chat.css';

/**
 * AI Chat Pill — ambient overlay for the always-on-top Electron window.
 *
 * Idle: a small pulsing circle showing the "AI" avatar.
 * Active: expands horizontally and types out ACLA's latest reply, then
 *         auto-collapses back to the circle.
 *
 * Voice is started from the main application — this window is read-only.
 * The pill subscribes to assistant transcripts via a shared localStorage key
 * (`storage` events fire across all same-origin BrowserWindows).
 */

const SHARED_KEY = 'acla-pill-msg';
const EMOTION_GIFS_KEY = 'acla-emotion-gifs';
const TYPE_INTERVAL_MS = 28;
const POST_TYPE_HOLD_MS = 3800;
const MIN_W = 220;
const MAX_W = 620;
// Match the source prototype's measurement: pill-height (72 = left-pad +
// avatar + right-pad at idle) + body left margin (16) + open right padding
// (26) + a little buffer (8) = 122.
const CHROME = 72 + 16 + 26 + 8;

interface PillPayload {
    text: string;
    ts: number;
    /** Optional override label for the name line; defaults to "ACLA". */
    name?: string;
    /** Emotion tag emitted by the AI (e.g. "vibing", "sad"). */
    emotion?: string;
}

const parsePayload = (raw: string | null): PillPayload | null => {
    if (!raw) return null;
    try {
        const obj = JSON.parse(raw);
        if (typeof obj?.text === 'string' && obj.text.trim()) {
            return {
                text: obj.text,
                ts: Number(obj.ts) || Date.now(),
                name: typeof obj.name === 'string' ? obj.name : undefined,
                emotion: typeof obj.emotion === 'string' ? obj.emotion : undefined,
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

const FloatingChat: React.FC = () => {
    const [open, setOpen] = useState(false);
    const [displayText, setDisplayText] = useState('');
    const [showCaret, setShowCaret] = useState(false);
    const [name, setName] = useState('ACLA');
    const [targetWidth, setTargetWidth] = useState<number>(MIN_W);
    const [currentEmotion, setCurrentEmotion] = useState<string | null>(null);
    const [emotionGifs, setEmotionGifs] = useState<Record<string, string>>(readEmotionGifs);

    const sizerRef = useRef<HTMLSpanElement>(null);
    const msgRef = useRef<HTMLDivElement>(null);
    const msgInnerRef = useRef<HTMLSpanElement>(null);
    const hideTimerRef = useRef<number | null>(null);
    const typeTimerRef = useRef<number | null>(null);
    const caretTimerRef = useRef<number | null>(null);
    const lastTsRef = useRef<number>(0);

    const clearTimers = () => {
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
    };

    const measure = (text: string): number => {
        const sizer = sizerRef.current;
        if (!sizer) return MIN_W;
        sizer.textContent = text;
        const textW = sizer.getBoundingClientRect().width;
        return Math.max(MIN_W, Math.min(MAX_W, Math.ceil(textW + CHROME)));
    };

    const resetScroll = () => {
        const inner = msgInnerRef.current;
        if (inner) inner.style.transform = 'translateX(0)';
    };

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

    const shrink = () => {
        clearTimers();
        setOpen(false);
        setShowCaret(false);
        // Wait for the collapse transition to finish before clearing text so
        // it doesn't peek through the avatar.
        window.setTimeout(() => {
            setDisplayText('');
            setCurrentEmotion(null);
            resetScroll();
        }, 400);
    };

    const speak = (text: string, displayName?: string, emotion?: string) => {
        clearTimers();
        setName(displayName || 'ACLA');
        setCurrentEmotion(emotion ?? null);
        setTargetWidth(measure(text));
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
                hideTimerRef.current = window.setTimeout(shrink, POST_TYPE_HOLD_MS);
            }
        }, TYPE_INTERVAL_MS);
    };

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
        // Seed lastTsRef from whatever's in storage so the FIRST genuine new
        // event is always strictly greater. This avoids replaying old state.
        const seed = parsePayload(localStorage.getItem(SHARED_KEY));
        if (seed) lastTsRef.current = seed.ts;

        const onStorage = (event: StorageEvent) => {
            if (event.key === EMOTION_GIFS_KEY) {
                setEmotionGifs(readEmotionGifs());
                return;
            }
            if (event.key !== SHARED_KEY) return;
            const payload = parsePayload(event.newValue);
            if (!payload) return;
            if (payload.ts <= lastTsRef.current) return;
            lastTsRef.current = payload.ts;
            speak(payload.text, payload.name, payload.emotion);
        };
        window.addEventListener('storage', onStorage);
        return () => {
            window.removeEventListener('storage', onStorage);
            clearTimers();
        };
    }, []);

    // Roll the typed text after every paint so the caret stays visible.
    // useLayoutEffect runs synchronously post-DOM mutation, so we measure
    // and translate before the browser commits the next frame — no flicker.
    useLayoutEffect(() => {
        updateScroll();
    }, [displayText, open]);

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
        if (open) {
            resize(targetWidth, 72);
            return;
        }
        const t = window.setTimeout(() => resize(72, 72), 720);
        return () => window.clearTimeout(t);
    }, [open, targetWidth]);

    // Click the pill itself to dismiss when it's open.
    const handlePillClick = () => {
        if (open) shrink();
    };

    const pillStyle: React.CSSProperties = {
        ['--target-w' as any]: `${targetWidth}px`,
    };

    return (
        <div className="floating-pill-stage">
            <div
                className={`pill${open ? ' open' : ''}`}
                style={pillStyle}
                onClick={handlePillClick}
                aria-live="polite"
            >
                <div className="avatar" aria-hidden="true">
                    {currentEmotion && emotionGifs[currentEmotion]
                        ? <img src={emotionGifs[currentEmotion]} alt={currentEmotion} />
                        : 'AI'
                    }
                </div>
                <div className="body">
                    <div className="name">{name}</div>
                    <div className="msg" ref={msgRef}>
                        <span className="msg-inner" ref={msgInnerRef}>
                            {displayText}
                            {showCaret && <span className="caret" />}
                        </span>
                    </div>
                </div>
            </div>
            <span className="sizer" ref={sizerRef} aria-hidden="true" />
        </div>
    );
};

export default FloatingChat;
