import React, { useRef, useState } from 'react';
import type { AiChatScreenPillInfo } from 'contexts/AiChatScreenContext';

interface AiChatScreenInfoProps {
    label: string;
    info: AiChatScreenPillInfo;
}

const AiChatScreenInfo = ({ label, info }: AiChatScreenInfoProps) => {
    const [open, setOpen] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);
    const triggerRef = useRef<HTMLButtonElement>(null);

    const handlePointerLeave = (event: React.PointerEvent<HTMLDivElement>) => {
        if (event.relatedTarget instanceof Node && event.currentTarget.contains(event.relatedTarget)) return;
        if (containerRef.current?.contains(document.activeElement)) return;
        setOpen(false);
    };

    const handleBlur = (event: React.FocusEvent<HTMLDivElement>) => {
        if (event.currentTarget.contains(event.relatedTarget as Node | null)) return;
        setOpen(false);
    };

    const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
        if (event.key !== 'Escape') return;
        event.preventDefault();
        triggerRef.current?.focus();
        setOpen(false);
    };

    return (
        <div
            className="ai-chat__screen-info"
            ref={containerRef}
            onPointerEnter={(event) => {
                if (event.pointerType !== 'touch') setOpen(true);
            }}
            onPointerLeave={handlePointerLeave}
            onBlur={handleBlur}
            onKeyDown={handleKeyDown}
        >
            <button
                type="button"
                ref={triggerRef}
                className="ai-chat__chip ai-chat__chip--blue ai-chat__screen-info-trigger"
                aria-haspopup="dialog"
                aria-expanded={open}
                aria-controls="ai-chat-screen-info-card"
                aria-label={`Screen information: ${label}`}
                onFocus={() => setOpen(true)}
                onClick={() => setOpen((current) => !current)}
            >
                {label}
            </button>
            {open && (
                <div
                    id="ai-chat-screen-info-card"
                    className="ai-chat__screen-info-card"
                    role="dialog"
                    aria-label={`${label} information`}
                >
                    <div className="ai-chat__screen-info-head">
                        <strong>{info.title}</strong>
                        <span className={`ai-chat__screen-info-status ai-chat__screen-info-status--${info.status.tone}`}>
                            {info.status.label}
                        </span>
                    </div>
                    <p>{info.description}</p>
                    {info.facts.length > 0 && (
                        <dl>
                            {info.facts.map((fact) => (
                                <div key={fact.label}>
                                    <dt>{fact.label}</dt>
                                    <dd>{fact.value}</dd>
                                </div>
                            ))}
                        </dl>
                    )}
                </div>
            )}
        </div>
    );
};

export default AiChatScreenInfo;
