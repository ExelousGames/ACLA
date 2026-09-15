import React, { forwardRef, useImperativeHandle, useMemo, useRef } from 'react';
import { NamedOperationComponentHandle, useRegisterOperationComponentRef } from 'contexts/OperationComponentRefContext';
import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import suspensionCar from 'assets/suspension-car.png';
import { useCurrentTelemetry, useTelemetryStatus } from './live-telemetry-store';
import type { LiveTelemetry } from './live-session-types';
import './LiveSuspension.css';

const WHEELS = [
    { id: 'fl', label: 'Front left', field: 'Physics_suspension_travel_front_left', x: 217, y: 254, width: 81, height: 179, line: 'M -128 302 H 145 L 200 340' },
    { id: 'fr', label: 'Front right', field: 'Physics_suspension_travel_front_right', x: 725, y: 254, width: 81, height: 179, line: 'M 1152 302 H 879 L 824 340' },
    { id: 'rl', label: 'Rear left', field: 'Physics_suspension_travel_rear_left', x: 193, y: 1021, width: 88, height: 195, line: 'M -128 1195 H 135 L 176 1120' },
    { id: 'rr', label: 'Rear right', field: 'Physics_suspension_travel_rear_right', x: 743, y: 1021, width: 88, height: 195, line: 'M 1152 1195 H 889 L 848 1120' },
] as const;

interface LiveSuspensionProps {
    name: string;
    telemetry?: LiveTelemetry;
}

const LiveSuspension = forwardRef<NamedOperationComponentHandle, LiveSuspensionProps>(({ name, telemetry }, forwardedRef) => {
    const currentTelemetry = useCurrentTelemetry();
    const telemetryStatus = useTelemetryStatus();
    const displayedTelemetry = telemetry ?? currentTelemetry;
    const handle = useMemo(() => ({ getComponentName: () => name }), [name]);
    useImperativeHandle(forwardedRef, () => handle, [handle]);
    const registeredHandleRef = useRef(handle);
    registeredHandleRef.current = handle;
    useRegisterOperationComponentRef(registeredHandleRef);

    const wheels = WHEELS.map((wheel) => {
        // The standard suspension channels retain the SDK's meters; only the display uses mm.
        const meters = displayedTelemetry[wheel.field];
        const mm = typeof meters === 'number' ? meters * 1000 : NaN;
        return { ...wheel, value: Number.isFinite(mm) ? mm : null };
    });
    const values = wheels.flatMap(({ value }) => value === null ? [] : [value]);
    // One shared display scale keeps all corners comparable, including signed deflection.
    const minimum = Math.min(0, Math.floor(Math.min(...values, 0) / 50) * 50);
    const maximum = Math.max(100, Math.ceil(Math.max(...values, 0) / 50) * 50);
    const position = (value: number) => (value - minimum) / (maximum - minimum) * 100;
    const zero = position(0);
    const status = displayedTelemetry.Graphics_status ?? telemetryStatus;
    const stateLabel = values.length === 0 ? 'Waiting for suspension data'
        : status === ACC_STATUS.ACC_PAUSE ? 'Paused'
            : status === ACC_STATUS.ACC_REPLAY ? 'Replay'
                : status === ACC_STATUS.ACC_OFF ? 'Session inactive'
                    : values.length < 4 ? `${values.length}/4 wheels available`
                        : telemetry ? 'Telemetry snapshot' : 'Live telemetry';

    return (
        <section className="live-suspension" aria-label="Suspension travel" data-testid="live-suspension">
            <div className="live-suspension__header">
                <p>Live suspension travel at each wheel.</p>
                <span className="live-suspension__status" data-active={values.length > 0 && status === ACC_STATUS.ACC_LIVE}>
                    <span aria-hidden="true" />{stateLabel}
                </span>
            </div>
            <div className="live-suspension__layout">
                <div className="live-suspension__car" aria-hidden="true">
                    <div className="live-suspension__direction"><span>↑</span>Front</div>
                    <div className="live-suspension__artwork">
                        <img src={suspensionCar} alt="" draggable={false} />
                        <svg viewBox="0 0 1024 1536" className="live-suspension__overlay">
                            {wheels.map((wheel) => (
                                <g key={wheel.id} className="live-suspension__wheel" data-available={wheel.value !== null}>
                                    <path d={wheel.line} fill="none" className="live-suspension__connector" />
                                    <rect x={wheel.x} y={wheel.y} width={wheel.width} height={wheel.height} rx="22"
                                        fill="currentColor" fillOpacity={wheel.value === null ? 0.02 : 0.08 + Math.min(Math.abs(wheel.value) / 100, 1) * 0.22} />
                                </g>
                            ))}
                        </svg>
                    </div>
                </div>
                {wheels.map(({ id, label, value }) => {
                    const marker = value === null ? zero : position(value);
                    return (
                        <div key={id} className={`live-suspension__reading live-suspension__reading--${id}`} role="group" aria-label={`${label} suspension travel`}>
                            <div className="live-suspension__wheel-label"><span>{id.toUpperCase()}</span>{label}</div>
                            <div className="live-suspension__caption">Suspension travel</div>
                            <div className="live-suspension__value">
                                <span>{value === null ? '—' : value.toFixed(1)}</span><span>mm</span>
                            </div>
                            <div className="live-suspension__gauge" aria-hidden="true">
                                <span className="live-suspension__fill" style={{ left: `${Math.min(zero, marker)}%`, width: `${Math.abs(marker - zero)}%` }} />
                                {minimum < 0 && <span className="live-suspension__zero" style={{ left: `${zero}%` }} />}
                                {value !== null && <span className="live-suspension__marker" style={{ left: `${marker}%` }} />}
                            </div>
                            {value === null && <span className="live-suspension__unavailable">No data</span>}
                        </div>
                    );
                })}
            </div>
            <p className="live-suspension__scale">Shared scale: {minimum} to {maximum} mm</p>
        </section>
    );
});

LiveSuspension.displayName = 'LiveSuspension';

export default LiveSuspension;
