import React, { useContext, useEffect, useMemo, useRef, useState } from 'react';
import { useRegisterAiToolComponentRef } from 'contexts/AiToolComponentRefContext';
import { InvalidLiveRangeTodoListError } from 'contexts/AiToolComponentError';
import { LiveSessionContext } from 'views/live-session/LiveSessionContext';
import { AiToolComponentBase } from './AiToolComponentBase';
import type {
    JsonValue,
    LiveRangeTodoContent,
    LiveRangeTodoEventInput,
    LiveRangeTodoEventUpdate,
    LiveRangeTodoListHandle,
    LiveRangeTodoListSnapshot,
    LiveRangeTodoListToolResult,
    LiveRangeTodoSnapshotEvent,
} from './live-range-todo-list-types';
import type { TaskStartFunction } from './task-start-function';

const DEFAULT_LEAD_TIME_SECONDS = 2;
const SAMPLE_WINDOW_MS = 2000;
const ROLLOVER_HIGH_POSITION = 0.8;
const ROLLOVER_LOW_POSITION = 0.2;

export interface LiveRangeTelemetrySample {
    position: number;
    receivedAt: number;
    lap?: number;
}

interface RuntimeEvent extends LiveRangeTodoSnapshotEvent {
    taskStart: TaskStartFunction;
    due?: boolean;
}

type RuntimeSnapshot = Omit<LiveRangeTodoListSnapshot, 'events'> & { events: RuntimeEvent[] };

interface ActiveRun {
    controller: AbortController;
    token: symbol;
}

const isRecord = (value: unknown): value is Record<string, any> => (
    Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

const hasOwn = (value: Record<string, any>, key: string) => (
    Object.prototype.hasOwnProperty.call(value, key)
);

const isJsonSafe = (value: unknown): value is JsonValue => {
    if (value === null || typeof value === 'string' || typeof value === 'boolean') return true;
    if (typeof value === 'number') return Number.isFinite(value);
    if (Array.isArray(value)) return value.every(isJsonSafe);
    if (!isRecord(value)) return false;
    return Object.values(value).every(isJsonSafe);
};

const cloneJson = <T extends JsonValue>(value: T): T => JSON.parse(JSON.stringify(value));

export const getLiveRangeNormalizedPosition = (
    telemetry: Record<string, any> | null | undefined,
): number | undefined => {
    if (!telemetry) return undefined;
    const keys = [
        'Graphics_normalized_car_position',
        'graphics_normalized_car_position',
        'normalized_car_position',
        'car_position',
    ];
    for (const key of keys) {
        if (key in telemetry) {
            const value = Number(telemetry[key]);
            if (Number.isFinite(value)) return Math.max(0, Math.min(1, value));
        }
    }
    return undefined;
};

export const getLiveRangeTelemetryLap = (
    telemetry: Record<string, any> | null | undefined,
): number | undefined => {
    if (!telemetry) return undefined;
    const raw = telemetry.Graphics_completed_laps
        ?? telemetry.Graphics_completed_lap
        ?? telemetry.Graphics?.completed_laps;
    if (raw === undefined || raw === null || raw === '') return undefined;
    const parsed = Math.floor(Number(raw));
    return Number.isFinite(parsed) && parsed >= 0 ? parsed : undefined;
};

const getForwardDelta = (
    previous: LiveRangeTelemetrySample,
    current: LiveRangeTelemetrySample,
): number => {
    if (previous.lap !== undefined && current.lap !== undefined) {
        const lapDelta = current.lap - previous.lap;
        if (lapDelta < 0) return 0;
        if (lapDelta > 0) {
            return Math.max(0, lapDelta + current.position - previous.position);
        }
        return Math.max(0, current.position - previous.position);
    }

    if (current.position >= previous.position) {
        return current.position - previous.position;
    }
    if (
        previous.position >= ROLLOVER_HIGH_POSITION
        && current.position <= ROLLOVER_LOW_POSITION
    ) {
        return 1 - previous.position + current.position;
    }
    return 0;
};

export const calculateRollingForwardRate = (
    samples: LiveRangeTelemetrySample[],
): number | null => {
    if (samples.length < 2) return null;
    const elapsedSeconds = (
        samples[samples.length - 1].receivedAt - samples[0].receivedAt
    ) / 1000;
    if (elapsedSeconds <= 0) return null;

    let distance = 0;
    for (let index = 1; index < samples.length; index += 1) {
        distance += getForwardDelta(samples[index - 1], samples[index]);
    }
    if (distance <= 0) return null;
    return distance / elapsedSeconds;
};

export const calculateForwardCircularDistance = (
    currentPosition: number,
    targetPosition: number,
): number => {
    const direct = targetPosition - currentPosition;
    return direct >= 0 ? direct : direct + 1;
};

export const calculateLiveRangeEta = (
    currentPosition: number,
    targetPosition: number,
    rollingRate: number | null,
): number | null => {
    if (rollingRate === null || rollingRate <= 0) return null;
    return calculateForwardCircularDistance(currentPosition, targetPosition) / rollingRate;
};

export const crossedLiveRangeTodoPosition = (
    previous: LiveRangeTelemetrySample,
    current: LiveRangeTelemetrySample,
    targetPosition: number,
): boolean => {
    if (previous.lap !== undefined && current.lap !== undefined) {
        const lapDelta = current.lap - previous.lap;
        if (lapDelta < 0) return false;
        if (lapDelta > 1) return true;
        if (lapDelta === 1) {
            return targetPosition > previous.position || targetPosition <= current.position;
        }
        return current.position >= previous.position
            && targetPosition > previous.position
            && targetPosition <= current.position;
    }

    if (current.position >= previous.position) {
        return targetPosition > previous.position && targetPosition <= current.position;
    }
    const inferredRollover = previous.position >= ROLLOVER_HIGH_POSITION
        && current.position <= ROLLOVER_LOW_POSITION;
    return inferredRollover
        && (targetPosition > previous.position || targetPosition <= current.position);
};

const serializeEvent = (event: RuntimeEvent): LiveRangeTodoSnapshotEvent => {
    const { taskStart: _taskStart, due: _due, ...snapshotEvent } = event;
    return {
        ...snapshotEvent,
        content: {
            title: event.content.title,
            ...(event.content.detail !== undefined ? { detail: event.content.detail } : {}),
            ...(event.content.metadata !== undefined ? { metadata: cloneJson(event.content.metadata) } : {}),
        },
        data: cloneJson(event.data),
    };
};

const serializeSnapshot = (snapshot: RuntimeSnapshot): LiveRangeTodoListSnapshot => ({
    ...snapshot,
    events: snapshot.events.map(serializeEvent),
});

const formatPosition = (value: number): string => (
    value.toFixed(3).replace(/0+$/, '').replace(/\.$/, '')
);

const parseContent = (
    value: unknown,
    partial = false,
): { content?: Partial<LiveRangeTodoContent>; error?: string } => {
    if (!isRecord(value)) return { error: 'Each event requires a structured content object.' };
    const title = value.title;
    if (!partial && (typeof title !== 'string' || !title.trim())) {
        return { error: 'Each event content requires a non-empty title.' };
    }
    if (title !== undefined && (typeof title !== 'string' || !title.trim())) {
        return { error: 'Event content title must be a non-empty string.' };
    }
    if (value.detail !== undefined && typeof value.detail !== 'string') {
        return { error: 'Event content detail must be a string.' };
    }
    if (value.metadata !== undefined && !isJsonSafe(value.metadata)) {
        return { error: 'Event content metadata must be JSON-safe.' };
    }

    return {
        content: {
            ...(title !== undefined ? { title: title.trim() } : {}),
            ...(value.detail !== undefined ? { detail: value.detail } : {}),
            ...(value.metadata !== undefined ? { metadata: cloneJson(value.metadata) } : {}),
        },
    };
};

const parsePosition = (value: unknown): number | null => {
    const parsed = Number(value);
    return Number.isFinite(parsed) && parsed >= 0 && parsed <= 1 ? parsed : null;
};

const parseLeadTime = (value: unknown): number | null => {
    const parsed = Number(value);
    return Number.isFinite(parsed) && parsed >= 0 ? parsed : null;
};

const parseIds = (value: unknown): { ids?: string[]; error?: string } => {
    if (!Array.isArray(value)) return { error: 'Provide an array of event ids.' };
    const ids = value.map((id) => typeof id === 'string' ? id.trim() : '').filter(Boolean);
    if (ids.length !== value.length) return { error: 'Every event id must be a non-empty string.' };
    return { ids: Array.from(new Set(ids)) };
};

type LiveRangeTodoListDisplayProps = {
    snapshot: LiveRangeTodoListSnapshot | null;
    surface?: 'panel' | 'chat' | 'pill';
};

export const LiveRangeTodoListDisplay: React.FC<LiveRangeTodoListDisplayProps> = ({
    snapshot,
    surface = 'chat',
}) => {
    if (!snapshot || (snapshot.events.length === 0 && surface !== 'panel')) return null;
    const events = surface === 'pill' ? snapshot.events.slice(0, 3) : snapshot.events;

    return (
        <div className={`ai-chat__range-todo ai-chat__range-todo--${surface}`} aria-label="Live range to-do list">
            <div className="ai-chat__range-todo-head">
                <div>
                    <span className="ai-chat__range-todo-kicker">LIVE RANGE TO-DO</span>
                    <div className="ai-chat__range-todo-title">
                        {snapshot.events.length} planned event{snapshot.events.length === 1 ? '' : 's'}
                    </div>
                </div>
                {snapshot.rolling_rate !== null && (
                    <span className="ai-chat__range-todo-rate">
                        {snapshot.rolling_rate.toFixed(3)}/s
                    </span>
                )}
            </div>
            {events.length === 0 ? (
                <div className="ai-chat__range-todo-empty" data-testid="live-range-todo-list-empty">
                    No planned events.
                </div>
            ) : (
                <ul className="ai-chat__range-todo-list">
                    {events.map((event) => (
                        <li key={event.id} className={`ai-chat__range-todo-item ai-chat__range-todo-item--${event.status}`}>
                            <div className="ai-chat__range-todo-item-main">
                                <span className="ai-chat__range-todo-item-name">{event.content.title}</span>
                                <span className="ai-chat__range-todo-item-status">{event.status}</span>
                            </div>
                            {surface !== 'pill' && event.content.detail && (
                                <div className="ai-chat__range-todo-detail">{event.content.detail}</div>
                            )}
                            <div className="ai-chat__range-todo-metrics">
                                <span>Target {formatPosition(event.normalized_position)}</span>
                                <span>{event.eta_seconds === null ? 'ETA --' : `ETA ${event.eta_seconds.toFixed(1)}s`}</span>
                                <span>Lead {event.lead_time_seconds.toFixed(1)}s</span>
                            </div>
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
};

export interface LiveRangeTodoListProps {
    name: string;
    onSnapshotChange?: (snapshot: LiveRangeTodoListSnapshot | null) => void;
    surface?: 'panel' | 'chat' | 'pill';
}

export class LiveRangeTodoListRunner
extends AiToolComponentBase<LiveRangeTodoListSnapshot | null>
implements LiveRangeTodoListHandle {
    private runtime: RuntimeSnapshot;
    private samples: LiveRangeTelemetrySample[] = [];
    private previousSample: LiveRangeTelemetrySample | null = null;
    private readonly activeRuns = new Map<string, ActiveRun>();
    private readonly onChange?: (snapshot: LiveRangeTodoListSnapshot | null) => void;

    constructor(
        componentName: string,
        onChange?: (snapshot: LiveRangeTodoListSnapshot | null) => void,
    ) {
        super(componentName, null);
        const now = Date.now();
        this.runtime = {
            events: [],
            current_position: null,
            rolling_rate: null,
            created_at: now,
            updated_at: now,
        };
        this.onChange = onChange;
    }

    addEvent(eventInput: LiveRangeTodoEventInput): LiveRangeTodoListToolResult {
        const now = Date.now();
        const parsed = this.parseNewEvent(eventInput, now);
        if (!parsed.event) return this.invalidList(parsed.error || 'Invalid live range to-do event.');
        if (this.runtime.events.some((event) => event.id === parsed.event!.id)) {
            return this.invalidList(`Duplicate live range to-do event id: ${parsed.event.id}.`);
        }
        const event = {
            ...parsed.event,
            eta_seconds: this.runtime.current_position === null
                ? null
                : calculateLiveRangeEta(
                    this.runtime.current_position,
                    parsed.event.normalized_position,
                    this.runtime.rolling_rate,
                ),
        };
        const next = this.commit({
            ...this.runtime,
            events: [...this.runtime.events, event],
            updated_at: now,
        });
        return { status: 'ready', todo_list: next, message: `Added event '${event.id}'.` };
    }

    replaceEvents(eventInputs: readonly LiveRangeTodoEventInput[]): LiveRangeTodoListToolResult {
        if (!Array.isArray(eventInputs)) return this.invalidList('Provide an events array.');
        const now = Date.now();
        const parsed = eventInputs.map((event) => this.parseNewEvent(event, now));
        const invalid = parsed.find((entry) => !entry.event);
        if (invalid) return this.invalidList(invalid.error || 'Invalid live range to-do event.');
        const events = parsed.map((entry) => entry.event!);
        const ids = new Set<string>();
        for (const event of events) {
            if (ids.has(event.id)) return this.invalidList(`Duplicate live range to-do event id: ${event.id}.`);
            ids.add(event.id);
        }

        this.abortEvents();
        this.previousSample = null;
        const eventsWithEta = events.map((event) => ({
            ...event,
            eta_seconds: this.runtime.current_position === null
                ? null
                : calculateLiveRangeEta(
                    this.runtime.current_position,
                    event.normalized_position,
                    this.runtime.rolling_rate,
                ),
        }));
        const next = this.commit({
            events: eventsWithEta,
            current_position: this.runtime.current_position,
            rolling_rate: this.runtime.rolling_rate,
            lap: this.runtime.lap,
            created_at: now,
            updated_at: now,
        });
        return {
            status: eventsWithEta.length > 0 ? 'ready' : 'empty',
            todo_list: next,
            message: eventsWithEta.length > 0
                ? `Replaced the queue with ${eventsWithEta.length} event${eventsWithEta.length === 1 ? '' : 's'}.`
                : 'The live range to-do list is empty.',
        };
    }

    updateEvents(eventUpdates: readonly LiveRangeTodoEventUpdate[]): LiveRangeTodoListToolResult {
        if (!Array.isArray(eventUpdates) || eventUpdates.length === 0) {
            return this.invalidList('Provide at least one event update.');
        }
        const now = Date.now();
        const updates = new Map<string, RuntimeEvent>();

        for (const raw of eventUpdates) {
            if (!isRecord(raw) || typeof raw.id !== 'string' || !raw.id.trim()) {
                return this.invalidList('Each event update requires a non-empty id.');
            }
            const id = raw.id.trim();
            if (updates.has(id)) return this.invalidList(`Duplicate live range to-do event id: ${id}.`);
            const existing = this.runtime.events.find((event) => event.id === id);
            if (!existing) return this.invalidList(`Live range to-do event '${id}' was not found.`);

            let next: RuntimeEvent = {
                ...existing,
                status: 'pending',
                due: undefined,
                updated_at: now,
            };
            if (hasOwn(raw, 'content')) {
                const parsedContent = parseContent(raw.content, true);
                if (!parsedContent.content) return this.invalidList(`Event '${id}': ${parsedContent.error}`);
                next.content = { ...existing.content, ...parsedContent.content };
            }
            if (hasOwn(raw, 'normalized_position')) {
                const position = parsePosition(raw.normalized_position);
                if (position === null) return this.invalidList(`Event '${id}' normalized_position must be between 0 and 1.`);
                next.normalized_position = position;
            }
            if (hasOwn(raw, 'lead_time_seconds')) {
                const leadTime = parseLeadTime(raw.lead_time_seconds);
                if (leadTime === null) return this.invalidList(`Event '${id}' lead_time_seconds must be zero or greater.`);
                next.lead_time_seconds = leadTime;
            }
            if (hasOwn(raw, 'data')) {
                if (!isJsonSafe(raw.data)) return this.invalidList(`Event '${id}' data must be JSON-safe.`);
                next.data = cloneJson(raw.data);
            }
            if (hasOwn(raw, 'taskStart')) {
                if (typeof raw.taskStart !== 'function') return this.invalidList(`Event '${id}' taskStart must be a function.`);
                next.taskStart = raw.taskStart as TaskStartFunction;
            }
            next = {
                ...next,
                eta_seconds: this.runtime.current_position === null
                    ? null
                    : calculateLiveRangeEta(
                        this.runtime.current_position,
                        next.normalized_position,
                        this.runtime.rolling_rate,
                    ),
                started_at: undefined,
                lap: undefined,
            };
            updates.set(id, next);
        }

        const affectedIds = new Set(updates.keys());
        this.abortEvents(affectedIds);
        const next = this.commit({
            ...this.runtime,
            events: this.runtime.events.map((event) => updates.get(event.id) ?? event),
            updated_at: now,
        });
        this.runNextDueEvent();
        return { status: 'ready', todo_list: next, message: `Updated ${updates.size} event${updates.size === 1 ? '' : 's'}.` };
    }

    removeEvents(idsInput: readonly string[]): LiveRangeTodoListToolResult {
        const parsed = parseIds(idsInput);
        if (!parsed.ids || parsed.ids.length === 0) return this.invalidList(parsed.error || 'Provide event ids to remove.');
        const ids = new Set(parsed.ids);
        this.abortEvents(ids);
        const events = this.runtime.events.filter((event) => !ids.has(event.id));
        const removedCount = this.runtime.events.length - events.length;
        const next = this.commit({ ...this.runtime, events, updated_at: Date.now() });
        this.runNextDueEvent();
        return {
            status: events.length > 0 ? 'ready' : 'empty',
            todo_list: next,
            message: `Removed ${removedCount} event${removedCount === 1 ? '' : 's'}.`,
        };
    }

    resetEvents(idsInput?: readonly string[]): LiveRangeTodoListToolResult {
        const parsed = idsInput === undefined
            ? { ids: this.runtime.events.map((event) => event.id) }
            : parseIds(idsInput);
        if (!parsed.ids) return this.invalidList(parsed.error || 'Provide valid event ids to reset.');
        const now = Date.now();
        const ids = new Set(parsed.ids);
        this.abortEvents(ids);
        const events = this.runtime.events.map((event): RuntimeEvent => ids.has(event.id) ? {
            ...event,
            status: 'pending',
            due: undefined,
            eta_seconds: this.runtime.current_position === null
                ? null
                : calculateLiveRangeEta(
                    this.runtime.current_position,
                    event.normalized_position,
                    this.runtime.rolling_rate,
                ),
            updated_at: now,
            started_at: undefined,
            lap: undefined,
        } : event);
        const next = this.commit({ ...this.runtime, events, updated_at: now });
        this.runNextDueEvent();
        return {
            status: events.length > 0 ? 'ready' : 'empty',
            todo_list: next,
            message: `Reset ${ids.size} event${ids.size === 1 ? '' : 's'}.`,
        };
    }

    clear(): LiveRangeTodoListToolResult {
        this.abortEvents();
        const next = this.commit({ ...this.runtime, events: [], updated_at: Date.now() });
        return { status: 'empty', todo_list: next, message: 'Cleared the live range to-do list.' };
    }

    get(): LiveRangeTodoListToolResult {
        const current = serializeSnapshot(this.runtime);
        return {
            status: current.events.length > 0 ? 'ready' : 'empty',
            todo_list: current,
            ...(current.events.length === 0 ? { message: 'The live range to-do list is empty.' } : {}),
        };
    }

    acceptTelemetry(telemetry: Record<string, any> | null | undefined): void {
        if (this.isDisposed()) return;
        const position = getLiveRangeNormalizedPosition(telemetry);
        if (position === undefined) return;
        const now = Date.now();
        const currentSample: LiveRangeTelemetrySample = {
            position,
            receivedAt: now,
            lap: getLiveRangeTelemetryLap(telemetry),
        };
        this.samples = [...this.samples, currentSample]
            .filter((sample) => sample.receivedAt >= now - SAMPLE_WINDOW_MS);
        const rate = calculateRollingForwardRate(this.samples);
        const previousSample = this.previousSample;
        this.previousSample = currentSample;
        const events = this.runtime.events.map((event): RuntimeEvent => {
            if (event.status !== 'pending') return event;
            const eta = calculateLiveRangeEta(position, event.normalized_position, rate);
            const crossed = previousSample
                ? crossedLiveRangeTodoPosition(previousSample, currentSample, event.normalized_position)
                : false;
            return {
                ...event,
                eta_seconds: eta,
                due: event.due || crossed || (eta !== null && eta <= event.lead_time_seconds),
                updated_at: now,
            };
        });
        this.commit({
            ...this.runtime,
            events,
            current_position: position,
            rolling_rate: rate,
            lap: currentSample.lap,
            updated_at: now,
        });
        this.runNextDueEvent();
    }

    reset(): void {
        this.abortEvents();
        this.samples = [];
        this.previousSample = null;
        const now = Date.now();
        this.commit({
            events: [],
            current_position: null,
            rolling_rate: null,
            created_at: now,
            updated_at: now,
        });
    }

    protected onDispose(): void {
        this.abortEvents();
        this.samples = [];
        this.previousSample = null;
    }

    private commit(next: RuntimeSnapshot): LiveRangeTodoListSnapshot {
        this.runtime = next;
        const snapshot = serializeSnapshot(next);
        const visibleSnapshot = snapshot.events.length > 0 ? snapshot : null;
        this.publishSnapshot(visibleSnapshot);
        this.onChange?.(visibleSnapshot);
        if (snapshot.events.length === 0) this.deleteComponentRef();
        return snapshot;
    }

    private invalidList(message: string): never {
        throw new InvalidLiveRangeTodoListError(
            this.getComponentName(),
            message,
        );
    }

    private parseNewEvent(
        value: unknown,
        now: number,
    ): { event?: RuntimeEvent; error?: string } {
        if (!isRecord(value)) return { error: 'Each event must be an object.' };
        const id = typeof value.id === 'string' ? value.id.trim() : '';
        if (!id) return { error: 'Each event requires a non-empty id.' };
        const position = parsePosition(value.normalized_position);
        if (position === null) return { error: `Event '${id}' normalized_position must be between 0 and 1.` };
        const leadTime = value.lead_time_seconds === undefined
            ? DEFAULT_LEAD_TIME_SECONDS
            : parseLeadTime(value.lead_time_seconds);
        if (leadTime === null) return { error: `Event '${id}' lead_time_seconds must be zero or greater.` };
        const parsedContent = parseContent(value.content);
        if (!parsedContent.content) return { error: `Event '${id}': ${parsedContent.error}` };
        if (!isJsonSafe(value.data)) return { error: `Event '${id}' requires JSON-safe data.` };
        if (typeof value.taskStart !== 'function') return { error: `Event '${id}' requires a task-start function.` };

        return {
            event: {
                id,
                normalized_position: position,
                lead_time_seconds: leadTime,
                content: parsedContent.content as LiveRangeTodoContent,
                data: cloneJson(value.data),
                taskStart: value.taskStart as TaskStartFunction,
                status: 'pending',
                eta_seconds: null,
                created_at: now,
                updated_at: now,
            },
        };
    }

    private abortEvents(ids?: Set<string>): void {
        Array.from(this.activeRuns.entries()).forEach(([id, run]) => {
            if (!ids || ids.has(id)) {
                this.activeRuns.delete(id);
                run.controller.abort();
            }
        });
    }

    private runNextDueEvent(): void {
        if (this.isDisposed() || this.activeRuns.size > 0) return;
        const event = this.runtime.events.find((candidate) => (
            candidate.status === 'pending' && candidate.due
        ));
        if (!event) return;

        const controller = new AbortController();
        const token = Symbol(event.id);
        const now = Date.now();
        const runningEvent: RuntimeEvent = {
            ...event,
            status: 'running',
            due: undefined,
            lap: this.runtime.lap,
            started_at: now,
            updated_at: now,
        };
        this.activeRuns.set(event.id, { controller, token });
        this.commit({
            ...this.runtime,
            events: this.runtime.events.map((candidate) => (
                candidate.id === event.id ? runningEvent : candidate
            )),
            updated_at: now,
        });

        try {
            Promise.resolve(runningEvent.taskStart(controller.signal)).then(
                () => this.finishEvent(event.id, token),
                (error) => this.finishEvent(event.id, token, error),
            );
        } catch (error) {
            this.finishEvent(event.id, token, error);
        }
    }

    private finishEvent(id: string, token: symbol, error?: unknown): void {
        if (this.isDisposed()) return;
        const activeRun = this.activeRuns.get(id);
        if (!activeRun || activeRun.token !== token || activeRun.controller.signal.aborted) return;
        this.activeRuns.delete(id);
        if (error !== undefined) console.error(`Live range to-do event '${id}' task failed.`, error);
        if (!this.runtime.events.some((event) => event.id === id)) return;
        this.commit({
            ...this.runtime,
            events: this.runtime.events.filter((event) => event.id !== id),
            updated_at: Date.now(),
        });
        this.runNextDueEvent();
    }
}

const LiveRangeTodoList: React.FC<LiveRangeTodoListProps> = ({
    name,
    onSnapshotChange,
    surface = 'chat',
}) => {
    const { currentTelemetry, sessionGame } = useContext(LiveSessionContext);
    const [snapshot, setSnapshot] = useState<LiveRangeTodoListSnapshot | null>(null);
    const snapshotChangeRef = useRef(onSnapshotChange);
    snapshotChangeRef.current = onSnapshotChange;
    const previousSessionGameRef = useRef(sessionGame);
    const runnerRef = useRef<LiveRangeTodoListRunner | null>(null);
    if (!runnerRef.current) {
        runnerRef.current = new LiveRangeTodoListRunner(name, (next) => {
            setSnapshot(next);
            snapshotChangeRef.current?.(next);
        });
    }

    const handle = useMemo<LiveRangeTodoListHandle>(() => ({
        getComponentName: () => name,
        addEvent: (event) => runnerRef.current!.addEvent(event),
        replaceEvents: (events) => runnerRef.current!.replaceEvents(events),
        updateEvents: (updates) => runnerRef.current!.updateEvents(updates),
        removeEvents: (ids) => runnerRef.current!.removeEvents(ids),
        resetEvents: (ids) => runnerRef.current!.resetEvents(ids),
        clear: () => runnerRef.current!.clear(),
        get: () => runnerRef.current!.get(),
    }), [name]);
    useRegisterAiToolComponentRef(name, handle);

    useEffect(() => () => {
        runnerRef.current?.dispose();
        snapshotChangeRef.current?.(null);
    }, []);

    useEffect(() => {
        const previousSessionGame = previousSessionGameRef.current;
        previousSessionGameRef.current = sessionGame;
        if (previousSessionGame === sessionGame) return;
        if (sessionGame === null) return;
        runnerRef.current?.reset();
    }, [sessionGame]);

    useEffect(() => {
        runnerRef.current?.acceptTelemetry(currentTelemetry);
    }, [currentTelemetry]);

    return <LiveRangeTodoListDisplay snapshot={snapshot} surface={surface} />;
};

export default LiveRangeTodoList;
