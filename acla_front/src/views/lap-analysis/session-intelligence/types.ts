export type EventType = 'CORNER' | 'STRAIGHT' | 'CRASHED' | 'OVERTAKE';

export interface SessionEvent {
    id: string;
    type: EventType;
    startSampleIdx: number;
    endSampleIdx: number;
    lap: number;
    trackPosition: number; // normalized 0.0–1.0 at event start
    timestamp: number;     // ms since session start
    metadata?: Record<string, any>;
}

export interface TelemetrySample {
    [key: string]: any;
}

export type ReduceOp = 'raw' | 'avg' | 'min' | 'max' | 'stats';

export type QueryScope =
    | { type: 'last_seconds'; seconds: number }
    | { type: 'event'; eventType: EventType; which: 'last' | 'current' }
    | { type: 'lap'; lap: 'current' | 'last' | number }
    | { type: 'range'; start: number; end: number };

export interface TelemetryQuery {
    fields: string[];
    scope: QueryScope;
    reduce: ReduceOp;
}

export interface FieldStats {
    avg: number;
    min: number;
    max: number;
    stddev: number;
}

export type QueryResult = Record<string, number | number[] | FieldStats>;

export interface CornerDefinition {
    name: string;
    from: number; // normalized position
    to: number;
}

export interface CornerLookahead {
    name: string;
    trackPosition: number;
    distanceAhead: number; // normalized distance from current position
}
