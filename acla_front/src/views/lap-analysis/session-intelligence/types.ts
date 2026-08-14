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
    | { type: 'now' }
    | { type: 'last_seconds'; seconds: number }
    | { type: 'event'; eventType: EventType; which: 'last' | 'current' }
    | { type: 'lap'; lap: 'current' | 'last' | number }
    | { type: 'range'; start: number; end: number };

export interface TelemetryQuery<TReduce extends ReduceOp> {
    fields: string[];
    scope: QueryScope;
    reduce: TReduce;
}

export interface FieldStats {
    avg: number;
    min: number;
    max: number;
    stddev: number;
}

export type TelemetryValueByReduce = {
    raw: number[];
    avg: number;
    min: number;
    max: number;
    stats: FieldStats;
};

export type QueryResult<TReduce extends ReduceOp> =
    Record<string, TelemetryValueByReduce[TReduce]>;

export interface CornerDefinition {
    name: string;
    from: number; // normalized position
    to: number;
    guideFrom?: number; // normalized position where coaching should request guidance
}

export interface CornerLookahead {
    name: string;
    trackPosition: number;
    distanceAhead: number; // normalized distance from current position
}
