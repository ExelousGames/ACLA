import { SessionEvent, EventType } from './types';

export interface EventSearchParams {
    eventType: EventType;
    scope: 'last' | 'last_n' | 'lap_current' | 'lap_last' | 'all';
    n?: number;
    currentLap?: number;
}

export class EventLog {
    private events: SessionEvent[] = [];

    push(event: SessionEvent): void {
        this.events.push(event);
    }

    // Find events matching search params.
    find(params: EventSearchParams): SessionEvent[] {
        const matches = this.events.filter(e => e.type === params.eventType);

        switch (params.scope) {
            case 'last':
                return matches.length > 0 ? [matches[matches.length - 1]] : [];

            case 'last_n': {
                const n = params.n ?? 1;
                return matches.slice(-n);
            }

            case 'lap_current': {
                const lap = params.currentLap ?? 0;
                return matches.filter(e => e.lap === lap);
            }

            case 'lap_last': {
                const lap = (params.currentLap ?? 1) - 1;
                return matches.filter(e => e.lap === lap);
            }

            case 'all':
                return matches;

            default:
                return [];
        }
    }

    // Most recent event of any type.
    latest(): SessionEvent | null {
        return this.events.length > 0 ? this.events[this.events.length - 1] : null;
    }

    // All events for a given lap.
    byLap(lap: number): SessionEvent[] {
        return this.events.filter(e => e.lap === lap);
    }

    get length(): number {
        return this.events.length;
    }

    reset(): void {
        this.events = [];
    }
}
