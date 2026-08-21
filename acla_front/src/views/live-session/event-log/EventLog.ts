import { EventType, SessionEvent } from 'views/lap-analysis/session-intelligence/types';

export interface EventSearchParams {
    eventType: EventType;
    scope: 'last' | 'last_n' | 'lap_current' | 'lap_last' | 'all';
    n?: number;
    currentLap?: number;
}

export class EventLog {
    private events: SessionEvent[];

    constructor(initialEvents: SessionEvent[] = []) {
        this.events = initialEvents.slice();
    }

    push(event: SessionEvent): void {
        this.events.push(event);
    }

    replace(events: SessionEvent[]): void {
        this.events = events.slice();
    }

    find(params: EventSearchParams): SessionEvent[] {
        const matches = this.events.filter((event) => event.type === params.eventType);

        switch (params.scope) {
            case 'last':
                return matches.length > 0 ? [matches[matches.length - 1]] : [];

            case 'last_n':
                return matches.slice(-(params.n ?? 1));

            case 'lap_current':
                return matches.filter((event) => event.lap === (params.currentLap ?? 0));

            case 'lap_last':
                return matches.filter((event) => event.lap === (params.currentLap ?? 1) - 1);

            case 'all':
                return matches;

            default:
                return [];
        }
    }

    all(): SessionEvent[] {
        return this.events.slice();
    }

    get length(): number {
        return this.events.length;
    }

    reset(): void {
        this.events = [];
    }
}
