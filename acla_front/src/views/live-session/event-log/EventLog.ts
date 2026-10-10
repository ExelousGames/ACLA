import { LiveEventType, LiveSessionEvent } from 'views/session-shared/session-intelligence/types';

const shouldLogEvent = (event: LiveSessionEvent): boolean => event.type === 'STRAIGHT';

export interface EventSearchParams {
    eventType: LiveEventType;
    scope: 'last' | 'last_n' | 'lap_current' | 'lap_last' | 'all';
    n?: number;
    currentLap?: number;
}

export class EventLog {
    private events: LiveSessionEvent[];

    constructor(initialEvents: LiveSessionEvent[] = []) {
        this.events = initialEvents.filter(shouldLogEvent);
    }

    replace(events: LiveSessionEvent[]): void {
        this.events = events.filter(shouldLogEvent);
    }

    find(params: EventSearchParams): LiveSessionEvent[] {
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

    all(): LiveSessionEvent[] {
        return this.events.slice();
    }

    reset(): void {
        this.events = [];
    }
}
