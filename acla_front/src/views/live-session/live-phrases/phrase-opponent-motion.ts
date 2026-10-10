import type { StandardTelemetrySample } from '../live-session-types';
import { lapDistance } from './phrase-corner-geometry';

const normalized = (value: unknown): value is number => typeof value === 'number'
    && Number.isFinite(value) && value >= 0 && value <= 1;

/** Follow the nearest car ahead by ID; repeated position packets do not erase its rate. */
export class PhraseOpponentMotion {
    private previous?: { id: string; position: number; changedAt: number; receivedAt: number; rate?: number };

    reset(): void { this.previous = undefined; }

    update(sample: StandardTelemetrySample, now: number, maxAgeMs: number): void {
        const player = sample.Graphics_normalized_car_position;
        const playerId = sample.Graphics_player_car_id;
        if (!normalized(player) || playerId === undefined || !Number.isSafeInteger(playerId) || playerId < 0) {
            this.reset();
            return;
        }
        const opponent = Object.entries(sample.Graphics_normalized_positions ?? {})
            .filter(([id, position]) => id !== String(playerId) && normalized(position)
                && lapDistance(player, position) > 0 && lapDistance(player, position) < 0.5)
            .sort(([, a], [, b]) => lapDistance(player, a) - lapDistance(player, b))[0];
        if (!opponent) {
            this.reset();
            return;
        }
        const [id, position] = opponent;
        const previous = this.previous;
        if (!previous || previous.id !== id || now <= previous.receivedAt || now - previous.receivedAt > maxAgeMs) {
            this.previous = { id, position, changedAt: now, receivedAt: now };
            return;
        }
        if (position === previous.position) {
            previous.receivedAt = now;
            if (now - previous.changedAt > maxAgeMs) previous.rate = undefined;
            return;
        }
        const delta = lapDistance(previous.position, position);
        const elapsed = now - previous.changedAt;
        this.previous = { id, position, changedAt: now, receivedAt: now,
            // Wrapping forwards is valid; reversing and long gaps cannot estimate arrival.
            rate: delta > 0 && delta < 0.5 && elapsed <= maxAgeMs ? delta * 1000 / elapsed : undefined };
    }

    positionAt(now: number, maxAgeMs: number): number | undefined {
        const previous = this.previous;
        return previous && now >= previous.receivedAt && now - previous.receivedAt <= maxAgeMs
            ? previous.position : undefined;
    }

    at(now: number, maxAgeMs: number): { position: number; rate: number } | undefined {
        const previous = this.previous;
        return previous?.rate !== undefined && now >= previous.receivedAt && now - previous.changedAt <= maxAgeMs
            ? { position: previous.position, rate: previous.rate } : undefined;
    }
}
