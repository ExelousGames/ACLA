import type { StandardTelemetrySample } from '../live-session-types';
import type { LiveTelemetryEvent } from '../live-telemetry-store';
import type { TrackVisionAnalysis, TrackVisionDetection, CornerDirection, CornerPosition } from '../track-vision/track-vision-types';
import { VISION_MAX_AGE_MS } from '../track-vision/track-vision-types';

export const TELEMETRY_MAX_AGE_MS = 1500;
export const REARM_MS = 500;
export const COOLDOWN_MS = 8000;
const HISTORY_LIMIT = 50;

const INPUT_LABELS = {
    speed: 'Speed (km/h)', carAhead: 'Opponent ahead on visible track',
    cornerDirection: 'Visible corner', playerPosition: 'Player position', opponentPosition: 'Opponent position',
} as const;
type Input = keyof typeof INPUT_LABELS;
type Inputs = TrackVisionAnalysis & { speed?: number };
type PhraseVisionInput = Pick<TrackVisionDetection, 'capturedAt' | 'calibration' | 'analysis'>;
type Condition = { input: Input; operator: '>=' | '='; value: number | string };
export interface PhraseRule {
    id: string;
    sentence: string;
    category: 'Corner position';
    holdMs: number;
    conditions: readonly Condition[];
}
const condition = (input: Input, operator: Condition['operator'], value: Condition['value']): Condition => ({ input, operator, value });
const following = [condition('speed', '>=', 30), condition('carAhead', '=', 1)];
const directions: CornerDirection[] = ['left', 'right'];
const positions: CornerPosition[] = ['inside', 'middle', 'outside'];
const positionText = (position: CornerPosition) => position === 'middle' ? 'in the middle of the track' : `on the ${position}`;

// The catalog drives both evaluation and the displayed conditions.
export const PHRASE_RULES: readonly PhraseRule[] = directions.flatMap((direction) => positions.flatMap((player) => positions.map((opponent) => ({
    id: `corner-${direction}-player-${player}-opponent-${opponent}`,
    category: 'Corner position' as const,
    sentence: `${direction === 'left' ? 'Left' : 'Right'}-hand corner: you are ${positionText(player)}; the opponent ahead is ${positionText(opponent)}.`,
    holdMs: 800,
    conditions: [...following, condition('cornerDirection', '=', direction), condition('playerPosition', '=', player), condition('opponentPosition', '=', opponent)],
}))));

export const describeConditions = (rule: PhraseRule): string => rule.conditions.map(({ input, operator, value }) => (
    `${INPUT_LABELS[input]} ${operator} ${value}`
)).join(' AND ');

export type RuleStatus = 'Missing input' | 'Not matched' | 'Confirming' | 'Active' | 'Cooldown';
export interface PhraseEvent { id: number; ruleId: string; sentence: string; timestamp: number }
export interface PhraseSnapshot {
    telemetryReady: boolean;
    visionReady: boolean;
    rules: Array<{ id: string; status: RuleStatus; missing: string[] }>;
    events: PhraseEvent[];
}
interface RuleMemory { since?: number; clearSince?: number; fired: boolean; lastEmitted?: number }
const finite = (value: unknown): number | undefined => typeof value === 'number' && Number.isFinite(value) ? value : undefined;
const nonnegative = (value: unknown): number | undefined => {
    const number = finite(value);
    return number !== undefined && number >= 0 ? number : undefined;
};

function readInputs(sample: StandardTelemetrySample, vision: TrackVisionAnalysis): Inputs {
    const velocity = [sample.Physics_velocity_x, sample.Physics_velocity_y, sample.Physics_velocity_z];
    return {
        speed: nonnegative(sample.Physics_speed_kmh) ?? (velocity.every((v) => finite(v) !== undefined)
            ? Math.hypot(...velocity as number[]) * 3.6 : undefined),
        ...vision,
    };
}

/** Local rules only. World position, lap position, chat and remote services are not inputs. */
export class PhraseEngine {
    private sample: StandardTelemetrySample = {};
    private receivedAt = -Infinity;
    private vision: PhraseVisionInput | null = null;
    private visionAfter = -Infinity;
    private memory = new Map<string, RuleMemory>();
    private events: PhraseEvent[] = [];
    private nextId = 0;

    reset(now: number) {
        this.sample = {};
        this.receivedAt = -Infinity;
        this.vision = null;
        this.visionAfter = now;
        this.memory.clear();
        this.events = [];
    }

    receiveTelemetry(event: LiveTelemetryEvent, now: number): PhraseSnapshot {
        if (event.type !== 'frame') {
            this.reset(now);
        } else {
            // A gap cannot count toward the continuous hold, even without a UI timer.
            if (now - this.receivedAt > TELEMETRY_MAX_AGE_MS) {
                this.memory.forEach((memory) => { memory.since = undefined; });
            }
            // Paused, replay and off frames cannot generate live position phrases.
            this.sample = event.telemetryStatus === 2 ? event.sample : {};
            this.receivedAt = event.telemetryStatus === 2 ? now : -Infinity;
        }
        return this.evaluate(now, true);
    }

    receiveVision(vision: PhraseVisionInput | null, now: number): PhraseSnapshot {
        // A fresh result cannot retroactively fill a capture gap when no timer ran.
        if (!this.vision || now - this.vision.capturedAt > VISION_MAX_AGE_MS
            || JSON.stringify(vision?.calibration) !== JSON.stringify(this.vision.calibration)) {
            this.memory.forEach((memory) => { memory.since = undefined; });
        }
        this.vision = vision;
        return this.evaluate(now);
    }

    evaluate(now: number, emit = false): PhraseSnapshot {
        const telemetryReady = now >= this.receivedAt && now - this.receivedAt <= TELEMETRY_MAX_AGE_MS;
        const visionReady = Boolean(this.vision && this.vision.capturedAt >= this.visionAfter
            && now >= this.vision.capturedAt && now - this.vision.capturedAt <= VISION_MAX_AGE_MS
            && this.vision.analysis);
        const inputs = readInputs(telemetryReady ? this.sample : {}, visionReady ? this.vision!.analysis! : {});
        const rules = PHRASE_RULES.map((rule) => {
            const memory = this.memory.get(rule.id) ?? { fired: false };
            this.memory.set(rule.id, memory);
            const missing = rule.conditions.filter(({ input }) => inputs[input] === undefined).map(({ input }) => INPUT_LABELS[input]);
            const matched = telemetryReady && missing.length === 0 && rule.conditions.every(({ input, operator, value }) => {
                const actual = inputs[input]!;
                return operator === '>=' ? typeof actual === 'number' && typeof value === 'number' && actual >= value : actual === value;
            });
            let status: RuleStatus;
            if (!matched) {
                memory.since = undefined;
                memory.clearSince ??= now;
                if (now - memory.clearSince >= REARM_MS) memory.fired = false;
                status = missing.length || !telemetryReady ? 'Missing input' : 'Not matched';
            } else {
                // Also rearm when the next frame arrives after a sustained false condition.
                if (memory.clearSince !== undefined && now - memory.clearSince >= REARM_MS) memory.fired = false;
                memory.clearSince = undefined;
                memory.since ??= now;
                const cooldown = memory.lastEmitted !== undefined && now - memory.lastEmitted < COOLDOWN_MS;
                status = memory.fired ? 'Active' : now - memory.since < rule.holdMs ? 'Confirming' : cooldown ? 'Cooldown' : 'Confirming';
                if (!memory.fired && !cooldown && now - memory.since >= rule.holdMs && emit) {
                    this.events = [...this.events, { id: ++this.nextId, ruleId: rule.id, sentence: rule.sentence, timestamp: now }].slice(-HISTORY_LIMIT);
                    memory.lastEmitted = now;
                    memory.fired = true;
                    status = 'Active';
                }
            }
            return { id: rule.id, status, missing };
        });
        return { telemetryReady, visionReady, rules, events: this.events };
    }
}
