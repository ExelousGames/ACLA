import type { StandardTelemetrySample } from '../live-session-types';
import type { LiveTelemetryEvent } from '../live-telemetry-store';
import type { TrackVisionDetection } from '../track-vision/track-vision-types';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { getAccTelemetryTrackKey } from 'views/session-shared/visualization/charts/circuitTrackLayout';
import { createPhraseMapContext, PhraseMapContext } from './phrase-map-context';
import { VISION_MAX_AGE_MS } from '../track-vision/track-vision-types';
import { getPhrasePositions, CornerPosition } from './phrase-positions';

export const TELEMETRY_MAX_AGE_MS = 1500;
export const REARM_MS = 500;
export const COOLDOWN_MS = 8000;
const HISTORY_LIMIT = 50;

const INPUT_LABELS = {
    speed: 'Speed (km/h)', carAhead: 'Opponent ahead on visible track',
    playerPosition: 'Player position', opponentPosition: 'Opponent position',
    phase: 'Live Map section', cornerSpeed: 'Mapped corner speed', cornerShape: 'Mapped corner shape',
    linkedOpposite: 'Next linked corner turns the opposite way', linkedSameDirection: 'Next linked corner turns the same way',
    sequenceRemaining: 'Corners remaining in the mapped sequence',
    insideLine: 'Player inside, opponent off the inside', outsideLine: 'Player outside, opponent inside',
} as const;
type Input = keyof typeof INPUT_LABELS;
type Inputs = PhraseMapContext & { carAhead?: 0 | 1; playerPosition?: CornerPosition; opponentPosition?: CornerPosition;
    speed?: number; insideLine?: 0 | 1; outsideLine?: 0 | 1 };
type PhraseVisionInput = Pick<TrackVisionDetection, 'capturedAt' | 'calibration' | 'birdsEyeScene'>;
export class PhraseCondition {
    readonly conditionFit: boolean;
    readonly inputMissing: boolean;

    constructor(
        readonly input: Input,
        readonly operator: '>=' | '=' | 'in',
        readonly value: number | string | readonly string[],
        actual?: Inputs[Input],
    ) {
        this.inputMissing = actual === undefined;
        this.conditionFit = actual !== undefined && (operator === '>='
            ? typeof actual === 'number' && typeof value === 'number' && actual >= value
            : operator === 'in' ? Array.isArray(value) && value.includes(actual) : actual === value);
    }

    get description(): string {
        return `${INPUT_LABELS[this.input]} ${this.operator} ${Array.isArray(this.value) ? this.value.join(' / ') : this.value}`;
    }

    evaluate(inputs: Inputs): PhraseCondition {
        // Keep the catalog and previously published snapshots independent of live updates.
        return new PhraseCondition(this.input, this.operator, this.value, inputs[this.input]);
    }
}

export interface PhraseRule {
    id: string;
    sentence: string;
    category: string;
    holdMs: number;
    conditions: readonly PhraseCondition[];
}
const condition = (input: Input, operator: PhraseCondition['operator'], value: PhraseCondition['value']): PhraseCondition => new PhraseCondition(input, operator, value);
const following = [condition('speed', '>=', 30), condition('carAhead', '=', 1)];
const positioned = [condition('playerPosition', 'in', ['inside', 'middle', 'outside']), condition('opponentPosition', 'in', ['inside', 'middle', 'outside'])];

// The catalog drives both evaluation and the displayed conditions.
// More specific tactics take priority when conditions overlap.
export const PHRASE_RULES: readonly PhraseRule[] = [
    {
        id: 'next-corner', category: 'Setting up the next corner', holdMs: 800,
        sentence: 'Set up the next corner: if you can establish overlap on the outside, stay alongside through this turn and leave room; your side becomes the inside at the following corner.',
        conditions: [...following, ...positioned, condition('phase', 'in', ['entry', 'middle']), condition('outsideLine', '=', 1), condition('linkedOpposite', '=', 1)],
    },
    {
        id: 'same-direction', category: 'Linked corners in the same direction', holdMs: 800,
        sentence: 'These corners turn the same way: keep the car balanced and leave room for the next apex. Build your passing run from the final exit once a clear lane opens.',
        conditions: [...following, ...positioned, condition('phase', 'in', ['entry', 'middle']), condition('linkedSameDirection', '=', 1)],
    },
    {
        id: 'sequence-exit', category: 'Exit into another corner', holdMs: 800,
        sentence: 'More corners follow in this sequence: keep this exit controlled and position for the next turn. Save the full exit-speed attack for the final corner and a clear passing lane.',
        conditions: [...following, ...positioned, condition('phase', '=', 'exit'), condition('sequenceRemaining', '>=', 1)],
    },
    {
        id: 's-bend', category: 'Direction change within a corner', holdMs: 800,
        sentence: 'The road changes direction through this section: make a smooth steering transition, leave room alongside, and prioritize the final exit before attempting a pass.',
        conditions: [...following, ...positioned, condition('phase', 'in', ['entry', 'middle']), condition('cornerShape', '=', 's-bend')],
    },
    {
        id: 'tightening-corner', category: 'Tightening corner', holdMs: 800,
        sentence: 'This corner tightens: keep some grip in reserve and delay full throttle until the steering can unwind. Avoid committing to a passing line that closes at the exit.',
        conditions: [...following, ...positioned, condition('phase', 'in', ['entry', 'middle']), condition('cornerShape', '=', 'tightening')],
    },
    {
        id: 'opening-corner', category: 'Opening corner', holdMs: 800,
        sentence: 'This corner opens out: progressively unwind the steering and build traction. Use the exit to prepare a pass when a clear lane appears.',
        conditions: [...following, ...positioned, condition('phase', 'in', ['middle', 'exit']), condition('cornerShape', '=', 'opening')],
    },
    {
        id: 'hairpin-exit', category: 'Hairpin exit', holdMs: 800,
        sentence: 'Hairpin: finish rotating the car and feed in throttle as the steering opens. Prioritize traction, then use any exit-speed advantage when a passing lane is clear.',
        conditions: [...following, ...positioned, condition('phase', 'in', ['middle', 'exit']), condition('cornerShape', '=', 'hairpin')],
    },
    {
        id: 'inside-outbraking', category: 'Outbraking on the inside', holdMs: 800,
        sentence: 'Slow corner ahead: prepare an inside outbraking move. Establish overlap before turn-in, brake later only if you can still make the apex, and leave exit room.',
        conditions: [...following, ...positioned, condition('phase', '=', 'entry'), condition('cornerSpeed', '=', 'slow'), condition('insideLine', '=', 1)],
    },
    {
        id: 'around-outside', category: 'Around the outside', holdMs: 800,
        sentence: 'Faster corner: carry momentum around the outside only if you have overlap and room. Hold your line and allow space for the car inside.',
        conditions: [...following, ...positioned, condition('phase', '=', 'entry'), condition('cornerSpeed', '=', 'fast'), condition('outsideLine', '=', 1)],
    },
    {
        id: 'switchback', category: 'Switchback', holdMs: 800,
        sentence: 'Prepare a switchback: let the inside car commit, delay your turn-in, then cut back underneath if they run wide and the gap opens.',
        conditions: [...following, ...positioned, condition('phase', 'in', ['entry', 'middle']), condition('cornerSpeed', '=', 'slow'), condition('outsideLine', '=', 1)],
    },
    {
        id: 'better-exit', category: 'Better exit pass', holdMs: 800,
        sentence: 'Build a better exit: open the steering and feed in throttle smoothly. Use any speed advantage to draw alongside once there is a clear passing lane.',
        conditions: [...following, ...positioned, condition('phase', '=', 'exit'), condition('cornerSpeed', 'in', ['slow', 'fast'])],
    },
    {
        id: 'slipstream', category: 'Slipstream', holdMs: 800,
        sentence: 'Use the slipstream on this straight: tuck in behind to build a run, then pull out when you are closing and there is clear space before braking.',
        conditions: [condition('speed', '>=', 80), condition('carAhead', '=', 1), condition('phase', '=', 'straight')],
    },
    {
        id: 'pressure-feint', category: 'Pressure and a feint', holdMs: 800,
        sentence: 'Apply pressure: show a possible attack before the braking zone to invite a defensive line, then exploit an opening, mistake or compromised exit if they react. Keep your move controlled.',
        conditions: [...following, ...positioned, condition('phase', '=', 'entry'), condition('cornerSpeed', 'in', ['slow', 'fast']), condition('insideLine', '=', 0), condition('outsideLine', '=', 0)],
    },
];

export type RuleStatus = 'Missing input' | 'Not matched' | 'Confirming' | 'Active' | 'Cooldown';
export interface PhraseEvent { id: number; ruleId: string; sentence: string; timestamp: number }
export interface PhraseSnapshot {
    telemetryReady: boolean;
    visionReady: boolean;
    mapReady: boolean;
    mapContext: PhraseMapContext;
    rules: Array<{ id: string; status: RuleStatus; missing: string[]; conditions: readonly PhraseCondition[] }>;
    events: PhraseEvent[];
}
interface RuleMemory { since?: number; clearSince?: number; fired: boolean; lastEmitted?: number }
const finite = (value: unknown): number | undefined => typeof value === 'number' && Number.isFinite(value) ? value : undefined;
const nonnegative = (value: unknown): number | undefined => {
    const number = finite(value);
    return number !== undefined && number >= 0 ? number : undefined;
};

function readInputs(sample: StandardTelemetrySample, scene: TrackVisionDetection['birdsEyeScene'], map: PhraseMapContext): Inputs {
    const velocity = [sample.Physics_velocity_x, sample.Physics_velocity_y, sample.Physics_velocity_z];
    const { carAhead, playerPosition, opponentPosition } = getPhrasePositions(scene);
    return {
        speed: nonnegative(sample.Physics_speed_kmh) ?? (velocity.every((v) => finite(v) !== undefined)
            ? Math.hypot(...velocity as number[]) * 3.6 : undefined),
        carAhead, playerPosition, opponentPosition,
        ...map,
        insideLine: playerPosition && opponentPosition
            ? Number(playerPosition === 'inside' && opponentPosition !== 'inside') as 0 | 1 : undefined,
        outsideLine: playerPosition && opponentPosition
            ? Number(playerPosition === 'outside' && opponentPosition === 'inside') as 0 | 1 : undefined,
    };
}

/** Local rules combining published vision with the tagged Live Map and live lap position. */
export class PhraseEngine {
    private sample: StandardTelemetrySample = {};
    private receivedAt = -Infinity;
    private vision: PhraseVisionInput | null = null;
    private visionAfter = -Infinity;
    private memory = new Map<string, RuleMemory>();
    private events: PhraseEvent[] = [];
    private nextId = 0;
    private map: CircuitMapDto | null = null;
    private mapContext = createPhraseMapContext(null);
    private sectionId?: string;
    private game?: string;

    receiveMap(map: CircuitMapDto | null, now: number): PhraseSnapshot {
        if (map !== this.map) {
            this.map = map;
            this.mapContext = createPhraseMapContext(map);
            this.memory.forEach((memory) => { memory.since = undefined; memory.fired = false; });
            this.sectionId = undefined;
        }
        return this.evaluate(now);
    }

    reset(now: number) {
        this.sample = {};
        this.receivedAt = -Infinity;
        this.vision = null;
        this.visionAfter = now;
        this.memory.clear();
        this.events = [];
        this.sectionId = undefined;
        this.game = undefined;
    }

    receiveTelemetry(event: LiveTelemetryEvent, now: number): PhraseSnapshot {
        if (event.type !== 'frame') {
            this.reset(now);
        } else {
            this.game = event.update.game;
            // A gap cannot count toward the continuous hold, even without a UI timer.
            if (now - this.receivedAt > TELEMETRY_MAX_AGE_MS) {
                this.memory.forEach((memory) => { memory.since = undefined; });
            }
            // Paused, replay and off frames cannot generate live guidance.
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
            && this.vision.birdsEyeScene);
        const trackKey = (track: string) => this.game === 'acc' ? getAccTelemetryTrackKey(track) ?? track.trim() : track.trim();
        const sameMap = Boolean(this.map && (!this.game || this.map.game === this.game)
            && (!this.sample.Static_track || trackKey(this.sample.Static_track) === trackKey(this.map.source_track_key || this.map.circuit_name)));
        const mapContext = telemetryReady && sameMap ? this.mapContext.at(this.sample.Graphics_normalized_car_position) : {};
        if (mapContext.sectionId !== this.sectionId) {
            this.memory.forEach((memory) => { memory.since = undefined; });
            this.sectionId = mapContext.sectionId;
        }
        const inputs = readInputs(telemetryReady ? this.sample : {}, visionReady ? this.vision!.birdsEyeScene : null, mapContext);
        let selected = false;
        const rules = PHRASE_RULES.map((rule) => {
            const memory = this.memory.get(rule.id) ?? { fired: false };
            this.memory.set(rule.id, memory);
            const conditions = rule.conditions.map((condition) => condition.evaluate(inputs));
            const missing = conditions.filter((condition) => condition.inputMissing).map(({ input }) => INPUT_LABELS[input]);
            const matched = !selected && telemetryReady && conditions.every((condition) => condition.conditionFit);
            if (matched) selected = true;
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
            return { id: rule.id, status, missing, conditions };
        });
        return { telemetryReady, visionReady, mapReady: sameMap && this.mapContext.ready, mapContext, rules, events: this.events };
    }
}
