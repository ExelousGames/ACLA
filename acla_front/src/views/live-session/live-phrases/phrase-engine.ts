import type { StandardTelemetrySample } from '../live-session-types';
import type { LiveTelemetryEvent } from '../live-telemetry-store';
import type { TrackVisionDetection } from '../track-vision/track-vision-types';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { getAccTelemetryTrackKey } from 'views/session-shared/visualization/charts/circuitTrackLayout';
import { createPhraseMapContext, PhraseMapContext } from './phrase-map-context';
import { VISION_MAX_AGE_MS } from '../track-vision/track-vision-types';
import { getPhrasePositions } from './phrase-positions';
import { Action, Closure, State, describeCondition, type ConditionSnapshot, type NodeInspection, type NodeSnapshot } from './closure';
import { PhraseOpponentMotion } from './phrase-opponent-motion';
import { lapDistance } from './phrase-corner-geometry';

export const TELEMETRY_MAX_AGE_MS = 1500;
export const REARM_MS = 500;
export const COOLDOWN_MS = 8000;
const HISTORY_LIMIT = 50;

const INPUT_LABELS = {
    speed: 'Speed (km/h)', carAhead: 'Opponent ahead on visible track',
    playerCorner: 'Player turn corner', opponentCorner: 'Opponent turn corner',
    playerPosition: 'Player track position', opponentPosition: 'Opponent track position',
    phase: 'Live Map section', cornerSpeed: 'Mapped corner speed', cornerShape: 'Mapped corner shape',
    linkedOpposite: 'Next linked corner turns the opposite way', linkedSameDirection: 'Next linked corner turns the same way',
    sequenceRemaining: 'Corners remaining in the mapped sequence',
    insideLine: 'Player inside, opponent off the inside', outsideLine: 'Player outside, opponent inside',
    opponentDistanceM: 'Estimated opponent distance (m)', closingOnOpponent: 'Closing on opponent',
    directlyBehindOpponent: 'Directly behind opponent',
    opponentCornerEtaS: 'Opponent estimated time to corner entry (s)',
    approachCornerSpeed: 'Upcoming mapped corner speed',
    approachCornerDirection: 'Upcoming corner direction',
    approachSequenceRemaining: 'Corners following in the labeled consecutive-corners sequence',
    approachLinkedOpposite: 'Next linked corner turns the opposite way',
} as const;
type Input = keyof typeof INPUT_LABELS;
type Inputs = PhraseMapContext & ReturnType<typeof getPhrasePositions> & {
    speed?: number; insideLine?: 0 | 1; outsideLine?: 0 | 1; opponentDistanceM?: number;
    closingOnOpponent?: 0 | 1; directlyBehindOpponent?: 0 | 1;
    opponentCornerEtaS?: number; approachCornerSpeed?: PhraseMapContext['cornerSpeed'];
    approachCornerDirection?: PhraseMapContext['cornerDirection'];
    approachSequenceRemaining?: number; approachLinkedOpposite?: 0 | 1 };
type PhraseVisionInput = Pick<TrackVisionDetection, 'capturedAt' | 'calibration' | 'birdsEyeScene'>;
export type PhraseConditionConnector = 'and' | 'or';
export class PhraseCondition {
    readonly conditionFit: boolean;
    readonly inputMissing: boolean;

    constructor(
        readonly input: Input,
        readonly operator: '>=' | '<=' | '<' | '=' | 'in',
        readonly value: number | string | readonly string[],
        actual?: Inputs[Input],
        // Joins this condition to the previous one; ignored for the first condition.
        readonly connector: PhraseConditionConnector = 'and',
    ) {
        this.inputMissing = actual === undefined;
        this.conditionFit = actual !== undefined && (operator === '>=' || operator === '<=' || operator === '<'
            ? typeof actual === 'number' && typeof value === 'number' && (operator === '>=' ? actual >= value : operator === '<' ? actual < value : actual <= value)
            : operator === 'in' ? Array.isArray(value) && value.includes(actual) : actual === value);
    }

    get description(): string {
        if (this.operator === 'in' || this.operator === '=') {
            const values = Array.isArray(this.value) ? this.value : [this.value];
            if (this.input === 'approachCornerDirection') {
                return `Upcoming corner turns ${values.join(' or ')}`;
            }
            const subject = this.input.startsWith('player') ? 'Player' : 'Opponent';
            if (this.input === 'playerCorner' || this.input === 'opponentCorner') {
                return `${subject} in a ${values.join(' or ')} turn corner`;
            }
            if (this.input === 'playerPosition' || this.input === 'opponentPosition') {
                return `${subject} near the ${values.map((value) => value === 'middle' ? 'middle of the track' : `${value} edge`).join(' or ')}`;
            }
        }
        return `${INPUT_LABELS[this.input]} ${this.operator} ${Array.isArray(this.value) ? this.value.join(' / ') : this.value}`;
    }

    evaluate(inputs: Inputs): PhraseCondition {
        // Keep the catalog and previously published snapshots independent of live updates.
        return new PhraseCondition(this.input, this.operator, this.value, inputs[this.input], this.connector);
    }
}

export type PhraseConditionNode = PhraseCondition | PhraseConditionGroup;

/** A parenthesized expression; its connector joins the whole group to its previous sibling. */
export class PhraseConditionGroup {
    readonly conditionFit: boolean;
    readonly inputMissing: boolean;

    constructor(
        readonly conditions: readonly PhraseConditionNode[],
        readonly connector: PhraseConditionConnector = 'and',
    ) {
        this.conditionFit = conditionsMatch(conditions);
        this.inputMissing = !this.conditionFit && conditions.some((condition) => condition.inputMissing);
    }

    evaluate(inputs: Inputs): PhraseConditionGroup {
        return new PhraseConditionGroup(this.conditions.map((condition) => condition.evaluate(inputs)), this.connector);
    }
}

function conditionsMatch(conditions: readonly PhraseConditionNode[]): boolean {
    // AND binds more tightly than OR: A OR B AND C means A OR (B AND C).
    let previousGroupMatched = false;
    let groupMatched = conditions[0]?.conditionFit ?? false;
    for (const condition of conditions.slice(1)) {
        if (condition.connector === 'or') {
            previousGroupMatched ||= groupMatched;
            groupMatched = condition.conditionFit;
        } else {
            groupMatched = groupMatched && condition.conditionFit;
        }
    }
    return previousGroupMatched || groupMatched;
}

function missingConditionInputs(conditions: readonly PhraseConditionNode[]): string[] {
    return conditions.flatMap((condition) => condition instanceof PhraseConditionGroup
        ? condition.conditionFit ? [] : missingConditionInputs(condition.conditions)
        : condition.inputMissing ? [INPUT_LABELS[condition.input]] : []);
}

export interface PhraseActionDefinition {
    sentence: string;
    actionConditions?: readonly PhraseConditionNode[];
}
export interface PhraseDefinition extends PhraseActionDefinition {
    id: string;
    name: string;
    description: string;
    holdMs: number;
    conditions: readonly PhraseConditionNode[];
    additionalActions?: readonly PhraseActionDefinition[];
}
export function getPhraseActions(phrase: PhraseDefinition) {
    return [phrase, ...(phrase.additionalActions ?? [])].map((action, index) => ({
        id: `${phrase.id}:say:${index}`, sentence: action.sentence, conditions: action.actionConditions,
    }));
}
const condition = (input: Input, operator: PhraseCondition['operator'], value: PhraseCondition['value'], connector: PhraseConditionConnector = 'and'): PhraseCondition => new PhraseCondition(input, operator, value, undefined, connector);
export const conditionGroup = (conditions: readonly PhraseConditionNode[], connector: PhraseConditionConnector = 'and'): PhraseConditionGroup => new PhraseConditionGroup(conditions, connector);
const following = [condition('speed', '>=', 30), condition('carAhead', '=', 1)];
const positioned = [
    condition('playerCorner', 'in', ['left', 'right']), condition('playerPosition', 'in', ['left', 'middle', 'right']),
    condition('opponentCorner', 'in', ['left', 'right']), condition('opponentPosition', 'in', ['left', 'middle', 'right']),
];

// Sentence metadata is compiled into child closures by createPhraseRoot.
// More specific tactics take priority when entering from the root.
export const PHRASE_DEFINITIONS: readonly PhraseDefinition[] = [
    {
        id: 'second-apex', name: 'Chicane overtake', holdMs: 0,
        description: 'Prepare for opposite linked corners, then speak when the opponent is less than 0.5 seconds from corner entry: go wide if the opponent is inside, or brake early and hold inside if the opponent is outside.',
        sentence: 'Go wide in the first turn. then take second apex if possible',
        conditions: [condition('opponentDistanceM', '<=', 10), condition('opponentCornerEtaS', '<=', 2),
            condition('approachCornerSpeed', '=', 'slow'), condition('approachSequenceRemaining', '>=', 1),
            condition('approachLinkedOpposite', '=', 1)],
        actionConditions: [condition('opponentCornerEtaS', '<', 0.5), conditionGroup([
            conditionGroup([condition('approachCornerDirection', '=', 'left'), condition('opponentPosition', '=', 'left')]),
            conditionGroup([condition('approachCornerDirection', '=', 'right'), condition('opponentPosition', '=', 'right')], 'or'),
        ])],
        additionalActions: [{
            sentence: 'brake early, Hold inside',
            actionConditions: [condition('opponentCornerEtaS', '<', 0.5), conditionGroup([
                conditionGroup([condition('approachCornerDirection', '=', 'right'), condition('opponentPosition', '=', 'left')]),
                conditionGroup([condition('approachCornerDirection', '=', 'left'), condition('opponentPosition', '=', 'right')], 'or'),
            ])],
        }],
    },
    {
        id: 'next-corner', name: 'Setting up the next corner', holdMs: 800,
        description: 'Use the outside of this turn to prepare for the inside of the next linked corner.',
        sentence: 'Set up the next corner: if you can establish overlap on the outside, stay alongside through this turn and leave room; your side becomes the inside at the following corner.',
        conditions: [...following, ...positioned, condition('phase', 'in', ['entry', 'middle']), condition('outsideLine', '=', 1), condition('linkedOpposite', '=', 1)],
    },
    {
        id: 'same-direction', name: 'Linked corners in the same direction', holdMs: 800,
        description: 'Preserve balance and room through linked turns in the same direction before building a passing run.',
        sentence: 'These corners turn the same way: keep the car balanced and leave room for the next apex. Build your passing run from the final exit once a clear lane opens.',
        conditions: [...following, ...positioned, condition('phase', 'in', ['entry', 'middle']), condition('linkedSameDirection', '=', 1)],
    },
    {
        id: 'sequence-exit', name: 'Exit into another corner', holdMs: 800,
        description: 'Keep the current exit controlled to prepare for the remaining corners in the sequence.',
        sentence: 'More corners follow in this sequence: keep this exit controlled and position for the next turn. Save the full exit-speed attack for the final corner and a clear passing lane.',
        conditions: [...following, ...positioned, condition('phase', '=', 'exit'), condition('sequenceRemaining', '>=', 1)],
    },
    {
        id: 's-bend', name: 'Direction change within a corner', holdMs: 800,
        description: 'Manage the steering transition through an S-bend and prioritize its final exit.',
        sentence: 'The road changes direction through this section: make a smooth steering transition, leave room alongside, and prioritize the final exit before attempting a pass.',
        conditions: [...following, ...positioned, condition('phase', 'in', ['entry', 'middle']), condition('cornerShape', '=', 's-bend')],
    },
    {
        id: 'tightening-corner', name: 'Tightening corner', holdMs: 800,
        description: 'Keep grip in reserve as the corner tightens and avoid a passing line that closes at the exit.',
        sentence: 'This corner tightens: keep some grip in reserve and delay full throttle until the steering can unwind. Avoid committing to a passing line that closes at the exit.',
        conditions: [...following, ...positioned, condition('phase', 'in', ['entry', 'middle']), condition('cornerShape', '=', 'tightening')],
    },
    {
        id: 'opening-corner', name: 'Opening corner', holdMs: 800,
        description: 'Build traction as the corner opens to prepare a pass from the exit.',
        sentence: 'This corner opens out: progressively unwind the steering and build traction. Use the exit to prepare a pass when a clear lane appears.',
        conditions: [...following, ...positioned, condition('phase', 'in', ['middle', 'exit']), condition('cornerShape', '=', 'opening')],
    },
    {
        id: 'hairpin-exit', name: 'Hairpin exit', holdMs: 800,
        description: 'Finish rotating through a hairpin before using exit traction to build a passing run.',
        sentence: 'Hairpin: finish rotating the car and feed in throttle as the steering opens. Prioritize traction, then use any exit-speed advantage when a passing lane is clear.',
        conditions: [...following, ...positioned, condition('phase', 'in', ['middle', 'exit']), condition('cornerShape', '=', 'hairpin')],
    },
    {
        id: 'inside-outbraking', name: 'Outbraking on the inside', holdMs: 800,
        description: 'Prepare an inside outbraking attempt at a slow corner while preserving room at the apex and exit.',
        sentence: 'Slow corner ahead: prepare an inside outbraking move. Establish overlap before turn-in, brake later only if you can still make the apex, and leave exit room.',
        conditions: [...following, ...positioned, condition('phase', '=', 'entry'), condition('cornerSpeed', '=', 'slow'), condition('insideLine', '=', 1)],
    },
    {
        id: 'around-outside', name: 'Around the outside', holdMs: 800,
        description: 'Carry momentum around the outside of a fast corner when overlap and space permit.',
        sentence: 'Faster corner: carry momentum around the outside only if you have overlap and room. Hold your line and allow space for the car inside.',
        conditions: [...following, ...positioned, condition('phase', '=', 'entry'), condition('cornerSpeed', '=', 'fast'), condition('outsideLine', '=', 1)],
    },
    {
        id: 'switchback', name: 'Switchback', holdMs: 800,
        description: 'Prepare to cut back underneath the inside car if it runs wide and leaves a gap.',
        sentence: 'Prepare a switchback: let the inside car commit, delay your turn-in, then cut back underneath if they run wide and the gap opens.',
        conditions: [...following, ...positioned, condition('phase', 'in', ['entry', 'middle']), condition('cornerSpeed', '=', 'slow'), condition('outsideLine', '=', 1)],
    },
    {
        id: 'better-exit', name: 'Better exit pass', holdMs: 800,
        description: 'Build exit speed with smooth steering and throttle to prepare a pass in a clear lane.',
        sentence: 'Build a better exit: open the steering and feed in throttle smoothly. Use any speed advantage to draw alongside once there is a clear passing lane.',
        conditions: [...following, ...positioned, condition('phase', '=', 'exit'), condition('cornerSpeed', 'in', ['slow', 'fast'])],
    },
    {
        id: 'slipstream', name: 'Slipstream', holdMs: 800,
        description: 'Move behind a nearby opponent on a straight to build a run when the gap is steady or increasing.',
        sentence: 'Use the slipstream on this straight: tuck in behind to build a run, then pull out when you are closing and there is clear space before braking.',
        conditions: [condition('speed', '>=', 80), condition('carAhead', '=', 1), condition('phase', '=', 'straight'),
            condition('opponentDistanceM', '<=', 10), condition('closingOnOpponent', '=', 0), condition('directlyBehindOpponent', '=', 0)],
    },
    {
        id: 'pressure-feint', name: 'Pressure and a feint', holdMs: 800,
        description: 'Show a controlled attack before braking and use an opening if the opponent reacts.',
        sentence: 'Apply pressure: show a possible attack before the braking zone to invite a defensive line, then exploit an opening, mistake or compromised exit if they react. Keep your move controlled.',
        conditions: [...following, ...positioned, condition('phase', '=', 'entry'), condition('cornerSpeed', 'in', ['slow', 'fast']), condition('insideLine', '=', 0), condition('outsideLine', '=', 0)],
    },
];

export type ClosureStatus = 'Missing input' | 'Not matched' | 'Confirming' | 'Waiting for action' | 'Active' | 'Cooldown';
export interface PhraseEvent { id: number; ruleId: string; sentence: string; timestamp: number }
export interface PhraseSnapshot {
    telemetryReady: boolean;
    visionReady: boolean;
    mapReady: boolean;
    mapContext: PhraseMapContext;
    root: NodeSnapshot;
    // Compatibility summary for phrase events, overlays and speech. The UI renders root.
    closures: Array<{
        id: string; name: string; description: string; holdMs: number;
        status: ClosureStatus; missing: string[]; conditions: readonly PhraseConditionNode[];
        actionConditions?: readonly PhraseConditionNode[];
        actions: Array<Pick<Action<PhraseContext>, 'name' | 'description'>>;
    }>;
    state: { current: string; description: string; kind: 'closure' | 'action'; path: string[] };
    events: PhraseEvent[];
}
interface PhraseMemory { since?: number; clearSince?: number; armed: boolean; lastEmitted?: number }
export interface PhraseContext {
    now: number;
    emit: boolean;
    matched: ReadonlyMap<string, boolean>;
    actionMatched: ReadonlyMap<string, boolean>;
    memory: Map<string, PhraseMemory>;
    entryConditions: ReadonlyMap<string, readonly PhraseConditionNode[]>;
    actionConditions: ReadonlyMap<string, readonly PhraseConditionNode[]>;
    inspections: ReadonlyMap<string, NodeInspection>;
    candidate?: string;
    publish: (phrase: PhraseDefinition, sentence?: string) => void;
}

function snapshotConditions(conditions: readonly PhraseConditionNode[]): ConditionSnapshot[] {
    return conditions.map((condition) => ({
        description: condition instanceof PhraseConditionGroup ? 'Group' : condition.description,
        conditionFit: condition.conditionFit, inputMissing: condition.inputMissing, connector: condition.connector,
        ...(condition instanceof PhraseConditionGroup ? { conditions: snapshotConditions(condition.conditions) } : {}),
    }));
}

export function createPhraseRoot(phrases: readonly PhraseDefinition[] = PHRASE_DEFINITIONS): Closure<PhraseContext> {
    return new Closure('root',
        'Select the first matching guide after its entry hold time. Each guide has an 8 s cooldown and requires a 0.5 s clear period before repeating. An entered closure stays selected until its exit to root action runs.',
        describeCondition(() => true, 'Always eligible'), phrases.map((phrase) => {
            const say = getPhraseActions(phrase).map((action) => new Action<PhraseContext>(
                'say phrase', action.sentence, (context) => context.publish(phrase, action.sentence),
                // Once a speech action runs, only the exit may run in this entry.
                describeCondition((context, state) => state.current === state.closure
                    && (!action.conditions || (context.emit && context.actionMatched.get(action.id) === true)),
                    action.conditions
                        ? (context) => snapshotConditions(context.actionConditions.get(action.id) ?? [])
                        : 'Always eligible')));
            return new Closure<PhraseContext>(
                phrase.name,
                phrase.description,
                describeCondition((context) => {
                    if (!context.matched.get(phrase.id) || (context.candidate && context.candidate !== phrase.id)) return false;
                    context.candidate = phrase.id;
                    const memory = context.memory.get(phrase.id)!;
                    memory.since ??= context.now;
                    return context.emit && memory.armed && context.now - memory.since >= phrase.holdMs
                        && (memory.lastEmitted === undefined || context.now - memory.lastEmitted >= COOLDOWN_MS);
                }, (context) => snapshotConditions(context.entryConditions.get(phrase.id) ?? [])),
                [...say, Action.exitToRoot(describeCondition((_context, state) =>
                    say.some((action) => state.current === action), 'The phrase action has run'))],
                { id: phrase.id, inspect: (context) => ({
                    ...context.inspections.get(phrase.id),
                    fields: { 'Hold for': `${phrase.holdMs / 1000} s`, Cooldown: `${COOLDOWN_MS / 1000} s`, 'Clear period': `${REARM_MS / 1000} s` },
                }) },
            );
        }));
}
const finite = (value: unknown): number | undefined => typeof value === 'number' && Number.isFinite(value) ? value : undefined;
const nonnegative = (value: unknown): number | undefined => {
    const number = finite(value);
    return number !== undefined && number >= 0 ? number : undefined;
};

function readInputs(sample: StandardTelemetrySample, scene: TrackVisionDetection['birdsEyeScene'], map: PhraseMapContext): Inputs {
    const velocity = [sample.Physics_velocity_x, sample.Physics_velocity_y, sample.Physics_velocity_z];
    const { carAhead, playerCorner, opponentCorner, playerPosition, opponentPosition,
        opponentDistanceM, opponentLateralOffsetM } = getPhrasePositions(scene);
    const sameCorner = playerCorner && playerCorner === opponentCorner && playerPosition && opponentPosition;
    return {
        speed: nonnegative(sample.Physics_speed_kmh) ?? (velocity.every((v) => finite(v) !== undefined)
            ? Math.hypot(...velocity as number[]) * 3.6 : undefined),
        carAhead, playerCorner, opponentCorner, playerPosition, opponentPosition, opponentDistanceM,
        // A one-meter lateral tolerance represents being tucked directly behind.
        directlyBehindOpponent: opponentLateralOffsetM === undefined ? undefined : Number(Math.abs(opponentLateralOffsetM) <= 1) as 0 | 1,
        ...map,
        // The inside edge is left in a left turn and right in a right turn.
        insideLine: sameCorner
            ? Number(playerPosition === playerCorner && opponentPosition !== opponentCorner) as 0 | 1 : undefined,
        outsideLine: sameCorner
            ? Number(playerPosition !== 'middle' && playerPosition !== playerCorner && opponentPosition === opponentCorner) as 0 | 1 : undefined,
    };
}

/** Local rules combining published vision with the tagged Live Map and live lap position. */
export class PhraseEngine {
    readonly root: Closure<PhraseContext>;
    state: State<PhraseContext>;
    private sample: StandardTelemetrySample = {};
    private receivedAt = -Infinity;
    private vision: PhraseVisionInput | null = null;
    private visionAfter = -Infinity;
    private opponentDistanceM?: number;
    private closingOnOpponent?: 0 | 1;
    private memory = new Map<string, PhraseMemory>();
    private events: PhraseEvent[] = [];
    private nextId = 0;
    private map: CircuitMapDto | null = null;
    private mapContext = createPhraseMapContext(null);
    private sectionId?: string;
    private game?: string;
    private opponentMotion = new PhraseOpponentMotion();
    private phraseClosures = new Map<string, Closure<PhraseContext>>();

    constructor(
        private readonly phrases: readonly PhraseDefinition[] = PHRASE_DEFINITIONS,
        createRoot: (phrases: readonly PhraseDefinition[]) => Closure<PhraseContext> = createPhraseRoot,
    ) {
        this.root = createRoot(phrases);
        const register = (node: Closure<PhraseContext> | Action<PhraseContext>) => {
            if (node instanceof Closure) {
                if (node.metadata.id) this.phraseClosures.set(node.metadata.id, node);
                node.children.forEach(register);
            }
        };
        register(this.root);
        this.state = new State(this.root);
    }

    receiveMap(map: CircuitMapDto | null, now: number): PhraseSnapshot {
        if (map !== this.map) {
            this.map = map;
            this.mapContext = createPhraseMapContext(map);
            this.memory.forEach((memory) => { memory.since = undefined; memory.armed = true; });
            this.sectionId = undefined;
            this.opponentMotion.reset();
        }
        return this.evaluate(now);
    }

    reset(now: number) {
        this.sample = {};
        this.receivedAt = -Infinity;
        this.vision = null;
        this.visionAfter = now;
        this.opponentDistanceM = undefined;
        this.closingOnOpponent = undefined;
        this.memory.clear();
        this.state = new State(this.root);
        this.events = [];
        this.sectionId = undefined;
        this.game = undefined;
        this.opponentMotion.reset();
    }

    receiveTelemetry(event: LiveTelemetryEvent, now: number): PhraseSnapshot {
        if (event.type !== 'frame') {
            this.reset(now);
        } else {
            if (this.game !== event.update.game || this.sample.Static_track !== event.sample.Static_track) this.opponentMotion.reset();
            this.game = event.update.game;
            // A gap cannot count toward the continuous hold, even without a UI timer.
            if (now - this.receivedAt > TELEMETRY_MAX_AGE_MS) {
                this.memory.forEach((memory) => { memory.since = undefined; });
            }
            // Paused, replay and off frames cannot generate live guidance.
            this.sample = event.telemetryStatus === 2 ? event.sample : {};
            this.receivedAt = event.telemetryStatus === 2 ? now : -Infinity;
            this.opponentMotion.update(this.sample, now, TELEMETRY_MAX_AGE_MS);
        }
        return this.evaluate(now, true);
    }

    receiveVision(vision: PhraseVisionInput | null, now: number): PhraseSnapshot {
        const sameCalibration = JSON.stringify(vision?.calibration) === JSON.stringify(this.vision?.calibration);
        // Repeated or out-of-order captures cannot establish a new relative-motion sample.
        if (vision && this.vision && sameCalibration && this.vision.capturedAt <= now
            && vision.capturedAt <= this.vision.capturedAt) return this.evaluate(now);
        // A fresh result cannot retroactively fill a capture gap when no timer ran.
        if (!this.vision || now - this.vision.capturedAt > VISION_MAX_AGE_MS
            || !sameCalibration) {
            this.memory.forEach((memory) => { memory.since = undefined; });
            this.opponentDistanceM = undefined;
        }
        const fresh = vision && vision.capturedAt >= this.visionAfter && vision.capturedAt <= now
            && now - vision.capturedAt <= VISION_MAX_AGE_MS;
        const distance = fresh ? getPhrasePositions(vision.birdsEyeScene).opponentDistanceM : undefined;
        this.closingOnOpponent = distance !== undefined && this.opponentDistanceM !== undefined
            ? Number(distance < this.opponentDistanceM) as 0 | 1 : undefined;
        this.opponentDistanceM = distance;
        this.vision = vision;
        return this.evaluate(now);
    }

    private step(context: PhraseContext): void {
        // State records action failures for the UI instead of rejecting in a live
        // subscription. Errors in conditions still propagate to the caller.
        void this.state.step(context, () => undefined);
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
        inputs.closingOnOpponent = visionReady ? this.closingOnOpponent : undefined;
        const opponent = telemetryReady && sameMap ? this.opponentMotion.at(now, TELEMETRY_MAX_AGE_MS) : undefined;
        const approach = opponent && this.mapContext.approaching(opponent.position);
        if (opponent && approach) {
            inputs.opponentCornerEtaS = lapDistance(opponent.position, approach.cornerStartPosition) / opponent.rate;
            inputs.approachCornerSpeed = approach.cornerSpeed;
            inputs.approachCornerDirection = approach.cornerDirection;
            inputs.approachSequenceRemaining = approach.sequenceRemaining;
            inputs.approachLinkedOpposite = approach.linkedOpposite;
        }
        const matched = new Map<string, boolean>();
        const actionMatched = new Map<string, boolean>();
        const evaluatedActionConditions = new Map<string, readonly PhraseConditionNode[]>();
        const evaluated = this.phrases.map((phrase) => {
            const memory = this.memory.get(phrase.id) ?? { armed: true };
            this.memory.set(phrase.id, memory);
            const conditions = phrase.conditions.map((condition) => condition.evaluate(inputs));
            const actions = getPhraseActions(phrase).map((action) => {
                const conditions = action.conditions?.map((condition) => condition.evaluate(inputs));
                actionMatched.set(action.id, telemetryReady && (conditions === undefined || conditionsMatch(conditions)));
                evaluatedActionConditions.set(action.id, conditions ?? []);
                return { conditions };
            });
            const actionConditions = actions[0].conditions;
            const conditionFit = conditionsMatch(conditions);
            const missing = conditionFit ? [] : missingConditionInputs(conditions);
            const fits = telemetryReady && conditionFit;
            matched.set(phrase.id, fits);
            if (!fits) {
                memory.since = undefined;
                memory.clearSince ??= now;
                if (now - memory.clearSince >= REARM_MS) memory.armed = true;
            } else {
                // Also rearm when the next frame arrives after a sustained false condition.
                if (memory.clearSince !== undefined && now - memory.clearSince >= REARM_MS) memory.armed = true;
                memory.clearSince = undefined;
            }
            return { phrase, memory, missing, conditions, actionConditions };
        });
        const inspections = new Map<string, NodeInspection>();
        const context: PhraseContext = {
            now, emit, matched, actionMatched, memory: this.memory,
            entryConditions: new Map(evaluated.map(({ phrase, conditions }) => [phrase.id, conditions])),
            actionConditions: evaluatedActionConditions,
            inspections,
            publish: (phrase, sentence = phrase.sentence) => {
                this.events = [...this.events, { id: ++this.nextId, ruleId: phrase.id, sentence, timestamp: now }].slice(-HISTORY_LIMIT);
                const memory = this.memory.get(phrase.id)!;
                memory.lastEmitted = now;
                memory.armed = false;
            },
        };
        // Only State chooses a closure and invokes its action. A completed action is
        // retained until the next step; the sentence closure's next action exits to root.
        const previousClosure = this.state.closure;
        this.step(context);
        if (previousClosure !== this.root && this.state.current === this.root) this.step(context);
        const closures = evaluated.flatMap(({ phrase, memory, missing, conditions, actionConditions }) => {
            const closure = this.phraseClosures.get(phrase.id);
            if (!closure) return [];
            const selected = context.candidate === phrase.id || this.state.path.includes(closure);
            let status: ClosureStatus = missing.length || !telemetryReady ? 'Missing input' : 'Not matched';
            if (this.state.closure === closure && (phrase.actionConditions || phrase.additionalActions?.length)) {
                status = this.state.current instanceof Action ? 'Active' : 'Waiting for action';
            } else if (selected && matched.get(phrase.id)) {
                const cooldown = memory.lastEmitted !== undefined && now - memory.lastEmitted < COOLDOWN_MS;
                status = !memory.armed ? 'Active' : memory.since === undefined || now - memory.since < phrase.holdMs ? 'Confirming' : cooldown ? 'Cooldown' : 'Confirming';
            } else if (context.candidate !== phrase.id) {
                memory.since = undefined;
            }
            inspections.set(phrase.id, { status });
            return [{
                id: phrase.id, name: closure.name, description: closure.description, holdMs: phrase.holdMs,
                status, missing, conditions, actionConditions,
                actions: closure.children.filter((child) => child instanceof Action).map(({ name, description }) => ({ name, description })),
            }];
        });
        return {
            telemetryReady, visionReady, mapReady: sameMap && this.mapContext.ready, mapContext, closures, events: this.events,
            root: this.state.snapshot(context),
            state: { current: this.state.current.name, description: this.state.current.description,
                kind: this.state.current instanceof Closure ? 'closure' : 'action', path: this.state.path.map((node) => node.name) },
        };
    }
}
