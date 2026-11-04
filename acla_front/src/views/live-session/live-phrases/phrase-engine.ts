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
import type { DriverCornerSummary } from '../event-log/corner-summary';
import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import { readReferenceGuidance, REFERENCE_CUE_COOLDOWN_MS, REFERENCE_POSITION_MAX_AGE_MS, type ReferenceCue, type ReferencePedalZone, type ReferenceSteeringPhase, type ReferencePositionSample } from './phrase-reference-guidance';
import { ReferenceMovement, REFERENCE_MOVEMENTS, type ReferenceMovementCue, type MovementDirection } from './phrase-reference-movement';
import { copyPhraseMap, copyPhraseReference, normalizeDrivingAggression, REFERENCE_DRIVING_AGGRESSION } from './phrase-driving-reference';

export const TELEMETRY_MAX_AGE_MS = 1500;
export const REARM_MS = 500;
export const COOLDOWN_MS = 8000;
const HISTORY_LIMIT = 50;

const INPUT_LABELS = {
    speed: 'Speed (km/h)', carAhead: 'Opponent ahead on visible track',
    playerCorner: 'Player turn corner', opponentCorner: 'Opponent turn corner',
    playerPosition: 'Player track position', opponentPosition: 'Opponent track position',
    phase: 'Live Map section', cornerSpeed: 'Mapped corner speed', cornerShape: 'Mapped corner shape',
    linkedOpposite: 'Next linked corner turns the opposite way',
    sequenceRemaining: 'Corners remaining in the mapped sequence',
    insideLine: 'Player inside, opponent off the inside', outsideLine: 'Player outside, opponent inside',
    opponentDistanceM: 'Estimated opponent distance (m)', closingOnOpponent: 'Closing on opponent',
    directlyBehindOpponent: 'Directly behind opponent',
    opponentCornerEtaS: 'Opponent estimated time to corner entry (s)',
    approachCornerSpeed: 'Upcoming mapped corner speed',
    approachCornerDirection: 'Upcoming corner direction',
    approachSequenceRemaining: 'Upcoming mapped consecutive-corners sequence',
    approachLinkedOpposite: 'Next linked corner turns the opposite way',
    approachCornerInSequence: 'Upcoming corner is inside a Live Map consecutive corners label',
    betterCornerExitSpeed: 'Previous corner exits show the player has better exit speed than the car ahead',
    approachLongStraightAfterCorner: 'Upcoming slow corner is followed by a long straight',
    referenceCue: 'Reference-line guidance',
    driverOnStraight: 'Driver inside a Live Map straight or long straight',
    referenceBraking: 'Reference braking or brake correction needed',
    referenceCornering: 'Inside an interpolated reference brake or throttle zone',
    referencePedalZone: 'Interpolated reference pedal zone',
    referenceSteeringPhase: 'Reference steering phase',
} as const;
type Input = keyof typeof INPUT_LABELS;
type Inputs = PhraseMapContext & ReturnType<typeof getPhrasePositions> & {
    speed?: number; referenceCue?: ReferenceCue | 'none'; driverOnStraight?: 0 | 1; referenceBraking?: 0 | 1; referenceCornering?: 0 | 1;
    referencePedalZone?: ReferencePedalZone;
    referenceSteeringPhase?: ReferenceSteeringPhase | 'none';
    insideLine?: 0 | 1; outsideLine?: 0 | 1; opponentDistanceM?: number;
    closingOnOpponent?: 0 | 1; directlyBehindOpponent?: 0 | 1;
    opponentCornerEtaS?: number; approachCornerSpeed?: PhraseMapContext['cornerSpeed'];
    approachCornerDirection?: PhraseMapContext['cornerDirection'];
    approachSequenceRemaining?: number; approachLinkedOpposite?: 0 | 1; approachCornerInSequence?: 0 | 1;
    betterCornerExitSpeed?: 0 | 1; approachLongStraightAfterCorner?: 0 | 1 };
type PhraseVisionInput = Pick<TrackVisionDetection, 'capturedAt' | 'calibration' | 'birdsEyeScene' | 'position' | 'telemetry'>;
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
        if (this.input === 'referenceCue') {
            const descriptions: Partial<Record<ReferenceCue, string>> = {
                brake: 'Brake demand for the target speed is at least 10% and the brake is not yet applied',
                steerSoon: 'Reference turn-in is approaching or reference steering is turning into the corner',
                throttle: 'Throttle demand for the target speed is at least 15% and the throttle is not yet applied, near the line and clear of braking',
                moreBrake: 'Speed is at least 5 km/h above the interpolated reference and live deceleration is at least 0.1 m/s² lower, with brake applied below full pressure',
                releaseBrake: 'The brake is applied but demand for the target speed is below 10%',
                lessBrake: 'Speed is at least 5 km/h below the interpolated reference and live deceleration is at least 0.1 m/s² higher, with brake applied',
                moreThrottle: 'Applied throttle is at least 15 points below demand from reference throttle, acceleration and target speed, near the line and clear of braking',
                lessThrottle: 'Speed is at least 5 km/h above the local reference and the driver is moving outward from the reference line, with throttle applied',
            };
            const movement = REFERENCE_MOVEMENTS.find(({ cue }) => cue === this.value);
            return movement ? `Move to the ${movement.direction}: ${movement.description}`
                : descriptions[this.value as ReferenceCue] ?? INPUT_LABELS.referenceCue;
        }
        if (this.operator === 'in' || this.operator === '=') {
            if (this.input === 'referenceSteeringPhase') return this.value === 'prepare'
                ? 'Reference turn-in starts in 0.6–1.8 seconds'
                : this.value === 'easeBrake' ? 'Reference deceleration is easing before turn-in'
                    : 'Reference steering has reached the corner turn-in threshold';
            if (this.input === 'driverOnStraight') return INPUT_LABELS.driverOnStraight;
            if (this.input === 'referenceBraking' && this.value === 0) return 'Reference is not braking and no brake correction is needed';
            if (this.input === 'referenceBraking' && this.value === 1) return INPUT_LABELS.referenceBraking;
            if (this.input === 'referenceCornering' && this.value === 1) return INPUT_LABELS.referenceCornering;
            if (this.input === 'referencePedalZone') return `Inside the interpolated reference ${this.value} zone`;
            const values = Array.isArray(this.value) ? this.value : [this.value];
            if (this.input === 'approachCornerDirection') {
                return `Upcoming corner turns ${values.join(' or ')}`;
            }
            if (this.input === 'approachCornerInSequence' && this.value === 0) {
                return 'Corner is not inside a consecutive corners label of the Live Map';
            }
            if (this.input === 'betterCornerExitSpeed' || this.input === 'approachLongStraightAfterCorner') {
                return INPUT_LABELS[this.input];
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
    /** Additional wordings, chosen with the same probability as sentence on each execution. */
    sentenceAlternatives?: readonly string[];
    actionConditions?: readonly PhraseConditionNode[];
}
export interface PhraseDefinition extends PhraseActionDefinition {
    id: string;
    name: string;
    description: string;
    holdMs: number;
    referenceCue?: ReferenceCue;
    movementDirection?: MovementDirection;
    conditions: readonly PhraseConditionNode[];
    additionalActions?: readonly PhraseActionDefinition[];
}
export function getPhraseActions(phrase: PhraseDefinition) {
    return [phrase, ...(phrase.additionalActions ?? [])].map((action, index) => ({
        id: `${phrase.id}:say:${index}`, sentence: action.sentence, conditions: action.actionConditions,
        sentences: [action.sentence, ...(action.sentenceAlternatives ?? [])],
    }));
}
const condition = (input: Input, operator: PhraseCondition['operator'], value: PhraseCondition['value'], connector: PhraseConditionConnector = 'and'): PhraseCondition => new PhraseCondition(input, operator, value, undefined, connector);
const movementConditions = [condition('driverOnStraight', '=', 1), condition('referenceBraking', '=', 0)];
const MOVEMENT_CLOSURE_ID = 'reference-turn-to-line';
const corneringConditions = [condition('referenceCornering', '=', 1)];
const brakeCues: readonly ReferenceCue[] = ['brake', 'moreBrake', 'releaseBrake', 'lessBrake', 'steerSoon'];
const throttleCues: readonly ReferenceCue[] = ['throttle', 'moreThrottle', 'lessThrottle'];
const CORNERING_CLOSURE_ID = 'cornering';
const pedalZones = [
    { zone: 'brake', id: 'reference-brake-zone', name: 'Brake zone', cues: brakeCues },
    { zone: 'throttle', id: 'reference-throttle-zone', name: 'Throttle zone', cues: throttleCues },
] as const;
export const conditionGroup = (conditions: readonly PhraseConditionNode[], connector: PhraseConditionConnector = 'and'): PhraseConditionGroup => new PhraseConditionGroup(conditions, connector);
const following = [condition('speed', '>=', 30), condition('carAhead', '=', 1)];
const positioned = [
    condition('playerCorner', 'in', ['left', 'right']), condition('playerPosition', 'in', ['left', 'middle', 'right']),
    condition('opponentCorner', 'in', ['left', 'right']), condition('opponentPosition', 'in', ['left', 'middle', 'right']),
];
const exitSpeedAdvice: PhraseActionDefinition = {
    sentence: 'From pervious laps statistic, you generally have better exit speed. long straight after this corner, consider back up a little bit to have space to overtake at exit.',
    actionConditions: [condition('betterCornerExitSpeed', '=', 1), condition('opponentDistanceM', '<=', 2),
        condition('approachLongStraightAfterCorner', '=', 1)],
};

// Sentence metadata is compiled into child closures by createPhraseRoot.
// More specific tactics take priority when entering from the root.
export const PHRASE_DEFINITIONS: readonly PhraseDefinition[] = [
    {
        id: 'second-apex', name: 'Chicane overtake', holdMs: 0,
        description: 'Prepare for opposite linked corners, then speak when the opponent is less than 0.5 seconds from corner entry: go wide if the opponent is inside, or brake early and hold inside if the opponent is outside.',
        sentence: 'Go wide in the first turn. then take second apex if possible',
        conditions: [condition('approachSequenceRemaining', '>=', 1), condition('approachCornerSpeed', '=', 'slow'),
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
        }, exitSpeedAdvice],
    },
    {
        id: 'one-tight-slow-corner', name: 'One Tight Slow Corner', holdMs: 0,
        description: 'Follow an opponent toward the next corner labeled slow on the Live Map and outside any consecutive corners label: slipstream to a late brake when the opponent is outside, or brake earlier and clip the apex behind a defending opponent in the middle.',
        sentence: 'Opponent didnt defend, overtake from inside is possible here',
        conditions: [condition('approachCornerSpeed', '=', 'slow'), condition('approachCornerInSequence', '=', 0)],
        actionConditions: [
            condition('opponentDistanceM', '<=', 2),
            conditionGroup([
                conditionGroup([condition('opponentPosition', '=', 'left'), condition('approachCornerDirection', '=', 'right')]),
                conditionGroup([condition('opponentPosition', '=', 'right'), condition('approachCornerDirection', '=', 'left')], 'or'),
            ]),
        ],
        additionalActions: [{
            sentence: 'Opponent is defending, Pressure is on',
            actionConditions: [
                conditionGroup([condition('opponentPosition', '=', 'middle'), condition('playerPosition', '=', 'right'),
                    condition('approachCornerDirection', '=', 'right')]),
                conditionGroup([condition('opponentPosition', '=', 'middle'), condition('playerPosition', '=', 'left'),
                    condition('approachCornerDirection', '=', 'left')], 'or'),
            ],
        }, exitSpeedAdvice],
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
    ...([
        ['brake', 'Brake', 'Brake.', 'Apply the brake to follow the speed profile adjusted by Driving aggression.'],
        ['moreBrake', 'More brake', 'More brake.', 'Increase pressure when faster than the interpolated speed target and decelerating less than the reference.'],
        ['releaseBrake', 'Release the brake', 'Release the brake.', 'Release the pedal when the adjusted speed target no longer requires braking.'],
        ['lessBrake', 'Less brake', 'Less brake.', 'Reduce pressure when slower than the interpolated speed target and decelerating more than the reference.'],
    ] as const).map(([referenceCue, name, sentence, description]) => ({
        id: `reference-${referenceCue}`, referenceCue, name, sentence, description,
        holdMs: referenceCue === 'moreBrake' || referenceCue === 'lessBrake' ? 300 : 0,
        conditions: [condition('speed', '>=', 20), condition('referencePedalZone', '=', 'brake'), condition('referenceCue', '=', referenceCue)],
    })),
    {
        id: 'reference-steerSoon', referenceCue: 'steerSoon', name: 'Steer soon', holdMs: 0,
        description: 'Prepare for reference turn-in inside the brake zone, ease off as reference deceleration decreases, then turn with the reference steering.',
        sentence: 'turn, slowly lift up the brake',
        conditions: [condition('speed', '>=', 20), condition('referencePedalZone', '=', 'brake'), condition('referenceCue', '=', 'steerSoon')],
        actionConditions: [condition('referenceSteeringPhase', '=', 'easeBrake')],
        additionalActions: [
            { sentence: 'Turn', actionConditions: [condition('referenceSteeringPhase', '=', 'turn')] },
        ],
    },
    ...REFERENCE_MOVEMENTS.map(({ cue, direction, name, sentence, sentenceAlternatives, description }) => ({
        id: `reference-${cue}`, referenceCue: cue, movementDirection: direction, name, sentence, sentenceAlternatives, description, holdMs: 0,
        conditions: [condition('speed', '>=', 20), ...movementConditions, condition('referenceCue', '=', cue)],
    })),
    ...([
        ['throttle', 'Throttle', 'Throttle.', 'Apply throttle using reference throttle and acceleration to follow the speed profile adjusted by Driving aggression.'],
        ['moreThrottle', 'More throttle', 'More throttle.', 'Increase throttle when reference acceleration response indicates more is needed for the adjusted speed target.'],
        ['lessThrottle', 'Less throttle', 'Less throttle.', 'Reduce throttle when faster than the local reference and moving outward from its line, using reference curvature to identify the outside of the corner.'],
    ] as const).map(([referenceCue, name, sentence, description]) => ({
        id: `reference-${referenceCue}`, referenceCue, name, sentence, description, holdMs: 0,
        conditions: [condition('speed', '>=', 20), condition('referencePedalZone', '=', 'throttle'), condition('referenceCue', '=', referenceCue)],
    })),
];

export type ClosureStatus = 'Missing input' | 'Not matched' | 'Confirming' | 'Waiting for action' | 'Active' | 'Cooldown';
export interface PhraseEvent {
    id: number; ruleId: string; sentence: string; timestamp: number; expiresAt?: number;
    steeringPhase?: ReferenceSteeringPhase;
}
export interface PhraseSnapshot {
    telemetryReady: boolean;
    visionReady: boolean;
    mapReady: boolean;
    referenceLoaded: boolean;
    referenceSteeringPhase?: ReferenceSteeringPhase;
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
    referenceMovementAllowed?: boolean;
    referencePedalZone?: ReferencePedalZone;
    referenceSpeechReady?: boolean;
    referenceThrottleOutsideCorner?: boolean;
    driverLongStraightIds?: readonly string[];
    driverCornerIds?: readonly string[];
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

function createInsideLongStraightClosure(phrase: PhraseDefinition, say: Action<PhraseContext>[]): Closure<PhraseContext> {
    const enteredStraights = new WeakMap<State<PhraseContext>, string>();
    const entry = ({ driverLongStraightIds }: PhraseContext) => Boolean(driverLongStraightIds?.length);
    const exit = ({ driverLongStraightIds }: PhraseContext, state: State<PhraseContext>) => {
        const entered = enteredStraights.get(state);
        return entered !== undefined && driverLongStraightIds !== undefined && !driverLongStraightIds.includes(entered);
    };
    return new Closure('Inside the Long Straight',
        `Stay inside the driver’s current long straight, then return to ${phrase.name} when the driver leaves it.`,
        describeCondition((context, state) => {
            if (state.current !== state.closure || !entry(context)) return false;
            enteredStraights.set(state, context.driverLongStraightIds![0]);
            return true;
        }, (context) => [{
            description: 'Driver is inside a long straight', conditionFit: entry(context),
            inputMissing: context.driverLongStraightIds === undefined,
        }]), [
            Action.exitToParent(describeCondition(exit, (context, state) => [{
                description: 'Driver is no longer inside that long straight', conditionFit: exit(context, state),
                inputMissing: context.driverLongStraightIds === undefined,
            }])),
            ...say.map((action) => new Action(action.name, action.description, action.run,
                describeCondition((context, state) => entry(context) && action.condition(context, state), action.condition.describe!), action.metadata)),
        ], { id: `${phrase.id}:inside-long-straight` });
}

function createInsideCornerClosure(): Closure<PhraseContext> {
    const enteredCorners = new WeakMap<State<PhraseContext>, string>();
    const entry = ({ driverCornerIds }: PhraseContext) => Boolean(driverCornerIds?.length);
    const exit = ({ driverCornerIds }: PhraseContext, state: State<PhraseContext>) => {
        const entered = enteredCorners.get(state);
        return entered !== undefined && driverCornerIds !== undefined && !driverCornerIds.includes(entered);
    };
    return new Closure('Inside the Corner',
        'Stay inside the driver’s current corner, then return to One Tight Slow Corner when the driver leaves it.',
        describeCondition((context, state) => {
            if (state.current !== state.closure || !entry(context)) return false;
            enteredCorners.set(state, context.driverCornerIds![0]);
            return true;
        }, (context) => [{
            description: 'Driver is inside a corner', conditionFit: entry(context),
            inputMissing: context.driverCornerIds === undefined,
        }]), [
            Action.exitToParent(describeCondition(exit, (context, state) => [{
                description: 'Driver is no longer inside that corner', conditionFit: exit(context, state),
                inputMissing: context.driverCornerIds === undefined,
            }])),
        ], { id: 'one-tight-slow-corner:inside-corner' });
}

export function createPhraseRoot(phrases: readonly PhraseDefinition[] = PHRASE_DEFINITIONS): Closure<PhraseContext> {
    const closures = phrases.map((phrase) => {
            const say: Action<PhraseContext>[] = getPhraseActions(phrase).map((action, index) => new Action<PhraseContext>(
                'say phrase', action.sentence, (context) => {
                    const index = action.sentences.length > 1 ? Math.floor(Math.random() * action.sentences.length) : 0;
                    context.publish(phrase, action.sentences[index]);
                },
                // Turn preparation stays entered so each later phase can speak once.
                describeCondition((context, state) => (phrase.referenceCue === 'steerSoon'
                    ? context.referenceSpeechReady === true
                        && !say.slice(index + 1).some((later) => state.getExecution(later).status !== 'idle')
                    : state.current === state.closure)
                    && (!action.conditions || (context.emit && context.actionMatched.get(action.id) === true)),
                    action.conditions
                        ? (context) => snapshotConditions(context.actionConditions.get(action.id) ?? [])
                        : 'Always eligible'),
                action.sentences.length > 1 ? { inspect: () => ({ fields: {
                    'Phrase selection': 'Random (equal chance)',
                    ...Object.fromEntries(action.sentences.slice(1).map((sentence, index) => [`Alternative ${index + 1}`, sentence])),
                } }) } : {}));
            const parentSay = phrase.id === 'one-tight-slow-corner' ? say.slice(1, 2)
                : phrase.id === 'second-apex' ? say.slice(0, 2) : say;
            const children = phrase.id === 'one-tight-slow-corner'
                ? [createInsideLongStraightClosure(phrase, [...say.slice(2), say[0]]), createInsideCornerClosure()]
                : phrase.id === 'second-apex' ? [createInsideLongStraightClosure(phrase, say.slice(2))] : [];
            return new Closure<PhraseContext>(
                phrase.name,
                phrase.description,
                describeCondition((context) => {
                    if (phrase.referenceCue && context.candidate !== phrase.id) return false;
                    if (!context.matched.get(phrase.id) || (context.candidate && context.candidate !== phrase.id)) return false;
                    context.candidate = phrase.id;
                    const memory = context.memory.get(phrase.id)!;
                    memory.since ??= context.now;
                    return context.emit && memory.armed && context.now - memory.since >= phrase.holdMs
                        && (memory.lastEmitted === undefined || context.now - memory.lastEmitted >= (phrase.referenceCue ? REFERENCE_CUE_COOLDOWN_MS : COOLDOWN_MS));
                }, (context) => snapshotConditions(context.entryConditions.get(phrase.id) ?? [])),
                [...children, ...parentSay, (phrase.referenceCue
                    ? Action.exitToParent : Action.exitToRoot)(describeCondition((_context, state) =>
                    (phrase.referenceCue === 'steerSoon' ? parentSay.slice(-1) : parentSay).some((action) => state.current === action),
                    phrase.referenceCue === 'steerSoon' ? 'The Turn phrase has run' : 'The phrase action has run'))],
                { id: phrase.id, inspect: (context) => ({
                    ...context.inspections.get(phrase.id),
                    fields: { 'Hold for': `${phrase.holdMs / 1000} s`, Cooldown: `${(phrase.referenceCue ? REFERENCE_CUE_COOLDOWN_MS : COOLDOWN_MS) / 1000} s`, 'Clear period': phrase.referenceCue === 'brake' ? 'Until brake zone exit' : `${REARM_MS / 1000} s` },
                }) },
            );
        });
    const movements = phrases.filter((phrase) => phrase.movementDirection);
    const group = (name: string, id: string, members: readonly PhraseDefinition[], children: (Closure<PhraseContext> | Action<PhraseContext>)[]) =>
        new Closure<PhraseContext>(name, 'Guide the driver toward the reference line on a mapped straight until braking is needed.',
            describeCondition((context) => context.emit && context.referenceMovementAllowed === true
                && members.some((phrase) => context.candidate === phrase.id),
                (context) => [...snapshotConditions(context.entryConditions.get(MOVEMENT_CLOSURE_ID) ?? []),
                    { description: 'A reference-line movement cue matches the current position and progress',
                        conditionFit: members.some((phrase) => context.matched.get(phrase.id)),
                        inputMissing: members.every((phrase) => context.entryConditions.get(phrase.id)?.some((condition) => condition.inputMissing)),
                    }]), children, { id });
    const movementRoot = group('Turn to reference line', MOVEMENT_CLOSURE_ID, movements, [
        Action.exitToRoot(describeCondition((context) => context.referenceMovementAllowed !== true,
            'Reference braking has started, a pedal correction is needed, the driver has left the mapped straight, or required inputs are unavailable')),
        ...(['left', 'right'] as const).map((direction) => {
            const members = movements.filter((phrase) => phrase.movementDirection === direction);
            return group(`Move to the ${direction}`, `reference-move-${direction}`, members, [
                ...closures.filter((closure) => members.some((phrase) => phrase.id === closure.metadata.id)),
                Action.exitToParent<PhraseContext>(),
            ]);
        }),
    ]);
    const corneringPhrases = phrases.filter((phrase) => phrase.referenceCue && !phrase.movementDirection);
    const zoneClosures = pedalZones.map(({ zone, id, name, cues }) => new Closure<PhraseContext>(name,
        zone === 'throttle'
            ? 'Give throttle corrections until the driver leaves the corner.'
            : 'Stay in the interpolated reference brake zone, giving brake corrections and preparing for turn-in until the zone ends.',
        describeCondition((context) => context.emit && context.referencePedalZone === zone,
            (context) => snapshotConditions(context.entryConditions.get(id) ?? [])), [
            (zone === 'brake' ? Action.exitToParent : Action.exitToRoot)(describeCondition((context: PhraseContext) =>
                zone === 'throttle' ? context.driverCornerIds?.length === 0
                    : context.referencePedalZone !== undefined && context.referencePedalZone !== zone,
                zone === 'throttle' ? 'The driver has left the corner'
                    : 'The driver has left the interpolated reference brake zone')),
            ...closures.filter((closure) => phrases.some((phrase) => phrase.id === closure.metadata.id
                && phrase.referenceCue && cues.includes(phrase.referenceCue))),
        ], { id }));
    const corneringRoot = new Closure<PhraseContext>('Cornering',
        'Follow the interpolated reference brake and throttle zones. Pedal phrases return to their zone; the throttle zone returns to root when the driver leaves the corner.',
        describeCondition((context) => context.emit && conditionsMatch(context.entryConditions.get(CORNERING_CLOSURE_ID) ?? []),
            (context) => snapshotConditions(context.entryConditions.get(CORNERING_CLOSURE_ID) ?? [])),
        [Action.exitToRoot<PhraseContext>(describeCondition((context) => context.referenceThrottleOutsideCorner === true,
            'Reference throttle is outside the mapped corner')),
        ...zoneClosures],
        { id: CORNERING_CLOSURE_ID });
    return new Closure('root',
        'Select a matching guide after its entry hold time. Reference-line cues take priority, with a 2 s cooldown; overtaking guides use 8 s. Brake speaks once per brake zone; other guides require a 0.5 s clear period before repeating.',
        describeCondition(() => true, 'Always eligible'), closures.flatMap((closure, index) =>
            phrases[index].movementDirection ? phrases[index] === movements[0] ? [movementRoot] : []
                : phrases[index].referenceCue ? phrases[index] === corneringPhrases[0] ? [corneringRoot] : [] : [closure]));
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
    private deceleration?: number;
    private referencePosition?: ReferencePositionSample;
    private previousReferencePosition?: ReferencePositionSample;
    private vision: PhraseVisionInput | null = null;
    private visionAfter = -Infinity;
    private opponentDistanceM?: number;
    private closingOnOpponent?: 0 | 1;
    private memory = new Map<string, PhraseMemory>();
    private events: PhraseEvent[] = [];
    private nextId = 0;
    private map: CircuitMapDto | null = null;
    private sourceMap: CircuitMapDto | null = null;
    private sourceReferenceLap: TopLapReferenceEntry | null = null;
    private originalReferenceLap: TopLapReferenceEntry | null = null;
    private referenceLap: TopLapReferenceEntry | null = null;
    private drivingAggression = REFERENCE_DRIVING_AGGRESSION;
    private sessionGeneration?: number;
    private streamGeneration?: number;
    private lastReferenceSpeech = -Infinity;
    private referenceMovement = new ReferenceMovement();
    private mapContext = createPhraseMapContext(null);
    private sectionId?: string;
    private game?: string;
    private opponentMotion = new PhraseOpponentMotion();
    private cornerSummaries: readonly DriverCornerSummary[] = [];
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

    receiveMap(map: CircuitMapDto | null, now: number, referenceLap: TopLapReferenceEntry | null = null): PhraseSnapshot {
        if (referenceLap !== this.sourceReferenceLap) {
            // Keep source identity only to validate Live Map's fresh location predictions.
            this.sourceReferenceLap = referenceLap;
            this.originalReferenceLap = copyPhraseReference(referenceLap);
            this.referenceLap = copyPhraseReference(this.originalReferenceLap, this.drivingAggression);
            this.phrases.filter((phrase) => phrase.referenceCue).forEach((phrase) => this.memory.delete(phrase.id));
            this.lastReferenceSpeech = -Infinity;
            this.referenceMovement.reset();
            this.referencePosition = this.previousReferencePosition = undefined;
        }
        if (map !== this.sourceMap) {
            this.sourceMap = map;
            this.map = copyPhraseMap(map);
            this.mapContext = createPhraseMapContext(this.map);
            this.memory.forEach((memory, id) => {
                memory.since = undefined;
                if (!this.phrases.some((phrase) => phrase.id === id && phrase.referenceCue === 'brake')) memory.armed = true;
            });
            this.sectionId = undefined;
            this.opponentMotion.reset();
            this.cornerSummaries = [];
        }
        return this.evaluate(now);
    }

    setDrivingAggression(value: number, now: number): PhraseSnapshot {
        const aggression = normalizeDrivingAggression(value);
        if (aggression !== this.drivingAggression) {
            this.drivingAggression = aggression;
            // Always rebuild from the untouched local baseline, never from an already adjusted speed profile.
            this.referenceLap = copyPhraseReference(this.originalReferenceLap, aggression);
            this.memory.clear();
            this.state = new State(this.root);
            this.events = [];
            this.lastReferenceSpeech = -Infinity;
            this.referenceMovement.reset();
            this.referencePosition = this.previousReferencePosition = undefined;
        }
        return this.evaluate(now);
    }

    reset(now: number) {
        this.sample = {};
        this.receivedAt = -Infinity;
        this.deceleration = undefined;
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
        this.cornerSummaries = [];
        this.lastReferenceSpeech = -Infinity;
        this.referenceMovement.reset();
        this.referencePosition = this.previousReferencePosition = undefined;
    }

    receiveTelemetry(event: LiveTelemetryEvent, now: number, cornerSummaries: readonly DriverCornerSummary[] = []): PhraseSnapshot {
        if (event.type !== 'frame') {
            this.reset(now);
        } else {
            const previousSpeed = nonnegative(this.sample.Physics_speed_kmh), speed = nonnegative(event.sample.Physics_speed_kmh);
            const elapsed = now - this.receivedAt;
            this.deceleration = event.telemetryStatus === 2 && elapsed > 0 && elapsed <= TELEMETRY_MAX_AGE_MS
                && previousSpeed !== undefined && speed !== undefined && this.game === event.update.game
                && this.sample.Static_track === event.sample.Static_track && this.sample.Static_car_model === event.sample.Static_car_model
                && this.sessionGeneration === event.sessionGeneration && this.streamGeneration === event.streamGeneration
                ? (previousSpeed - speed) / 3.6 / (elapsed / 1000) : undefined;
            if (this.game !== event.update.game || this.sample.Static_track !== event.sample.Static_track) this.opponentMotion.reset();
            this.game = event.update.game;
            this.sessionGeneration = event.sessionGeneration;
            this.streamGeneration = event.streamGeneration;
            // A gap cannot count toward the continuous hold, even without a UI timer.
            if (now - this.receivedAt > TELEMETRY_MAX_AGE_MS) {
                this.memory.forEach((memory) => { memory.since = undefined; });
            }
            // Paused, replay and off frames cannot generate live guidance.
            this.sample = event.telemetryStatus === 2 ? event.sample : {};
            this.receivedAt = event.telemetryStatus === 2 ? now : -Infinity;
            this.opponentMotion.update(this.sample, now, TELEMETRY_MAX_AGE_MS);
            this.cornerSummaries = cornerSummaries;
        }
        return this.evaluate(now, true);
    }

    receiveVision(vision: PhraseVisionInput | null, now: number): PhraseSnapshot {
        const sameCalibration = JSON.stringify(vision?.calibration) === JSON.stringify(this.vision?.calibration);
        if (!sameCalibration) {
            this.referenceMovement.reset();
            this.referencePosition = this.previousReferencePosition = undefined;
        }
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
        const paired = this.vision?.telemetry;
        const positionVision = telemetryReady && this.vision && this.vision.capturedAt >= this.visionAfter
            && paired?.telemetryStatus === 2 && paired.game === this.game
            && paired.sessionGeneration === this.sessionGeneration && paired.streamGeneration === this.streamGeneration
            && paired.currentTelemetry.Static_track === this.sample.Static_track
            && paired.currentTelemetry.Static_car_model === this.sample.Static_car_model
            && paired.receivedAt !== undefined && paired.receivedAt <= this.vision.capturedAt
            && this.vision.capturedAt - paired.receivedAt <= REFERENCE_POSITION_MAX_AGE_MS
            ? this.vision : null;
        const previousPosition = positionVision?.capturedAt === this.referencePosition?.capturedAt
            ? this.previousReferencePosition : this.referencePosition;
        const guidance = readReferenceGuidance(this.sample, positionVision, this.referenceLap, this.game, now,
            this.sourceReferenceLap, this.originalReferenceLap, telemetryReady ? this.deceleration : undefined, previousPosition);
        if (guidance.lateralOffset === undefined || !positionVision) {
            this.referencePosition = this.previousReferencePosition = undefined;
        } else if (positionVision.capturedAt !== this.referencePosition?.capturedAt) {
            this.previousReferencePosition = this.referencePosition;
            this.referencePosition = { capturedAt: positionVision.capturedAt, lateralOffset: guidance.lateralOffset };
        }
        const driverCornerIds = telemetryReady && sameMap
            ? this.mapContext.cornersAt(this.sample.Graphics_normalized_car_position) : undefined;
        const referenceThrottleOutsideCorner = driverCornerIds?.length === 0 && guidance.pedalZone === 'throttle';
        // Reference throttle often continues down the straight. Limit its cornering zone
        // to the actual mapped bounds; braking may still begin before corner entry.
        if (referenceThrottleOutsideCorner) guidance.pedalZone = 'none';
        const driverOnStraight = telemetryReady && sameMap
            ? this.mapContext.isStraightAt(this.sample.Graphics_normalized_car_position) : undefined;
        inputs.driverOnStraight = driverOnStraight === undefined ? undefined : driverOnStraight ? 1 : 0;
        inputs.referenceBraking = guidance.braking === undefined ? undefined : guidance.braking ? 1 : 0;
        inputs.referencePedalZone = guidance.pedalZone;
        inputs.referenceSteeringPhase = guidance.steeringPhase ?? (guidance.ready ? 'none' : undefined);
        inputs.referenceCornering = guidance.pedalZone === undefined ? undefined : guidance.pedalZone === 'none' ? 0 : 1;
        const corneringRoot = this.phraseClosures.get(CORNERING_CLOSURE_ID);
        const inCornering = Boolean(corneringRoot && this.state.path.includes(corneringRoot));
        const corneringAllowed = inCornering || Boolean(corneringRoot && inputs.referenceCornering === 1);
        const referenceMovementAllowed = !corneringAllowed && driverOnStraight === true && guidance.braking === false;
        const movementCue = this.referenceMovement.update(referenceMovementAllowed ? guidance.lateralOffset : undefined, positionVision?.capturedAt);
        if (movementCue) guidance[movementCue] = now + 1500;
        // Select only one short cue per speech window. Armed cues can interrupt a waiting overtaking closure.
        const referencePhrase = now - this.lastReferenceSpeech >= 1000 ? this.phrases.find((phrase) => {
            const memory = this.memory.get(phrase.id);
            const zone = pedalZones.find(({ cues }) => phrase.referenceCue && cues.includes(phrase.referenceCue));
            return phrase.referenceCue && guidance[phrase.referenceCue] !== undefined
                && (phrase.movementDirection ? !inCornering : corneringAllowed)
                && (!zone || guidance.pedalZone === zone.zone)
                && (!phrase.movementDirection || this.referenceMovement.canSpeak(now))
                && (!memory || memory.armed || (phrase.referenceCue !== 'brake'
                    && memory.clearSince !== undefined && now - memory.clearSince >= REARM_MS))
                && (memory?.lastEmitted === undefined || now - memory.lastEmitted >= REFERENCE_CUE_COOLDOWN_MS);
        }) : undefined;
        inputs.closingOnOpponent = visionReady ? this.closingOnOpponent : undefined;
        const opponent = telemetryReady && sameMap ? this.opponentMotion.at(now, TELEMETRY_MAX_AGE_MS) : undefined;
        const opponentPosition = telemetryReady && sameMap ? this.opponentMotion.positionAt(now, TELEMETRY_MAX_AGE_MS) : undefined;
        const approach = this.mapContext.approaching(opponentPosition);
        if (approach) {
            inputs.opponentCornerEtaS = opponent ? lapDistance(opponent.position, approach.cornerStartPosition) / opponent.rate : undefined;
            inputs.approachCornerSpeed = approach.cornerSpeed;
            inputs.approachCornerDirection = approach.cornerDirection;
            inputs.approachSequenceRemaining = approach.sequenceRemaining;
            inputs.approachLinkedOpposite = approach.linkedOpposite;
            inputs.approachCornerInSequence = approach.sequenceId === undefined ? 0 : 1;
            inputs.approachLongStraightAfterCorner = approach.cornerSpeed === 'slow' ? approach.longStraightAfterCorner : 0;
            const opponentId = this.opponentMotion.carIdAt(now, TELEMETRY_MAX_AGE_MS);
            const playerSummary = this.cornerSummaries.find((summary) => summary.carId === String(this.sample.Graphics_player_car_id)
                && summary.cornerId === approach.sectionId);
            const opponentSummary = this.cornerSummaries.find((summary) => summary.carId === opponentId
                && summary.cornerId === approach.sectionId);
            const playerSpeed = playerSummary?.averageExitSpeed;
            const opponentSpeed = opponentSummary?.averageExitSpeed;
            if (typeof playerSpeed === 'number' && Number.isFinite(playerSpeed) && playerSpeed > 0
                && typeof opponentSpeed === 'number' && Number.isFinite(opponentSpeed) && opponentSpeed > 0) {
                inputs.betterCornerExitSpeed = playerSpeed > opponentSpeed ? 1 : 0;
            }
        }
        const matched = new Map<string, boolean>();
        const actionMatched = new Map<string, boolean>();
        const evaluatedActionConditions = new Map<string, readonly PhraseConditionNode[]>();
        const evaluated = this.phrases.map((phrase) => {
            const memory = this.memory.get(phrase.id) ?? { armed: true };
            this.memory.set(phrase.id, memory);
            const conditions = phrase.conditions.map((condition) => condition.evaluate(phrase.referenceCue
                ? { ...inputs, referenceCue: guidance[phrase.referenceCue] !== undefined ? phrase.referenceCue : guidance.ready ? 'none' : undefined } : inputs));
            const actions = getPhraseActions(phrase).map((action) => {
                const conditions = action.conditions?.map((condition) => condition.evaluate(inputs));
                actionMatched.set(action.id, telemetryReady && (conditions === undefined || conditionsMatch(conditions)));
                evaluatedActionConditions.set(action.id, conditions ?? []);
                return { conditions };
            });
            const actionConditions = actions[0].conditions;
            const conditionFit = conditionsMatch(conditions);
            const missing = conditionFit ? [] : missingConditionInputs(conditions);
            const fits = telemetryReady && conditionFit && (!phrase.referenceCue || Boolean(phrase.movementDirection) || corneringAllowed);
            matched.set(phrase.id, fits);
            if (phrase.referenceCue === 'brake') {
                // Pedal changes and missing inputs cannot rearm Brake within the same zone.
                if (!fits) memory.since = undefined;
                if (guidance.pedalZone !== undefined && guidance.pedalZone !== 'brake') memory.armed = true;
            } else if (!fits) {
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
            now, emit, matched, actionMatched, memory: this.memory, referenceMovementAllowed, referencePedalZone: guidance.pedalZone,
            referenceSpeechReady: now - this.lastReferenceSpeech >= 1000,
            referenceThrottleOutsideCorner,
            candidate: now - this.lastReferenceSpeech < 1000 ? 'reference-speech-window'
                : referencePhrase?.id ?? (corneringAllowed ? CORNERING_CLOSURE_ID : undefined),
            driverLongStraightIds: telemetryReady && sameMap
                ? this.mapContext.longStraightsAt(this.sample.Graphics_normalized_car_position) : undefined,
            driverCornerIds,
            entryConditions: new Map<string, readonly PhraseConditionNode[]>([...evaluated.map(({ phrase, conditions }) => [phrase.id, conditions] as const),
                [MOVEMENT_CLOSURE_ID, movementConditions.map((condition) => condition.evaluate(inputs))],
                ...pedalZones.map(({ zone, id }) => [id, [condition('referencePedalZone', '=', zone).evaluate(inputs)]] as const),
                [CORNERING_CLOSURE_ID, corneringConditions.map((condition) => condition.evaluate(inputs))]]),
            actionConditions: evaluatedActionConditions,
            inspections,
            publish: (phrase, sentence = phrase.sentence) => {
                const expiresAt = phrase.referenceCue ? guidance[phrase.referenceCue] : undefined;
                this.events = [...this.events, { id: ++this.nextId, ruleId: phrase.id, sentence, timestamp: now,
                    ...(phrase.referenceCue === 'steerSoon' ? { steeringPhase: guidance.steeringPhase } : {}),
                    ...(expiresAt !== undefined ? { expiresAt } : {}) }].slice(-HISTORY_LIMIT);
                if (phrase.referenceCue) this.lastReferenceSpeech = now;
                if (phrase.movementDirection) this.referenceMovement.spoken(phrase.referenceCue as ReferenceMovementCue, now);
                const memory = this.memory.get(phrase.id)!;
                memory.lastEmitted = now;
                memory.armed = false;
            },
        };
        // Only State chooses a closure and invokes its action. A completed action is
        // retained until the next step. Pedal speech returns to its zone, never directly to root.
        const previousClosure = this.state.closure;
        const activeZone = pedalZones.find(({ id }) => this.state.path.includes(this.phraseClosures.get(id)!));
        if (activeZone && (activeZone.zone === 'throttle' ? driverCornerIds?.length === 0
            : guidance.pedalZone !== undefined && guidance.pedalZone !== activeZone.zone)) {
            const zoneRoot = this.phraseClosures.get(activeZone.id)!;
            // Check the enclosing zone exit even while a correction action is selected.
            while (this.state.closure !== zoneRoot) this.state.exitToParent();
            this.step(context);
        }
        const movementRoot = this.phraseClosures.get(MOVEMENT_CLOSURE_ID);
        const inMovement = movementRoot && this.state.path.includes(movementRoot);
        // Evaluate the enclosing exit immediately, even while a nested speech action is selected.
        // New movement cues also return here so either direction can be selected within the same scope.
        if (inMovement && (!referenceMovementAllowed || (emit && referencePhrase?.movementDirection))) {
            while (this.state.closure !== movementRoot) this.state.exitToParent();
        }
        if (inCornering && emit && this.phrases.some((phrase) =>
            phrase.referenceCue && phrase.id === this.state.closure.metadata.id)) {
            // Finish the previous speech action inside its existing zone before selecting another cue.
            this.step(context);
        }
        if (emit && (referencePhrase || corneringAllowed) && this.state.closure !== this.root && !inCornering
            && !(inMovement && (!referenceMovementAllowed || referencePhrase?.movementDirection))) {
            this.state = new State(this.root);
        }
        this.step(context);
        if (previousClosure !== this.root && this.state.current === this.root) this.step(context);
        const closures = evaluated.flatMap(({ phrase, memory, missing, conditions, actionConditions }) => {
            const closure = this.phraseClosures.get(phrase.id);
            if (!closure) return [];
            const selected = context.candidate === phrase.id || this.state.path.includes(closure);
            let status: ClosureStatus = missing.length || !telemetryReady ? 'Missing input' : 'Not matched';
            if (this.state.path.includes(closure) && (phrase.actionConditions || phrase.additionalActions?.length)) {
                status = this.state.current === closure ? 'Waiting for action' : 'Active';
            } else if (selected && matched.get(phrase.id)) {
                const cooldown = memory.lastEmitted !== undefined && now - memory.lastEmitted < (phrase.referenceCue ? REFERENCE_CUE_COOLDOWN_MS : COOLDOWN_MS);
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
            referenceLoaded: this.referenceLap !== null,
            referenceSteeringPhase: guidance.steeringPhase,
            root: this.state.snapshot(context),
            state: { current: this.state.current.name, description: this.state.current.description,
                kind: this.state.current instanceof Closure ? 'closure' : 'action', path: this.state.path.map((node) => node.name) },
        };
    }
}
