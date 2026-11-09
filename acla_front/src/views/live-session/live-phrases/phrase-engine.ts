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
import { BRAKE_RELEASE_RATE_THRESHOLD, PhraseLookahead } from './phrase-lookahead';
import { ReferenceMovement, REFERENCE_MOVEMENTS, type ReferenceMovementCue, type MovementDirection } from './phrase-reference-movement';
import { copyPhraseMap, copyPhraseReference, normalizeDrivingAggression, REFERENCE_DRIVING_AGGRESSION } from './phrase-driving-reference';
import type { ReferenceSteeringDirection } from './phrase-steering-direction';
import { Phrase, type PhraseCurves, type PhraseInstance, type PhraseSettings } from './phrase';
import brakeCurves from './curves/reference-brake.json';
import releaseBrakeCurves from './curves/slowly-release-brake.json';
import { BRAKE_RELEASE_SETTINGS, brakeReleaseSettings, SLOWLY_RELEASE_BRAKE } from './phrase-brake-release';

export const TELEMETRY_MAX_AGE_MS = 1500;
const DECELERATION_INTERVAL_MS = 500;
const STEERING_DEVIATION_HOLD_MS = 500;
export const REARM_MS = 500;
export const COOLDOWN_MS = 8000;
const HISTORY_LIMIT = 50;

const INPUT_LABELS = {
    drivingAggression: 'Driving aggression',
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
    referencePedalZone: 'Interpolated reference pedal zone',
    referenceSteeringPhase: 'Reference steering phase',
    referenceSteeringDirection: 'Required steering correction',
    referenceSteeringDeviationMs: 'Same direction deviation duration (ms)',
    referenceSteeringCorrectionSpoken: 'Direction correction already spoken during this deviation',
    referenceBrakeReleaseRate: 'Reference deceleration easing rate while braking (m/s³)',
} as const;
type Input = keyof typeof INPUT_LABELS;
type Inputs = PhraseMapContext & ReturnType<typeof getPhrasePositions> & {
    drivingAggression?: number;
    speed?: number; referenceCue?: ReferenceCue | 'none'; driverOnStraight?: 0 | 1; referenceBraking?: 0 | 1;
    referencePedalZone?: ReferencePedalZone;
    referenceBrakeReleaseRate?: number;
    referenceSteeringPhase?: ReferenceSteeringPhase | 'none';
    referenceSteeringDirection?: ReferenceSteeringDirection;
    referenceSteeringDeviationMs?: number;
    referenceSteeringCorrectionSpoken?: 0 | 1;
    insideLine?: 0 | 1; outsideLine?: 0 | 1; opponentDistanceM?: number;
    closingOnOpponent?: 0 | 1; directlyBehindOpponent?: 0 | 1;
    opponentCornerEtaS?: number; approachCornerSpeed?: PhraseMapContext['cornerSpeed'];
    approachCornerDirection?: PhraseMapContext['cornerDirection'];
    approachSequenceRemaining?: number; approachLinkedOpposite?: 0 | 1; approachCornerInSequence?: 0 | 1;
    betterCornerExitSpeed?: 0 | 1; approachLongStraightAfterCorner?: 0 | 1 };
type PhraseVisionInput = Pick<TrackVisionDetection, 'capturedAt' | 'calibration' | 'birdsEyeScene' | 'position' | 'telemetry'>;
export type PhraseConditionConnector = 'and' | 'or';
export class PhraseCondition {
    readonly conditionFit: boolean | null;
    readonly inputMissing: boolean;

    constructor(
        readonly input: Input,
        readonly operator: '>=' | '<=' | '<' | '=' | 'in',
        readonly value: number | string | readonly string[],
        actual?: Inputs[Input],
        // Joins this condition to the previous one; ignored for the first condition.
        readonly connector: PhraseConditionConnector = 'and',
        evaluated = true,
    ) {
        this.inputMissing = evaluated && actual === undefined;
        this.conditionFit = !evaluated ? null : actual !== undefined && (operator === '>=' || operator === '<=' || operator === '<'
            ? typeof actual === 'number' && typeof value === 'number' && (operator === '>=' ? actual >= value : operator === '<' ? actual < value : actual <= value)
            : operator === 'in' ? Array.isArray(value) && value.includes(actual) : actual === value);
    }

    get description(): string {
        if (this.input === 'referenceCue') {
            const descriptions: Partial<Record<ReferenceCue, string>> = {
                brake: 'Brake demand for the target speed is at least 10% and the brake is not yet applied',
                steering: 'The live phrase reference direction-change range has started toward an overlapping corner; the remaining trajectory is partially or fully inside the corner.',
                throttle: 'Throttle demand for the target speed is at least 15% and the throttle is not yet applied, near the line and clear of braking',
                moreBrake: 'Speed is at least 5 km/h above the local reference, or within the speed tolerance with deceleration below the required rate; the brake is applied below full pressure and is not being released',
                releaseBrake: 'The brake is applied and the speed profile needs no further deceleration; without reference speed, pedal demand is below 10%',
                releaseBrakeSlower: 'The brake pedal is being released while speed is at least 5 km/h above the local reference, or within the speed tolerance with deceleration below the required rate by more than 10% or 0.1 m/s², whichever is greater',
                releaseBrakeFaster: 'Speed is at least 5 km/h below the local reference with positive braking demand, or within the speed tolerance with deceleration above the required rate by more than 10% or 0.1 m/s², whichever is greater',
                holdBrake: 'Speed is within 5 km/h of the local reference and lookahead deceleration matches the required rate within 10% or 0.1 m/s², whichever is greater',
                moreThrottle: 'Applied throttle is at least 15% and at least 15 points below demand from reference throttle, acceleration and target speed, within 2 m of the line, with brake below 10% and speed less than 5 km/h above the reference',
                lessThrottle: 'Speed is at least 5 km/h above the local reference and the driver is at least 5 units outside the reference line and moving farther outward, with throttle applied',
            };
            const movement = REFERENCE_MOVEMENTS.find(({ cue }) => cue === this.value);
            return movement ? `Move to the ${movement.direction}: ${movement.description}`
                : descriptions[this.value as ReferenceCue] ?? INPUT_LABELS.referenceCue;
        }
        if (this.operator === 'in' || this.operator === '=') {
            if (this.input === 'referenceSteeringCorrectionSpoken' && this.value === 0) return 'No direction correction has been spoken during this deviation';
            if (this.input === 'referenceSteeringDirection') return `Predicted motion needs a correction to the ${this.value} along the adjusted reference curve`;
            if (this.input === 'referenceSteeringPhase') return this.value === 'prepare'
                ? 'The reference direction-change range has started toward an overlapping corner; reference steering has not reached the 0.08 turn-in threshold'
                : 'The reference direction-change range has started toward an overlapping corner; reference steering has reached the 0.08 turn-in threshold';
            if (this.input === 'driverOnStraight') return INPUT_LABELS.driverOnStraight;
            if (this.input === 'referenceBraking' && this.value === 0) return 'Reference is not braking and no brake correction is needed';
            if (this.input === 'referenceBraking' && this.value === 1) return INPUT_LABELS.referenceBraking;
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
    readonly conditionFit: boolean | null;
    readonly inputMissing: boolean;

    constructor(
        readonly conditions: readonly PhraseConditionNode[],
        readonly connector: PhraseConditionConnector = 'and',
    ) {
        this.conditionFit = conditions.some((condition) => condition.conditionFit === null) ? null : conditionsMatch(conditions);
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
            groupMatched = condition.conditionFit ?? false;
        } else {
            groupMatched = groupMatched && condition.conditionFit === true;
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
    amplifyCurves?: PhraseCurves;
    speechSettings?: PhraseSettings;
    /** All possible overrides must be included in preloadSettings; playback never synthesizes. */
    instanceSettings?: (context: PhraseContext) => PhraseSettings;
    preloadSettings?: readonly PhraseSettings[];
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
    return [phrase, ...(phrase.additionalActions ?? [])].map((action, index) => {
        const id = `${phrase.id}:say:${index}`;
        const sentences = [action.sentence, ...(action.sentenceAlternatives ?? [])];
        return {
            id, sentence: action.sentence, conditions: action.actionConditions, sentences,
            phrases: sentences.map((sentence) => new Phrase(id, sentence, action.amplifyCurves, action.speechSettings)),
            instanceSettings: action.instanceSettings,
            preloadSettings: action.preloadSettings,
        };
    });
}
const condition = (input: Input, operator: PhraseCondition['operator'], value: PhraseCondition['value'], connector: PhraseConditionConnector = 'and'): PhraseCondition => new PhraseCondition(input, operator, value, undefined, connector);
const movementConditions = [condition('driverOnStraight', '=', 1), condition('referenceBraking', '=', 0)];
const MOVEMENT_CLOSURE_ID = 'reference-turn-to-line';
const STEERING_CLOSURE_ID = 'reference-steering';
const BRAKE_ZONE_EXIT_DESCRIPTION = 'The live phrase reference direction-change range has started toward an overlapping corner';
const brakeCues: readonly ReferenceCue[] = ['brake', 'moreBrake', 'releaseBrake', 'releaseBrakeSlower', 'releaseBrakeFaster', 'holdBrake'];
const throttleCues: readonly ReferenceCue[] = ['throttle', 'moreThrottle', 'lessThrottle'];
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
        ['moreBrake', 'More brake', 'More brake.', 'Increase pressure when faster than the local adjusted reference, or near its speed but decelerating too little, without an active pedal release.'],
        ['releaseBrake', 'Release the brake', 'Release the brake.', 'Release the pedal when the adjusted speed target no longer requires braking.'],
        ['releaseBrakeSlower', 'Release brake slower', 'Release brake slower.', 'Slow the current pedal release when faster than the local adjusted reference, or near its speed but decelerating too little.'],
        ['releaseBrakeFaster', 'Release brake faster', 'Release brake faster.', 'Reduce pressure when already slower than the local adjusted reference, or near its speed but decelerating too much.'],
        ['holdBrake', 'Hold brake pressure', 'Hold brake pressure.', 'Keep brake pressure steady while both speed and deceleration match the local adjusted reference.'],
    ] as const).map(([referenceCue, name, sentence, description]) => ({
        id: `reference-${referenceCue}`, referenceCue, name, sentence, description,
        ...(referenceCue === 'brake' ? { amplifyCurves: brakeCurves } : {}),
        holdMs: 0,
        conditions: [condition('referencePedalZone', '=', 'brake'), condition('referenceCue', '=', referenceCue)],
    })),
    {
        id: STEERING_CLOSURE_ID, referenceCue: 'steering', name: 'Steering', holdMs: 0,
        description: 'Enter from root once the live phrase reference direction-change range starts toward an overlapping corner and immediately say Turn. After Turn, say Slowly release brake once when reference deceleration eases while braking; delivery follows its change rate. Say Stay right a bit or Stay left a bit once when the same direction deviation persists for 500 ms and Driving aggression is 100; realignment rearms these cues. Stay entered until the continuous reference trajectory direction-change range ends.',
        sentence: 'Turn',
        conditions: [condition('referenceCue', '=', 'steering')],
        additionalActions: [
            { sentence: 'Stay right a bit', actionConditions: [condition('drivingAggression', '=', 100), condition('referenceSteeringDirection', '=', 'right'),
                condition('referenceSteeringDeviationMs', '>=', STEERING_DEVIATION_HOLD_MS), condition('referenceSteeringCorrectionSpoken', '=', 0)] },
            { sentence: 'Stay left a bit', actionConditions: [condition('drivingAggression', '=', 100), condition('referenceSteeringDirection', '=', 'left'),
                condition('referenceSteeringDeviationMs', '>=', STEERING_DEVIATION_HOLD_MS), condition('referenceSteeringCorrectionSpoken', '=', 0)] },
            { sentence: SLOWLY_RELEASE_BRAKE, amplifyCurves: releaseBrakeCurves,
                actionConditions: [condition('referenceBrakeReleaseRate', '>=', 0.01)],
                instanceSettings: (context) => brakeReleaseSettings(context.guidance.brakeReleaseRate!),
                preloadSettings: BRAKE_RELEASE_SETTINGS },
        ],
    },
    ...REFERENCE_MOVEMENTS.map(({ cue, direction, name, sentence, sentenceAlternatives, description }) => ({
        id: `reference-${cue}`, referenceCue: cue, movementDirection: direction, name, sentence, sentenceAlternatives, description, holdMs: 0,
        conditions: [condition('speed', '>=', 20), ...movementConditions, condition('referenceCue', '=', cue)],
    })),
    ...([
        ['throttle', 'Throttle', 'Throttle.', 'Apply throttle using reference throttle and acceleration to follow the speed profile adjusted by Driving aggression.'],
        ['moreThrottle', 'More throttle', 'More throttle.', 'Increase throttle when reference acceleration response indicates more is needed for the adjusted speed target.'],
        ['lessThrottle', 'Less throttle', 'Less throttle.', 'Reduce throttle when faster than the local reference, at least 5 units outside its line and moving farther outward, using reference curvature to identify the outside of the corner.'],
    ] as const).map(([referenceCue, name, sentence, description]) => ({
        id: `reference-${referenceCue}`, referenceCue, name, sentence, description, holdMs: 0,
        conditions: [condition('speed', '>=', 20), condition('referencePedalZone', '=', 'throttle'), condition('referenceCue', '=', referenceCue)],
    })),
];

export type ClosureStatus = 'Not evaluated' | 'Missing input' | 'Not matched' | 'Confirming' | 'Waiting' | 'Active' | 'Cooldown';
export interface PhraseEvent {
    id: number; ruleId: string; sentence: string; timestamp: number; expiresAt?: number;
    /** Present on engine-produced events; optional for older snapshot consumers. */
    instance?: PhraseInstance;
    steeringPhase?: ReferenceSteeringPhase;
    steeringDirection?: ReferenceSteeringDirection;
    brakeRelease?: boolean;
}
export interface PhraseSnapshot {
    telemetryReady: boolean;
    visionReady: boolean;
    mapReady: boolean;
    referenceLoaded: boolean;
    referenceSteeringPhase?: ReferenceSteeringPhase;
    referenceSteeringDirection?: ReferenceSteeringDirection;
    referenceBrakeReleaseRate?: number;
    mapContext: PhraseMapContext;
    steeringCorner?: PhraseMapContext;
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
    telemetryReady: boolean;
    intendedTelemetry: StandardTelemetrySample;
    inputs: Inputs;
    guidance: ReturnType<typeof readReferenceGuidance>;
    movementCue: () => ReferenceMovementCue | undefined;
    movementCanSpeak: () => boolean;
    resetMovement: () => void;
    measureZone: (zone: 'brake' | 'throttle') => void;
    referenceMovementAllowed?: boolean;
    referencePedalZone?: ReferencePedalZone;
    driverLongStraightIds?: readonly string[];
    driverCornerIds?: readonly string[];
    memory: Map<string, PhraseMemory>;
    entryConditions: Map<string, readonly PhraseConditionNode[]>;
    actionConditions: Map<string, readonly PhraseConditionNode[]>;
    inspections: Map<string, NodeInspection>;
    candidate?: string;
    publish: (phrase: PhraseDefinition, instance: PhraseInstance) => void;
}

function unevaluatedConditions(conditions: readonly PhraseConditionNode[]): PhraseConditionNode[] {
    return conditions.map((condition) => condition instanceof PhraseConditionGroup
        ? new PhraseConditionGroup(unevaluatedConditions(condition.conditions), condition.connector)
        : new PhraseCondition(condition.input, condition.operator, condition.value, undefined, condition.connector, false));
}

function measureAction(context: PhraseContext, id: string, conditions?: readonly PhraseConditionNode[]): boolean {
    const evaluated = conditions?.map((condition) => condition.evaluate(context.inputs));
    context.actionConditions.set(id, evaluated ?? []);
    return context.telemetryReady && (evaluated === undefined || conditionsMatch(evaluated));
}

function steeringRangeEnded() {
    const enteredRanges = new WeakMap<State<PhraseContext>, NonNullable<PhraseContext['guidance']['steeringRange']>>();
    const ended = (context: PhraseContext, state: State<PhraseContext>) => {
        const current = context.guidance.steeringRange, entered = enteredRanges.get(state);
        if (current === undefined || !entered) return undefined;
        return current === null || current.start !== entered.start || current.end !== entered.end || current.turnSign !== entered.turnSign;
    };
    const description = 'The continuous reference trajectory direction-change range captured on Steering entry has ended';
    const condition = describeCondition<PhraseContext>((context, state) => {
        // The exit action is checked first on entry, before Turn or any correction.
        if (!enteredRanges.has(state) && context.guidance.steeringRange) enteredRanges.set(state, context.guidance.steeringRange);
        return ended(context, state) === true;
    }, (context, state) => [{ description, conditionFit: ended(context, state) ?? null,
        inputMissing: context.guidance.steeringRange === undefined }],
    [{ description, conditionFit: null, inputMissing: false }]);
    condition.reset = (state) => { enteredRanges.delete(state); };
    return condition;
}

/** An entered phrase owns the measurement and execution of its action children. */
class PhraseClosure extends Closure<PhraseContext> {
    evaluate(context: PhraseContext, state: State<PhraseContext>, onActionError?: (error: unknown) => void): unknown {
        if (!state.path.includes(this)) return;
        state.check(this, context);
        return super.evaluate(context, state, onActionError);
    }
}

class PedalZoneClosure extends Closure<PhraseContext> {
    evaluate(context: PhraseContext, state: State<PhraseContext>, onActionError?: (error: unknown) => void): unknown {
        if (!state.path.includes(this)) return;
        context.measureZone(this.metadata.id === 'reference-brake-zone' ? 'brake' : 'throttle');
        return super.evaluate(context, state, onActionError);
    }
}

function phraseEntry(phrase: PhraseDefinition) {
    const memories = new WeakMap<State<PhraseContext>, PhraseMemory>();
    const entry = describeCondition<PhraseContext>((context, state) => {
        const { now, inputs, guidance } = context;
        const memory = context.memory.get(phrase.id) ?? { armed: true };
        context.memory.set(phrase.id, memory);
        memories.set(state, memory);
        const cue = phrase.movementDirection ? context.movementCue() === phrase.referenceCue : phrase.referenceCue && guidance[phrase.referenceCue] !== undefined;
        const needsLateral = phrase.movementDirection || (phrase.referenceCue && throttleCues.includes(phrase.referenceCue));
        const referenceReady = guidance.ready && (!needsLateral || guidance.lateralOffset !== undefined);
        const conditions = phrase.conditions.map((condition) => condition.evaluate(phrase.referenceCue
            ? { ...inputs, referenceCue: cue ? phrase.referenceCue : referenceReady ? 'none' : undefined } : inputs));
        context.entryConditions.set(phrase.id, conditions);
        const fits = context.telemetryReady && conditionsMatch(conditions);
        const rearmAfterClear = phrase.referenceCue !== 'brake' && phrase.referenceCue !== 'steering';
        if (!fits) {
            memory.since = undefined;
            if (rearmAfterClear) {
                memory.clearSince ??= now;
                if (now - memory.clearSince >= REARM_MS) memory.armed = true;
            }
        } else {
            if (rearmAfterClear && memory.clearSince !== undefined && now - memory.clearSince >= REARM_MS) memory.armed = true;
            memory.clearSince = undefined;
        }
        const selected = !context.candidate || context.candidate === phrase.id || Boolean(phrase.referenceCue);
        if (fits && selected) {
            if (!phrase.referenceCue) context.candidate = phrase.id;
            memory.since ??= now;
        } else memory.since = undefined;
        const cooldown = phrase.referenceCue !== 'steering' && memory.lastEmitted !== undefined
            && now - memory.lastEmitted < (phrase.referenceCue ? REFERENCE_CUE_COOLDOWN_MS : COOLDOWN_MS);
        const status: ClosureStatus = !context.telemetryReady || (!fits && missingConditionInputs(conditions).length > 0) ? 'Missing input'
            : !fits ? 'Not matched' : !selected || !memory.armed ? 'Waiting'
                : memory.since === undefined || now - memory.since < phrase.holdMs ? 'Confirming' : cooldown ? 'Cooldown' : 'Confirming';
        context.inspections.set(phrase.id, { status });
        return fits && selected && context.emit && memory.armed && memory.since !== undefined
            && now - memory.since >= phrase.holdMs && !cooldown && (!phrase.movementDirection || context.movementCanSpeak());
    }, (context) => snapshotConditions(context.entryConditions.get(phrase.id) ?? []), snapshotConditions(unevaluatedConditions(phrase.conditions)));
    entry.reset = (state, parentExited) => {
        const memory = memories.get(state);
        if (!memory) return;
        memory.since = memory.clearSince = undefined;
        if (parentExited && phrase.referenceCue === 'brake') memory.armed = true;
    };
    return entry;
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
        }], [{ description: 'Driver is inside a long straight', conditionFit: null, inputMissing: false }]), [
            Action.exitToParent(describeCondition(exit, (context, state) => [{
                description: 'Driver is no longer inside that long straight', conditionFit: exit(context, state),
                inputMissing: context.driverLongStraightIds === undefined,
            }], [{ description: 'Driver is no longer inside that long straight', conditionFit: null, inputMissing: false }])),
            ...say.map((action) => new Action(action.name, action.description, action.run,
                describeCondition((context, state) => entry(context) && action.condition(context, state), action.condition.describe!, action.condition.unevaluated), action.metadata)),
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
        }], [{ description: 'Driver is inside a corner', conditionFit: null, inputMissing: false }]), [
            Action.exitToParent(describeCondition(exit, (context, state) => [{
                description: 'Driver is no longer inside that corner', conditionFit: exit(context, state),
                inputMissing: context.driverCornerIds === undefined,
            }], [{ description: 'Driver is no longer inside that corner', conditionFit: null, inputMissing: false }])),
        ], { id: 'one-tight-slow-corner:inside-corner' });
}

export function createPhraseRoot(phrases: readonly PhraseDefinition[] = PHRASE_DEFINITIONS): Closure<PhraseContext> {
    const closures = phrases.map((phrase) => {
            const say: Action<PhraseContext>[] = getPhraseActions(phrase).map((action) => new Action<PhraseContext>(
                'say phrase', action.sentence, (context) => {
                    const index = action.sentences.length > 1 ? Math.floor(Math.random() * action.sentences.length) : 0;
                    context.publish(phrase, action.phrases[index].createInstance(action.instanceSettings?.(context)));
                },
                describeCondition((context, state) => measureAction(context, action.id, action.conditions)
                    && (phrase.referenceCue === 'steering' && action.sentence === SLOWLY_RELEASE_BRAKE
                        ? state.getExecution(say[0]).status === 'completed' : state.current === state.closure) && context.emit,
                    action.conditions
                        ? (context) => snapshotConditions(context.actionConditions.get(action.id) ?? [])
                        : 'Always eligible', action.conditions ? snapshotConditions(unevaluatedConditions(action.conditions)) : undefined),
                { id: action.id, ...(action.sentence === SLOWLY_RELEASE_BRAKE ? { priority: 1 } : {}), ...(action.sentences.length > 1 ? { fields: {
                    'Phrase selection': 'Random (equal chance)',
                    ...Object.fromEntries(action.sentences.slice(1).map((sentence, index) => [`Alternative ${index + 1}`, sentence])),
                } } : {}) }));
            const parentSay = phrase.referenceCue === 'steering' ? [say[0], ...say.slice(3)]
                : phrase.id === 'one-tight-slow-corner' ? say.slice(1, 2)
                : phrase.id === 'second-apex' ? say.slice(0, 2) : say;
            const children = phrase.referenceCue === 'steering' ? say.slice(1, 3).map((action, index) =>
                new Closure<PhraseContext>(action.description,
                    'After Turn, require the same direction deviation for 500 ms; speak once until the car realigns.',
                    describeCondition((context, state) => state.getExecution(say[0]).status === 'completed'
                        && measureAction(context, `${phrase.id}:say:${index + 1}`, getPhraseActions(phrase)[index + 1].conditions) && context.emit,
                    (context, state) => [
                        { description: 'Turn has been spoken', conditionFit: state.getExecution(say[0]).status === 'completed', inputMissing: false },
                        ...snapshotConditions(context.actionConditions.get(`${phrase.id}:say:${index + 1}`)
                            ?? unevaluatedConditions(getPhraseActions(phrase)[index + 1].conditions ?? [])),
                    ], [
                        { description: 'Turn has been spoken', conditionFit: null, inputMissing: false },
                        ...snapshotConditions(unevaluatedConditions(getPhraseActions(phrase)[index + 1].conditions ?? [])),
                    ]), [action, Action.exitToParent(describeCondition((_context, state) => state.current === action,
                        'The direction correction has been spoken'))], { id: `${phrase.id}:direction:${index}` }))
                : phrase.id === 'one-tight-slow-corner'
                ? [createInsideLongStraightClosure(phrase, [...say.slice(2), say[0]]), createInsideCornerClosure()]
                : phrase.id === 'second-apex' ? [createInsideLongStraightClosure(phrase, say.slice(2))] : [];
            return new PhraseClosure(
                phrase.name,
                phrase.description,
                phraseEntry(phrase),
                [...(phrase.referenceCue === 'steering' ? [new Action<PhraseContext>('exit to root',
                    'Return to the root closure so another guide can be selected.', (context, state) => {
                        const memory = context.memory.get(phrase.id);
                        // The captured range has ended; the next turn needs no timed clear period.
                        if (memory) memory.armed = true;
                        state.exitToRoot();
                    }, steeringRangeEnded(), { interrupt: true, priority: 1 })] : []), ...children, ...parentSay, ...(phrase.referenceCue === 'steering' ? [] : [(phrase.referenceCue
                    ? Action.exitToParent : Action.exitToRoot)(describeCondition<PhraseContext>((_context, state) =>
                    parentSay.some((action) => state.current === action),
                    'The phrase action has run'))])],
                { id: phrase.id, inspect: (context) => context.inspections.get(phrase.id) ?? {},
                    fields: { 'Hold for': `${phrase.holdMs / 1000} s`,
                        ...(phrase.referenceCue === 'steering' ? {} : { Cooldown: `${(phrase.referenceCue ? REFERENCE_CUE_COOLDOWN_MS : COOLDOWN_MS) / 1000} s` }),
                        'Clear period': phrase.referenceCue === 'brake' ? 'Until brake zone exit'
                            : phrase.referenceCue === 'steering' ? 'Until trajectory range exit' : `${REARM_MS / 1000} s` },
                    ...(phrase.referenceCue === 'steering' ? { priority: 3, interrupt: true } : {}),
                },
            );
        });
    const movements = phrases.filter((phrase) => phrase.movementDirection);
    const group = (name: string, id: string, members: readonly PhraseDefinition[], children: (Closure<PhraseContext> | Action<PhraseContext>)[]) => {
        const visits = new WeakMap<State<PhraseContext>, PhraseContext>();
        const entry = describeCondition<PhraseContext>((context, state) => {
                visits.set(state, context);
                const conditions = movementConditions.map((condition) => condition.evaluate(context.inputs));
                context.entryConditions.set(id, conditions);
                return context.emit && context.referenceMovementAllowed === true
                    && (id === MOVEMENT_CLOSURE_ID
                        ? state.path.some((node) => node.metadata.id === id) || Math.abs(context.guidance.lateralOffset ?? 0) >= 3
                        : members.some((phrase) => context.movementCue() === phrase.referenceCue));
            }, (context) => snapshotConditions(context.entryConditions.get(id) ?? []), snapshotConditions(unevaluatedConditions(movementConditions)));
        if (id === MOVEMENT_CLOSURE_ID) entry.reset = (state) => {
            const context = visits.get(state);
            if (!context) return;
            context.resetMovement();
            // Spoken flags belong to this visit, not to exits from its movement children.
            members.forEach((phrase) => {
                const memory = context.memory.get(phrase.id);
                if (memory) memory.armed = true;
            });
        };
        return new Closure<PhraseContext>(name, 'Guide the driver toward the reference line on a mapped straight until braking is needed.',
            entry, children, { id, ...(id === MOVEMENT_CLOSURE_ID ? { priority: 1, interrupt: true } : {}) });
    };
    const movementRoot = group('Turn to reference line', MOVEMENT_CLOSURE_ID, movements, [
        Action.exitToRoot(describeCondition((context) => context.referenceMovementAllowed !== true,
            'Reference braking has started, a pedal correction is needed, the driver has left the mapped straight, or required inputs are unavailable'), { interrupt: true, priority: 1 }),
        ...(['left', 'right'] as const).map((direction) => {
            const members = movements.filter((phrase) => phrase.movementDirection === direction);
            return group(`Move to the ${direction}`, `reference-move-${direction}`, members, [
                ...closures.filter((closure) => members.some((phrase) => phrase.id === closure.metadata.id)),
                Action.exitToParent<PhraseContext>(),
            ]);
        }),
    ]);
    const pedalPhrases = phrases.filter((phrase) => phrase.referenceCue && phrase.referenceCue !== 'steering' && !phrase.movementDirection);
    const zoneClosures = pedalZones.map(({ zone, id, name, cues }) => new PedalZoneClosure(name,
        zone === 'throttle'
            ? 'Give throttle corrections only when reference steering stays constant from the current position until the reference reaches full throttle, releases to zero, or starts braking.'
            : 'Give brake corrections from reference brake onset until reference brake releases to zero, reference throttle starts rising, or a live phrase reference direction-change range starts toward the corner.',
        describeCondition((context, state) => context.emit && context.referencePedalZone === zone
            && !(zone === 'brake' && context.guidance.brakeZoneExit === true)
            && !state.path.some((node) => node.metadata.id === STEERING_CLOSURE_ID),
            (context, state) => [...snapshotConditions([condition('referencePedalZone', '=', zone).evaluate(context.inputs)]),
                ...(zone === 'brake' ? [{ description: 'The direction-change condition for brake-zone exit is not met',
                    conditionFit: context.guidance.brakeZoneExit !== true, inputMissing: false }] : []),
                { description: 'Steering is not active', conditionFit: !state.path.some((node) => node.metadata.id === STEERING_CLOSURE_ID), inputMissing: false }],
            [...snapshotConditions(unevaluatedConditions([condition('referencePedalZone', '=', zone)])),
                ...(zone === 'brake' ? [{ description: 'The direction-change condition for brake-zone exit is not met',
                    conditionFit: null, inputMissing: false }] : []),
                { description: 'Steering is not active', conditionFit: null, inputMissing: false }]), [
            (zone === 'brake' ? Action.exitToParent : Action.exitToRoot)(describeCondition((context: PhraseContext) =>
                (context.referencePedalZone !== undefined && context.referencePedalZone !== zone)
                    || (zone === 'brake' && context.guidance.brakeZoneExit === true),
                (context) => [{ description: `The driver has left the interpolated reference ${zone} zone`,
                    conditionFit: context.referencePedalZone === undefined ? null : context.referencePedalZone !== zone,
                    inputMissing: context.referencePedalZone === undefined },
                ...(zone === 'brake' ? [{ description: BRAKE_ZONE_EXIT_DESCRIPTION, connector: 'or' as const,
                    conditionFit: context.guidance.brakeZoneExit ?? null, inputMissing: context.guidance.brakeZoneExit === undefined }] : [])],
                [{ description: `The driver has left the interpolated reference ${zone} zone`, conditionFit: null, inputMissing: false },
                    ...(zone === 'brake' ? [{ description: BRAKE_ZONE_EXIT_DESCRIPTION, connector: 'or' as const,
                        conditionFit: null, inputMissing: false }] : [])]), { interrupt: true, priority: 1 }),
            ...closures.filter((closure) => phrases.some((phrase) => phrase.id === closure.metadata.id
                && phrase.referenceCue && cues.includes(phrase.referenceCue))),
        ], { id, priority: zone === 'brake' ? 4 : 2, interrupt: true, yieldToSiblings: zone === 'brake' }));
    return new Closure('root',
        'Select a matching guide after its entry hold time. Reference-line cues take priority; Steering entry has no cooldown, other reference guides use 2 s, and overtaking guides use 8 s. Brake speaks once per brake zone; other guides require a 0.5 s clear period before repeating.',
        describeCondition(() => true, 'Always eligible'), closures.flatMap((closure, index) =>
            phrases[index].movementDirection ? phrases[index] === movements[0] ? [movementRoot] : []
                : phrases[index].referenceCue && phrases[index].referenceCue !== 'steering'
                    ? phrases[index] === pedalPhrases[0] ? zoneClosures : [] : [closure]));
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
    private lookahead = new PhraseLookahead();
    private receivedAt = -Infinity;
    private deceleration?: number;
    private brakeReleasing = false;
    private decelerationSample?: { speed: number; brake?: number; receivedAt: number };
    private referencePosition?: ReferencePositionSample;
    private previousReferencePosition?: ReferencePositionSample;
    private steeringDeviation?: { direction: ReferenceSteeringDirection; since: number };
    private steeringCorrectionSpoken = false;
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
    // Used only to rebuild the active lap when caution changes, never to evaluate guidance.
    private referenceBaseline: TopLapReferenceEntry | null = null;
    private referenceLap: TopLapReferenceEntry | null = null;
    private drivingAggression = REFERENCE_DRIVING_AGGRESSION;
    private sessionGeneration?: number;
    private streamGeneration?: number;
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
            this.steeringDeviation = undefined;
            this.steeringCorrectionSpoken = false;
            // Keep source identity only to validate Live Map's fresh lateral-position predictions.
            this.sourceReferenceLap = referenceLap;
            this.referenceBaseline = copyPhraseReference(referenceLap);
            this.referenceLap = copyPhraseReference(this.referenceBaseline, this.drivingAggression);
            this.phrases.filter((phrase) => phrase.referenceCue).forEach((phrase) => this.memory.delete(phrase.id));
            this.referenceMovement.reset();
            this.referencePosition = this.previousReferencePosition = undefined;
        }
        if (map !== this.sourceMap) {
            this.steeringDeviation = undefined;
            this.steeringCorrectionSpoken = false;
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
            this.steeringDeviation = undefined;
            this.steeringCorrectionSpoken = false;
            this.drivingAggression = aggression;
            // Always rebuild from the untouched local baseline, never from an already adjusted speed profile.
            this.referenceLap = copyPhraseReference(this.referenceBaseline, aggression);
            this.memory.clear();
            this.state = new State(this.root);
            this.events = [];
            this.referenceMovement.reset();
            this.referencePosition = this.previousReferencePosition = undefined;
        }
        return this.evaluate(now);
    }

    reset(now: number) {
        this.steeringDeviation = undefined;
        this.steeringCorrectionSpoken = false;
        this.sample = {};
        this.lookahead.reset();
        this.receivedAt = -Infinity;
        this.deceleration = undefined;
        this.brakeReleasing = false;
        this.decelerationSample = undefined;
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
        this.referenceMovement.reset();
        this.referencePosition = this.previousReferencePosition = undefined;
    }

    receiveTelemetry(event: LiveTelemetryEvent, now: number, cornerSummaries: readonly DriverCornerSummary[] = []): PhraseSnapshot {
        if (event.type !== 'frame') {
            this.reset(now);
        } else {
            const previousSpeed = nonnegative(this.sample.Physics_speed_kmh), speed = nonnegative(event.sample.Physics_speed_kmh);
            const elapsed = now - this.receivedAt;
            const continuous = event.telemetryStatus === 2 && elapsed > 0 && elapsed <= TELEMETRY_MAX_AGE_MS
                && previousSpeed !== undefined && speed !== undefined && this.game === event.update.game
                && this.sample.Static_track === event.sample.Static_track && this.sample.Static_car_model === event.sample.Static_car_model
                && this.sessionGeneration === event.sessionGeneration && this.streamGeneration === event.streamGeneration;
            if (!continuous) {
                this.steeringDeviation = undefined;
                this.deceleration = undefined;
                this.brakeReleasing = false;
                this.decelerationSample = undefined;
            }
            const brake = event.sample.Physics_brake;
            if (brake === undefined || !Number.isFinite(brake) || brake < 0 || brake > 1) {
                this.brakeReleasing = false;
                if (this.decelerationSample) this.decelerationSample.brake = undefined;
            }
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
            this.lookahead.update(this.sample, now, continuous, this.referenceLap);
            this.receivedAt = event.telemetryStatus === 2 ? now : -Infinity;
            this.opponentMotion.update(this.sample, now, TELEMETRY_MAX_AGE_MS);
            this.cornerSummaries = cornerSummaries;
        }
        return this.evaluate(now, true);
    }

    receiveVision(vision: PhraseVisionInput | null, now: number): PhraseSnapshot {
        const sameCalibration = JSON.stringify(vision?.calibration) === JSON.stringify(this.vision?.calibration);
        if (!sameCalibration || !this.vision || now - this.vision.capturedAt > REFERENCE_POSITION_MAX_AGE_MS) {
            this.steeringDeviation = undefined;
        }
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
            this.memory.forEach((memory, id) => {
                if (!this.phrases.some((phrase) => phrase.id === id && phrase.referenceCue)) memory.since = undefined;
            });
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
        inputs.drivingAggression = this.drivingAggression;
        const paired = this.vision?.telemetry;
        const positionVision = telemetryReady && this.vision && this.vision.capturedAt >= this.visionAfter
            && paired?.telemetryStatus === 2 && paired.game === this.game
            && paired.sessionGeneration === this.sessionGeneration && paired.streamGeneration === this.streamGeneration
            && paired.currentTelemetry.Static_track === this.sample.Static_track
            && paired.currentTelemetry.Static_car_model === this.sample.Static_car_model
            && paired.receivedAt !== undefined && paired.receivedAt <= this.vision.capturedAt
            && this.vision.capturedAt - paired.receivedAt <= REFERENCE_POSITION_MAX_AGE_MS
            ? this.vision : null;
        const activeZone = pedalZones.find(({ id }) => this.state.path.includes(this.phraseClosures.get(id)!));
        if (activeZone?.zone !== 'brake') {
            this.deceleration = undefined;
            this.brakeReleasing = false;
            this.decelerationSample = undefined;
        }
        if (activeZone?.zone !== 'throttle') this.referencePosition = this.previousReferencePosition = undefined;
        const previousPosition = positionVision?.capturedAt === this.referencePosition?.capturedAt
            ? this.previousReferencePosition : this.referencePosition;
        // Anticipate reference conditions in the input layer, before normal parent/child evaluation.
        // Freshness, capture validation and closure clocks still use measured telemetry and real time.
        const intendedTelemetry = telemetryReady ? this.lookahead.project(this.sample, this.referenceLap) : {};
        const intendedDeceleration = telemetryReady ? this.lookahead.projectDeceleration(this.sample) : undefined;
        const intendedBrakeReleasing = telemetryReady ? this.lookahead.projectBrakeReleasing(this.sample) : undefined;
        const referencePosition = intendedTelemetry.Graphics_normalized_car_position;
        const referenceMapContext = telemetryReady && sameMap ? this.mapContext.at(referencePosition) : {};
        const steeringCorner = telemetryReady && sameMap
            ? referenceMapContext.cornerGeometry ? referenceMapContext : this.mapContext.approaching(referencePosition)
            : undefined;
        const steeringBounds = steeringCorner?.cornerStartPosition !== undefined && steeringCorner.cornerEndPosition !== undefined
            ? { start: steeringCorner.cornerStartPosition, end: steeringCorner.cornerEndPosition } : undefined;
        const readGuidance = () => readReferenceGuidance(telemetryReady ? this.sample : {}, positionVision, this.referenceLap, this.game, now,
            this.sourceReferenceLap, intendedDeceleration ?? (telemetryReady ? this.deceleration : undefined), previousPosition,
            steeringCorner?.cornerGeometry?.turnSign, steeringBounds, intendedBrakeReleasing ?? this.brakeReleasing,
            referencePosition, intendedTelemetry);
        let guidance = readGuidance();
        const driverCornerIds = telemetryReady && sameMap
            ? this.mapContext.cornersAt(this.sample.Graphics_normalized_car_position) : undefined;
        const driverOnStraight = telemetryReady && sameMap
            ? this.mapContext.isStraightAt(referencePosition) : undefined;
        inputs.driverOnStraight = driverOnStraight === undefined ? undefined : driverOnStraight ? 1 : 0;
        inputs.referenceBraking = guidance.braking === undefined ? undefined : guidance.braking ? 1 : 0;
        inputs.referencePedalZone = guidance.pedalZone;
        inputs.referenceSteeringPhase = guidance.steeringPhase ?? (guidance.ready ? 'none' : undefined);
        inputs.referenceSteeringDirection = guidance.steeringDirection;
        const direction = guidance.steeringDirection;
        // Missing samples reset the hold but do not prove that the car has realigned.
        if (direction === 'aligned') this.steeringCorrectionSpoken = false;
        inputs.referenceSteeringCorrectionSpoken = this.steeringCorrectionSpoken ? 1 : 0;
        if (this.drivingAggression === 100 && direction !== undefined && direction !== 'aligned'
            && this.state.path.some((node) => node.metadata.id === STEERING_CLOSURE_ID)) {
            if (this.steeringDeviation?.direction !== direction || now < this.steeringDeviation.since) {
                this.steeringDeviation = { direction, since: now };
            }
            inputs.referenceSteeringDeviationMs = now - this.steeringDeviation.since;
        } else this.steeringDeviation = undefined;
        inputs.referenceBrakeReleaseRate = guidance.brakeReleaseRate;
        const referenceZone = pedalZones.find(({ zone, id }) => guidance.pedalZone === zone && this.phraseClosures.has(id));
        const pedalGuidanceAllowed = Boolean(activeZone || referenceZone);
        const referenceMovementAllowed = !pedalGuidanceAllowed && driverOnStraight === true && guidance.braking === false
            && guidance.lateralOffset !== undefined;
        if (!referenceMovementAllowed) this.referenceMovement.reset();
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
        const context: PhraseContext = {
            now, emit, telemetryReady, intendedTelemetry, inputs, guidance, memory: this.memory,
            referenceMovementAllowed, referencePedalZone: guidance.pedalZone,
            movementCue: () => {
                const cue = this.referenceMovement.update(referenceMovementAllowed ? guidance.lateralOffset : undefined, positionVision?.capturedAt);
                if (cue) guidance[cue] = now + 1500;
                return cue;
            },
            movementCanSpeak: () => this.referenceMovement.canSpeak(now),
            resetMovement: () => this.referenceMovement.reset(),
            measureZone: (zone) => {
                if (zone === 'brake') {
                    const speed = telemetryReady ? nonnegative(this.sample.Physics_speed_kmh) : undefined;
                    if (speed === undefined) {
                        this.deceleration = undefined;
                        this.brakeReleasing = false;
                        this.decelerationSample = undefined;
                    } else {
                        const rawBrake = this.sample.Physics_brake;
                        const brake = rawBrake !== undefined && Number.isFinite(rawBrake) && rawBrake >= 0 && rawBrake <= 1 ? rawBrake : undefined;
                        const baseline = this.decelerationSample;
                        if (!baseline) this.decelerationSample = { speed, brake, receivedAt: this.receivedAt };
                        else if (this.receivedAt - baseline.receivedAt >= DECELERATION_INTERVAL_MS) {
                            const seconds = (this.receivedAt - baseline.receivedAt) / 1000;
                            this.deceleration = (baseline.speed - speed) / 3.6 / seconds;
                            this.brakeReleasing = baseline.brake !== undefined && brake !== undefined
                                && (baseline.brake - brake) / seconds > BRAKE_RELEASE_RATE_THRESHOLD;
                            this.decelerationSample = { speed, brake, receivedAt: this.receivedAt };
                            guidance = readGuidance();
                            context.guidance = guidance;
                        }
                    }
                } else if (guidance.lateralOffset === undefined || !positionVision) {
                    this.referencePosition = this.previousReferencePosition = undefined;
                } else if (positionVision.capturedAt !== this.referencePosition?.capturedAt) {
                    this.previousReferencePosition = this.referencePosition;
                    this.referencePosition = { capturedAt: positionVision.capturedAt, lateralOffset: guidance.lateralOffset };
                }
            },
            driverLongStraightIds: telemetryReady && sameMap
                ? this.mapContext.longStraightsAt(this.sample.Graphics_normalized_car_position) : undefined,
            driverCornerIds,
            entryConditions: new Map(), actionConditions: new Map(), inspections: new Map(),
            publish: (phrase, instance) => {
                const sentence = instance.speech.text;
                const brakeRelease = phrase.referenceCue === 'steering' && sentence === SLOWLY_RELEASE_BRAKE;
                const steeringCorrection = phrase.referenceCue === 'steering' && sentence !== phrase.sentence && !brakeRelease;
                if (steeringCorrection) this.steeringCorrectionSpoken = true;
                const expiresAt = brakeRelease ? now + 4000 : steeringCorrection ? now + 1500 : phrase.referenceCue ? guidance[phrase.referenceCue] : undefined;
                this.events = [...this.events, { id: ++this.nextId, ruleId: phrase.id, sentence, instance, timestamp: now,
                    ...(steeringCorrection ? { steeringDirection: guidance.steeringDirection }
                        : phrase.referenceCue === 'steering' ? { steeringPhase: guidance.steeringPhase } : {}),
                    ...(brakeRelease ? { brakeRelease: true } : {}),
                    ...(expiresAt !== undefined ? { expiresAt } : {}) }].slice(-HISTORY_LIMIT);
                if (phrase.movementDirection) this.referenceMovement.spoken(phrase.referenceCue as ReferenceMovementCue, now);
                const memory = this.memory.get(phrase.id)!;
                memory.lastEmitted = now;
                memory.armed = false;
            },
        };
        // Evaluation descends only through entered parents. Exit actions may return
        // through several levels before the next eligible phrase is selected.
        for (let transitions = 0; transitions <= this.phraseClosures.size + 1; transitions++) {
            const previous = this.state.current;
            const previousClosure = this.state.closure;
            const eventCount = this.nextId;
            this.step(context);
            if (this.nextId !== eventCount || this.state.current === previous || this.state.current instanceof Action) break;
            if (previousClosure === this.root || previousClosure.metadata.id === STEERING_CLOSURE_ID
                || (this.state.current !== this.root && !previousClosure.metadata.id?.startsWith('reference-'))) break;
        }
        if (!this.state.path.some((node) => node.metadata.id === STEERING_CLOSURE_ID)) {
            this.steeringDeviation = undefined;
            this.steeringCorrectionSpoken = false;
        }
        else if (!this.steeringDeviation && this.drivingAggression === 100 && direction !== undefined && direction !== 'aligned') {
            // Count the first valid deviation on the same frame that enters Steering and says Turn.
            this.steeringDeviation = { direction, since: now };
        }
        const root = this.state.snapshot(context);
        const nodes = new Map<string, NodeSnapshot>();
        const collect = (node: NodeSnapshot) => {
            nodes.set(node.id, node);
            node.children.forEach(collect);
        };
        collect(root);
        const closures = this.phrases.flatMap((phrase) => {
            const node = nodes.get(phrase.id);
            if (!node) return [];
            const conditions = node.conditions.some((condition) => condition.conditionFit !== null)
                ? context.entryConditions.get(phrase.id) ?? unevaluatedConditions(phrase.conditions) : unevaluatedConditions(phrase.conditions);
            const actionConditions = context.actionConditions.get(`${phrase.id}:say:0`);
            const missing = conditionsMatch(conditions) ? [] : missingConditionInputs(conditions);
            const status = node.status as ClosureStatus;
            return [{
                id: phrase.id, name: node.name, description: node.description, holdMs: phrase.holdMs,
                status, missing, conditions, actionConditions,
                actions: node.children.filter((child) => child.kind === 'action').map(({ name, description }) => ({ name, description })),
            }];
        });
        return {
            telemetryReady, visionReady, mapReady: sameMap && this.mapContext.ready, mapContext, closures, events: this.events,
            referenceLoaded: this.referenceLap !== null,
            referenceSteeringPhase: guidance.steeringPhase,
            referenceSteeringDirection: guidance.steeringDirection,
            referenceBrakeReleaseRate: guidance.brakeReleaseRate,
            steeringCorner: guidance.ready ? steeringCorner : undefined,
            root,
            state: { current: this.state.current.name, description: this.state.current.description,
                kind: this.state.current instanceof Closure ? 'closure' : 'action', path: this.state.path.map((node) => node.name) },
        };
    }
}
