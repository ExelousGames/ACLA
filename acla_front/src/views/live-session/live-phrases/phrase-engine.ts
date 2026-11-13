import type { StandardTelemetrySample } from '../live-session-types';
import type { LiveTelemetryEvent } from '../live-telemetry-store';
import type { TrackVisionDetection } from '../track-vision/track-vision-types';
import type { CircuitMapDto } from 'views/circuit-maps/circuit-map-types';
import { getAccTelemetryTrackKey } from 'views/session-shared/visualization/charts/circuitTrackLayout';
import { createPhraseMapContext, PhraseMapContext } from './phrase-map-context';
import { createConsecutiveCornerPositioning, type ConsecutiveCornerSetup } from './phrase-consecutive-corners';
import { VISION_MAX_AGE_MS } from '../track-vision/track-vision-types';
import { getPhrasePositions } from './phrase-positions';
import { Action, Closure, State, describeCondition, type ConditionSnapshot, type NodeInspection, type NodeSnapshot } from './closure';
import { PhraseOpponentMotion } from './phrase-opponent-motion';
import { lapDistance } from './phrase-corner-geometry';
import type { DriverCornerSummary } from '../event-log/corner-summary';
import type { TopLapReferenceEntry } from 'services/topLapReferenceService';
import { readReferenceGuidance, referenceBrakingDeceleration, referenceCarPerformance, REFERENCE_CUE_COOLDOWN_MS, REFERENCE_POSITION_MAX_AGE_MS, REFERENCE_LOOKAHEAD_MS, type ReferenceCue, type ReferencePedalZone, type ReferenceSteeringPhase } from './phrase-reference-guidance';
import type { ReferenceCarPerformance } from './phrase-car-performance';
import { BRAKE_RELEASE_RATE_THRESHOLD, PhraseLookahead, type PhrasePredictionFrame } from './phrase-lookahead';
import { ReferenceMovement, REFERENCE_MOVEMENTS, type ReferenceMovementCue, type MovementDirection } from './phrase-reference-movement';
import { copyPhraseMap, copyPhraseReference, normalizeDrivingAggression, REFERENCE_DRIVING_AGGRESSION } from './phrase-driving-reference';
import type { ReferenceSteeringDirection } from './phrase-steering-direction';
import { Phrase, PHRASE_REACTION_ALLOWANCE_MS, type PhraseAudioVariant, type PhraseCurves, type PhraseInstance, type PhraseSettings, type PhraseSpeechTiming } from './phrase';
import brakeCurves from './curves/reference-brake.json';
import releaseBrakeCurves from './curves/slowly-release-brake.json';
import { BRAKE_RELEASE_SETTINGS, brakeReleaseSettings, SLOWLY_RELEASE_BRAKE } from './phrase-brake-release';
import { comparePriority, schedulePhrases, type PhraseScheduleCandidate, type PhraseScheduleCategory, type PhraseScheduleItem, type PhraseScheduleSnapshot } from './phrase-scheduler';

export const TELEMETRY_MAX_AGE_MS = 1500;
const CLOSURE_PREPARATION_MS = 1000;
const DECELERATION_INTERVAL_MS = 500;
export const REARM_MS = 500;
export const COOLDOWN_MS = 8000;
const HISTORY_LIMIT = 50;

const INPUT_LABELS = {
    drivingAggression: 'Instructor aggression',
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
    referenceBrakeRequired: 'Braking required by reference or braking distance',
    referencePedalZone: 'Interpolated reference pedal zone',
    referenceCurrentPedalZone: 'Reference pedal zone at the current position',
    referenceSteeringPhase: 'Reference steering phase',
    referenceCornerEntry: 'First reference turning arc inside the mapped corner entry',
    referenceCornerPhase: 'Reference corner technique',
    referenceCornerInSequence: 'Reference corner is inside a Live Map consecutive corners label',
    referenceSequenceRemaining: 'Corners remaining after the reference corner in the mapped sequence',
    referenceNextCornerSetup: 'Reference track position and direction of the next linked corner',
    referenceSteeringDirection: 'Required steering correction',
    referenceBrakeReleaseRate: 'Reference deceleration easing rate while braking (m/s³)',
} as const;
type Input = keyof typeof INPUT_LABELS;
type Inputs = PhraseMapContext & ReturnType<typeof getPhrasePositions> & {
    drivingAggression?: number;
    speed?: number; referenceCue?: ReferenceCue | 'none'; driverOnStraight?: 0 | 1; referenceBraking?: 0 | 1;
    referenceBrakeRequired?: 0 | 1;
    referencePedalZone?: ReferencePedalZone;
    referenceCurrentPedalZone?: ReferencePedalZone;
    referenceBrakeReleaseRate?: number;
    referenceSteeringPhase?: ReferenceSteeringPhase | 'none';
    referenceCornerEntry?: 0 | 1;
    referenceCornerPhase?: 'hold' | 'unwind';
    referenceCornerInSequence?: 0 | 1;
    referenceSequenceRemaining?: number;
    referenceNextCornerSetup?: ConsecutiveCornerSetup;
    referenceSteeringDirection?: ReferenceSteeringDirection;
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
                brake: 'Braking distance requires braking now or reference brake demand is at least 10%, and the brake is not yet applied',
                steering: 'The live phrase reference direction-change range is active within the centerline corner start and end.',
                cornerExit: 'Reference steering is unwinding after its final loaded arc within the mapped corner, with the brake released',
                throttle: 'Throttle demand for the target speed is at least 15% and the throttle is not yet applied, with brake below 10% and clear of braking',
                moreBrake: 'Current measured deceleration is below reference deceleration at the current position; the current brake is at least 10%, below full pressure, and not being released. This must remain true for 500 ms',
                releaseBrake: 'The brake is applied, predicted speed is at or below the local reference, and the remaining speed profile needs no additional braking after coasting and road slope are accounted for',
                easeBrake: 'The brake is applied during cornering and further braking is needed to follow the reference speed and deceleration profile',
                holdBrake: 'Speed is within 5 km/h of the local reference and lookahead deceleration matches the required rate within 10% or 0.1 m/s², whichever is greater',
            };
            const movement = REFERENCE_MOVEMENTS.find(({ cue }) => cue === this.value);
            return movement ? `Move to the ${movement.direction}: ${movement.description}`
                : descriptions[this.value as ReferenceCue] ?? INPUT_LABELS.referenceCue;
        }
        if (this.operator === 'in' || this.operator === '=') {
            if (this.input === 'referenceNextCornerSetup') {
                const setups = Array.isArray(this.value) ? this.value : [this.value];
                const sides = Array.from(new Set(setups.map((setup) => String(setup).split('-for-')[0])))
                    .map((side) => side === 'middle' ? 'center' : side);
                return `The reference line specifies ${sides.join(' or ')} positioning for the next linked corner with a known direction`;
            }
            if (this.input === 'referenceSteeringDirection') return `Predicted motion needs a correction to the ${this.value} along the adjusted reference curve`;
            if (this.input === 'referenceSteeringPhase') return this.value === 'prepare'
                ? 'The reference direction-change range is active within the centerline corner bounds; reference steering has not reached the 0.08 turn-in threshold'
                : 'The reference direction-change range is active within the centerline corner bounds; reference steering has reached the 0.08 turn-in threshold';
            if (this.input === 'driverOnStraight') return INPUT_LABELS.driverOnStraight;
            if (this.input === 'referenceBraking' && this.value === 0) return 'Reference is not braking and no brake correction is needed';
            if (this.input === 'referenceBraking' && this.value === 1) return INPUT_LABELS.referenceBraking;
            if (this.input === 'referenceBrakeRequired') return INPUT_LABELS.referenceBrakeRequired;
            if (this.input === 'referencePedalZone') return `Inside the interpolated reference ${this.value} zone`;
            const values = Array.isArray(this.value) ? this.value : [this.value];
            if (this.input === 'approachCornerDirection') {
                return `Upcoming corner turns ${values.join(' or ')}`;
            }
            if (this.input === 'approachCornerInSequence' && this.value === 0) {
                return 'Corner is not inside a consecutive corners label of the Live Map';
            }
            if (this.input === 'referenceCornerInSequence') return this.value === 0
                ? 'Reference corner is not inside a Live Map consecutive corners label' : INPUT_LABELS.referenceCornerInSequence;
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

export function conditionsMatch(conditions: readonly PhraseConditionNode[]): boolean {
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

/** A playable phrase leaf. Entry/exit rules belong here; audio variations are only delivery data. */
export interface PhraseActionDefinition {
    sentence: string;
    /** Higher values are selected first when multiple predictions are eligible. */
    priority: number;
    reactionAllowanceMs?: number;
    /** Optional driving exit, independent of the entry condition and speech level. */
    exitConditions?: readonly PhraseConditionNode[];
    amplifyCurves?: PhraseCurves;
    speechSettings?: PhraseSettings;
    /** All possible overrides must be included in preloadSettings; playback never synthesizes. */
    instanceSettings?: (context: PhraseContext, phrase: Phrase) => PhraseSettings;
    preloadSettings?: readonly PhraseSettings[];
    /** Extra prepared deliveries in the random pool; audio has no entry or exit conditions. */
    audioVariations?: readonly PhraseSettings[];
    /** Interchangeable preloaded wordings, selected randomly from the scheduler's allowed pool. */
    randomAlternatives?: readonly string[];
    /** Phrase entry conditions within its enclosing guide (legacy field name). */
    actionConditions?: readonly PhraseConditionNode[];
}
export interface PhraseDefinition extends PhraseActionDefinition {
    preparationTimeMs?: number;
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
        const priority = action.priority;
        const source = new Phrase(id, action.sentence, action.amplifyCurves, {
            ...action.speechSettings, priority, randomAlternatives: action.randomAlternatives,
        });
        return {
            id, sentence: action.sentence, conditions: action.actionConditions, priority, phrase: source,
            instanceSettings: action.instanceSettings,
            preloadSettings: action.preloadSettings,
            audioVariations: action.audioVariations,
            reactionAllowanceMs: action.reactionAllowanceMs ?? PHRASE_REACTION_ALLOWANCE_MS,
            exitConditions: action.exitConditions,
        };
    });
}
const condition = (input: Input, operator: PhraseCondition['operator'], value: PhraseCondition['value'], connector: PhraseConditionConnector = 'and'): PhraseCondition => new PhraseCondition(input, operator, value, undefined, connector);
const movementConditions = [condition('drivingAggression', '=', 100), condition('driverOnStraight', '=', 1), condition('referenceBraking', '=', 0)];
const STEERING_CLOSURE_ID = 'reference-steering';
const TURN_IN_MEMORY_PREFIX = 'spoken-turn-in:';
const isTurnIn = (phrase: PhraseDefinition) => phrase.id === STEERING_CLOSURE_ID;
const CORNER_EXIT_CLOSURE_ID = 'reference-cornerExit';
const TRACK_GUIDE_ID = 'track-guide';
const referenceRangeKey = (guidance: PhraseContext['guidance']) => {
    const range = guidance.steeringRange;
    return range ? `${range.start}:${range.end}:${range.turnSign}` : undefined;
};
const turnInCornerKey = (guidance: PhraseContext['guidance']) => {
    const corner = guidance.turnInCorner;
    return corner ? `${corner.start}:${corner.end}` : undefined;
};
const brakeCues: readonly ReferenceCue[] = ['brake', 'moreBrake', 'releaseBrake', 'easeBrake', 'holdBrake'];
function getPhraseCategory(phrase: PhraseDefinition): PhraseScheduleCategory {
    if (phrase.referenceCue && brakeCues.includes(phrase.referenceCue)) return 'brake';
    if (phrase.referenceCue === 'throttle') return 'throttle';
    return phrase.referenceCue === 'steering' || phrase.referenceCue === 'cornerExit' || phrase.movementDirection ? 'turn' : 'other';
}
export const conditionGroup = (conditions: readonly PhraseConditionNode[], connector: PhraseConditionConnector = 'and'): PhraseConditionGroup => new PhraseConditionGroup(conditions, connector);
const following = [condition('speed', '>=', 30), condition('carAhead', '=', 1)];
const positioned = [
    condition('playerCorner', 'in', ['left', 'right']), condition('playerPosition', 'in', ['left', 'middle', 'right']),
    condition('opponentCorner', 'in', ['left', 'right']), condition('opponentPosition', 'in', ['left', 'middle', 'right']),
];
const exitSpeedAdvice: PhraseActionDefinition = {
    sentence: 'Leave room. Pass on exit.', priority: 0,
    actionConditions: [condition('betterCornerExitSpeed', '=', 1), condition('opponentDistanceM', '<=', 2),
        condition('approachLongStraightAfterCorner', '=', 1)],
};
// Sentence metadata is compiled into child closures by createPhraseRoot.
// More specific tactics take priority when entering from the root.
export const PHRASE_DEFINITIONS: readonly PhraseDefinition[] = [
    {
        id: 'second-apex', name: 'Chicane overtake', holdMs: 0, priority: 0,
        description: 'Prepare for opposite linked corners, then speak a prepared clip’s duration plus reaction allowance before the opponent reaches corner entry: go wide if the opponent is inside, or brake early and hold inside if the opponent is outside.',
        sentence: 'Go wide. Aim for second apex.',
        conditions: [condition('approachSequenceRemaining', '>=', 1), condition('approachCornerSpeed', '=', 'slow'),
            condition('approachLinkedOpposite', '=', 1)],
        actionConditions: [condition('opponentCornerEtaS', '<', 0.5), conditionGroup([
            conditionGroup([condition('approachCornerDirection', '=', 'left'), condition('opponentPosition', '=', 'left')]),
            conditionGroup([condition('approachCornerDirection', '=', 'right'), condition('opponentPosition', '=', 'right')], 'or'),
        ])],
        additionalActions: [{
            sentence: 'Brake early. Hold inside.', priority: 0,
            actionConditions: [condition('opponentCornerEtaS', '<', 0.5), conditionGroup([
                conditionGroup([condition('approachCornerDirection', '=', 'right'), condition('opponentPosition', '=', 'left')]),
                conditionGroup([condition('approachCornerDirection', '=', 'left'), condition('opponentPosition', '=', 'right')], 'or'),
            ])],
        }, exitSpeedAdvice],
    },
    {
        id: 'one-tight-slow-corner', name: 'One Tight Slow Corner', holdMs: 0, priority: 0,
        description: 'Follow an opponent toward the next corner labeled slow on the Live Map and outside any consecutive corners label: slipstream to a late brake when the opponent is outside, or brake earlier and clip the apex behind a defending opponent in the middle.',
        sentence: 'Inside pass possible.',
        conditions: [condition('approachCornerSpeed', '=', 'slow'), condition('approachCornerInSequence', '=', 0)],
        actionConditions: [
            condition('opponentDistanceM', '<=', 2),
            conditionGroup([
                conditionGroup([condition('opponentPosition', '=', 'left'), condition('approachCornerDirection', '=', 'right')]),
                conditionGroup([condition('opponentPosition', '=', 'right'), condition('approachCornerDirection', '=', 'left')], 'or'),
            ]),
        ],
        additionalActions: [{
            sentence: 'They defend. Stay close.', priority: 0,
            actionConditions: [
                conditionGroup([condition('opponentPosition', '=', 'middle'), condition('playerPosition', '=', 'right'),
                    condition('approachCornerDirection', '=', 'right')]),
                conditionGroup([condition('opponentPosition', '=', 'middle'), condition('playerPosition', '=', 'left'),
                    condition('approachCornerDirection', '=', 'left')], 'or'),
            ],
        }, exitSpeedAdvice],
    },
    {
        id: 'slipstream', name: 'Slipstream', holdMs: 800, priority: 0,
        description: 'Move behind a nearby opponent on a straight to build a run when the gap is steady or increasing.',
        sentence: 'Tuck in. Build a run.',
        conditions: [condition('speed', '>=', 80), condition('carAhead', '=', 1), condition('phase', '=', 'straight'),
            condition('opponentDistanceM', '<=', 10), condition('closingOnOpponent', '=', 0), condition('directlyBehindOpponent', '=', 0)],
    },
    {
        id: 'pressure-feint', name: 'Pressure and a feint', holdMs: 800, priority: 0,
        description: 'Show a controlled attack before braking and use an opening if the opponent reacts.',
        sentence: 'Show a move. Watch for gaps.',
        conditions: [...following, ...positioned, condition('phase', '=', 'entry'), condition('cornerSpeed', 'in', ['slow', 'fast']), condition('insideLine', '=', 0), condition('outsideLine', '=', 0)],
    },
    ...([
        ['brake', 3, 'Brake', 'Brake.', 'Apply the brake early enough to reach upcoming reference speeds using the braking strength captured at reference load, preserving earlier Instructor aggression braking points.'],
        ['moreBrake', 3, 'More brake', 'More brake.', 'Increase pressure after current measured deceleration stays below reference deceleration at the current position for 500 ms, without an active pedal release.'],
        ['releaseBrake', 3, 'Release the brake', 'Off brake.', 'Release the pedal once speed is at or below the local reference and no further braking is needed through the remaining corner.'],
        ['holdBrake', 3, 'Hold brake pressure', 'Hold brake.', 'Keep brake pressure steady while both speed and deceleration match the local adjusted reference.'],
    ] as const).map(([referenceCue, priority, name, sentence, description]) => ({
        id: `reference-${referenceCue}`, referenceCue, name, sentence, description, priority,
        ...(referenceCue === 'brake' ? { amplifyCurves: brakeCurves } : {}),
        holdMs: referenceCue === 'moreBrake' ? 500 : 0,
        conditions: [...(referenceCue === 'releaseBrake' ? []
            : referenceCue === 'brake' ? [condition('referenceBrakeRequired', '=', 1)]
                : [condition(referenceCue === 'moreBrake' ? 'referenceCurrentPedalZone' : 'referencePedalZone', '=', 'brake')]),
            condition('referenceCue', '=', referenceCue)],
    })),
    {
        id: STEERING_CLOSURE_ID, referenceCue: 'steering', name: 'Turn in', holdMs: 0, priority: 2,
        description: 'At 100% Instructor aggression, announce turn-in once at each mapped corner entry using this cue’s prepared audio duration.',
        sentence: 'Turn in',
        conditions: [condition('drivingAggression', '=', 100), condition('referenceCornerEntry', '=', 1), condition('referenceCue', '=', 'steering')],
    },
    {
        id: 'reference-steering-release', referenceCue: 'easeBrake', name: SLOWLY_RELEASE_BRAKE, holdMs: 0, priority: 3,
        description: 'Ease brake pressure while cornering when the driver still needs braking to follow the reference speed and deceleration, accounting for coasting, turn radius and road slope.',
        sentence: SLOWLY_RELEASE_BRAKE, amplifyCurves: releaseBrakeCurves,
        conditions: [condition('referenceCue', '=', 'easeBrake')],
        instanceSettings: (context, phrase) => brakeReleaseSettings(context.guidance.brakeReleaseDurationMs,
            (settings) => context.speechDurationMs?.(phrase.createInstance(settings))),
        preloadSettings: BRAKE_RELEASE_SETTINGS,
    },
    ...(['left', 'center', 'right'] as const).map((side): PhraseDefinition => ({
        id: `reference-consecutive-corners-${side}`, referenceCue: 'steering', name: `Set up ${side}`, holdMs: 0, priority: 2,
        description: `At 100% Instructor aggression, set up ${side} for the next linked corner using its reference entry position and known direction. ${side === 'center' ? 'Wait for the final steering unwind' : 'Prepare during the current loaded arc or steering release'}. Speak once per reference turn range, only with a known next corner and valid reference position between the mapped track boundaries.`,
        sentence: `Set up ${side}`,
        conditions: [condition('drivingAggression', '=', 100), condition('speed', '>=', 20),
            condition('referenceCornerInSequence', '=', 1), condition('referenceSequenceRemaining', '>=', 1),
            condition('referenceNextCornerSetup', 'in', ['left', 'right'].map((direction) =>
                `${side === 'center' ? 'middle' : side}-for-${direction}`)),
            condition('referenceCue', '=', 'steering'),
            condition('referenceCornerPhase', 'in', side === 'center' ? ['unwind'] : ['hold', 'unwind'])],
    })),
    {
        id: 'reference-steering-hold', referenceCue: 'steering', name: 'Hold line', holdMs: 0, priority: 2,
        description: 'At 100% Instructor aggression, hold the line when aligned on a stable reference arc.',
        sentence: 'Hold line', conditions: [condition('drivingAggression', '=', 100), condition('referenceCue', '=', 'steering'),
            condition('referenceCornerPhase', '=', 'hold'), condition('referenceSteeringDirection', '=', 'aligned')],
    },
    {
        id: CORNER_EXIT_CLOSURE_ID, referenceCue: 'cornerExit', name: 'Unwind', holdMs: 0, priority: 2,
        description: 'At 100% Instructor aggression, open the steering as the reference unwinds after its final loaded arc outside a mapped consecutive-corners sequence. Speak once per reference turn range; braking takes priority.',
        sentence: 'Unwind', conditions: [condition('drivingAggression', '=', 100),
            condition('referenceCornerInSequence', '=', 0), condition('referenceCue', '=', 'cornerExit')],
    },
    ...REFERENCE_MOVEMENTS.map(({ cue, direction, name, sentence, randomAlternatives, description, priority }) => ({
        id: `reference-${cue}`, referenceCue: cue, movementDirection: direction, name: `${name} · ${direction}`, sentence, randomAlternatives, description, holdMs: 0, priority,
        conditions: [condition('speed', '>=', 20), ...movementConditions, condition('referenceCue', '=', cue)],
    })),
    {
        id: 'reference-throttle', referenceCue: 'throttle', name: 'Throttle', sentence: 'Squeeze gas.', holdMs: 0, priority: 1,
        description: 'Apply throttle using reference throttle and acceleration to follow the speed profile adjusted by Instructor aggression.',
        conditions: [condition('speed', '>=', 20), condition('referencePedalZone', '=', 'throttle'), condition('referenceCue', '=', 'throttle')],
    },
];

export type ClosureStatus = 'Not evaluated' | 'Missing input' | 'Not matched' | 'Confirming' | 'Waiting' | 'Active' | 'Cooldown';
export interface PhraseEvent {
    id: number; ruleId: string; sentence: string; timestamp: number; expiresAt?: number;
    /** Present on engine-produced events; optional for older snapshot consumers. */
    instance?: PhraseInstance;
    speech?: PhraseSpeechTiming;
    steeringPhase?: ReferenceSteeringPhase;
    brakeRelease?: boolean;
}
export interface PhraseSnapshot {
    schedule?: PhraseScheduleSnapshot;
    trackGuideEnabled: boolean;
    telemetryReady: boolean;
    visionReady: boolean;
    mapReady: boolean;
    referenceLoaded: boolean;
    referenceSteeringPhase?: ReferenceSteeringPhase;
    referenceSteeringDirection?: ReferenceSteeringDirection;
    referenceBrakeReleaseRate?: number;
    referenceBrakeReleaseDurationMs?: number;
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
    state: { current: string; description: string; kind: NodeSnapshot['kind']; path: string[] };
    events: PhraseEvent[];
}
interface PhraseMemory {
    since?: number; clearSince?: number; armed: boolean; lastEmitted?: number; referenceRange?: string;
    referenceExit?: { distance: number; remaining: number };
    enteredBrakeZone?: boolean;
}
export interface PhraseContext {
    entryPriorities?: Map<string, number>;
    scheduleCandidates?: Map<string, PhraseScheduleCandidate>;
    predictionOffsetMs?: number;
    trackGuideEnabled?: boolean;
    now: number;
    emit: boolean;
    telemetryReady: boolean;
    intendedTelemetry: StandardTelemetrySample;
    predictionFrames: readonly PhrasePredictionFrame[];
    inputs: Inputs;
    guidance: ReturnType<typeof readReferenceGuidance>;
    movementCue: () => ReferenceMovementCue | undefined;
    movementCanSpeak: () => boolean;
    referenceMovementAllowed?: boolean;
    driverLongStraightIds?: readonly string[];
    driverCornerIds?: readonly string[];
    memory: Map<string, PhraseMemory>;
    entryConditions: Map<string, readonly PhraseConditionNode[]>;
    actionConditions: Map<string, readonly PhraseConditionNode[]>;
    inspections: Map<string, NodeInspection>;
    candidate?: string;
    forPhrase?: (phrase: PhraseDefinition, actionId?: string) => PhraseContext;
    forPreparation?: (preparationTimeMs: number, matches?: (candidate: PhraseContext) => boolean) => PhraseContext;
    speechPending?: (actionId: string) => boolean;
    speechLeadMs?: (actionId: string) => number;
    speechDurationMs?: (instance: PhraseInstance) => number | undefined;
    audioVariants?: (phrase: PhraseDefinition, actionId: string, instances?: readonly PhraseInstance[]) => readonly PhraseAudioVariant[];
    scheduleAudio?: (phrase: PhraseDefinition, actionId: string) => Partial<PhraseScheduleCandidate>;
    publish: (phrase: PhraseDefinition, instance: PhraseInstance) => void;
}

function unevaluatedConditions(conditions: readonly PhraseConditionNode[]): PhraseConditionNode[] {
    return conditions.map((condition) => condition instanceof PhraseConditionGroup
        ? new PhraseConditionGroup(unevaluatedConditions(condition.conditions), condition.connector)
        : new PhraseCondition(condition.input, condition.operator, condition.value, undefined, condition.connector, false));
}

function measureAction(context: PhraseContext, id: string, conditions?: readonly PhraseConditionNode[]): boolean {
    const evaluated = conditions?.map((condition) => {
        // Corner ETA describes an upcoming action, so longer prepared instructions start earlier.
        if (condition instanceof PhraseCondition && condition.input === 'opponentCornerEtaS'
            && condition.operator === '<' && context.speechLeadMs) {
            return new PhraseCondition(condition.input, '<', Math.min(REFERENCE_LOOKAHEAD_MS, context.speechLeadMs(id)) / 1000,
                context.inputs.opponentCornerEtaS, condition.connector);
        }
        return condition.evaluate(context.inputs);
    });
    context.actionConditions.set(id, evaluated ?? []);
    return context.telemetryReady && (evaluated === undefined || conditionsMatch(evaluated));
}

function isSpeechPending(context: PhraseContext, state: State<PhraseContext>): boolean {
    return state.current instanceof Action && context.speechPending?.(state.current.metadata.id ?? '') === true;
}

/** An entered phrase owns the measurement and execution of its action children. */
class PhraseClosure extends Closure<PhraseContext> {
    constructor(readonly phrase: PhraseDefinition, ...args: ConstructorParameters<typeof Closure<PhraseContext>>) { super(...args); }
    evaluate(context: PhraseContext, state: State<PhraseContext>, onActionError?: (error: unknown) => void): unknown {
        if (!state.path.includes(this)) return;
        context = context.forPhrase?.(this.phrase) ?? context;
        state.check(this, context);
        return super.evaluate(context, state, onActionError);
    }
}

/** A track cue is a playable leaf in the shared guide. */
class TrackGuidePhrase extends Action<PhraseContext> {
    constructor(readonly phrase: PhraseDefinition, readonly entry: ReturnType<typeof phraseEntry>,
        ...args: ConstructorParameters<typeof Action<PhraseContext>>) { super(...args); }
}

/** Every cue measures independently, including while another cue is speaking. */
class TrackGuideClosure extends Closure<PhraseContext> {
    readyPriority = 0;
    get priority(): number { return this.readyPriority; }
    evaluate(context: PhraseContext, state: State<PhraseContext>, onActionError?: (error: unknown) => void): unknown {
        if (!state.path.includes(this)) return;
        this.condition(context, state);
        this.children.forEach((child) => { if (child instanceof TrackGuidePhrase) state.check(child, context); });
        if (state.current instanceof TrackGuidePhrase) {
            if (!isSpeechPending(context, state)) state.exitToParent();
            return;
        }
        if (state.closure === this && Array.from(context.entryPriorities ?? []).some(([id, priority]) =>
            priority > this.priority && !this.children.some((child) =>
                child instanceof TrackGuidePhrase && child.phrase.id === id))) {
            state.exitToRoot();
            return;
        }
        return super.evaluate(context, state, onActionError);
    }
}

/** Measure competing phrase scopes before dispatch, without entering their children. */
class PhraseRootClosure extends Closure<PhraseContext> {
    evaluate(context: PhraseContext, state: State<PhraseContext>, onActionError?: (error: unknown) => void): unknown {
        if (!state.path.includes(this)) return;
        this.children.forEach((child) => { if (child instanceof TrackGuideClosure) state.check(child, context); });
        [...this.children].sort(comparePriority).forEach((child) => {
            if (child instanceof PhraseClosure) state.check(child, context);
        });
        return super.evaluate(context, state, onActionError);
    }
}

function phraseEntry(phrase: PhraseDefinition) {
    const memories = new WeakMap<State<PhraseContext>, PhraseMemory>();
    const entry = describeCondition<PhraseContext>((context, state) => {
        context = context.forPhrase?.(phrase) ?? context;
        const { now, inputs, guidance } = context;
        const memory = context.memory.get(phrase.id) ?? { armed: true };
        context.memory.set(phrase.id, memory);
        memories.set(state, memory);
        const rangeKey = referenceRangeKey(guidance);
        const cornerKey = turnInCornerKey(guidance);
        const memoryKey = isTurnIn(phrase) ? cornerKey : rangeKey;
        const oncePerRange = phrase.referenceCue === 'cornerExit' || phrase.referenceCue === 'steering' || phrase.referenceCue === 'easeBrake';
        if (oncePerRange && memoryKey !== undefined) {
            if (memory.referenceRange !== memoryKey) {
                memory.referenceRange = memoryKey;
                memory.armed = true;
                memory.lastEmitted = undefined;
            }
        }
        const exitMemory = context.memory.get(CORNER_EXIT_CLOSURE_ID);
        const unwindSpoken = rangeKey !== undefined && exitMemory?.referenceRange === rangeKey && !exitMemory.armed;
        const turnInSpoken = cornerKey !== undefined && context.memory.has(`${TURN_IN_MEMORY_PREFIX}${cornerKey}`);
        const cue = phrase.movementDirection ? context.referenceMovementAllowed && context.movementCue() === phrase.referenceCue : phrase.referenceCue && guidance[phrase.referenceCue] !== undefined
            && (!isTurnIn(phrase) || (guidance.cornerExit === undefined && !unwindSpoken && !turnInSpoken));
        const referenceReady = guidance.ready && (!phrase.movementDirection || guidance.lateralOffset !== undefined);
        const conditions = phrase.conditions.map((condition) => condition.evaluate(phrase.referenceCue
            ? { ...inputs, referenceCue: cue ? phrase.referenceCue : referenceReady ? 'none' : undefined } : inputs));
        context.entryConditions.set(phrase.id, conditions);
        const fits = context.telemetryReady && conditionsMatch(conditions);
        const rearmAfterClear = phrase.referenceCue !== 'brake' && !oncePerRange;
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
        if (fits) {
            if (!phrase.referenceCue && !context.candidate) context.candidate = phrase.id;
            memory.since ??= now;
        } else memory.since = undefined;
        const cooldown = !oncePerRange && memory.lastEmitted !== undefined
            && now - memory.lastEmitted < (phrase.referenceCue ? REFERENCE_CUE_COOLDOWN_MS : COOLDOWN_MS);
        const status: ClosureStatus = !context.telemetryReady || (!fits && missingConditionInputs(conditions).length > 0) ? 'Missing input'
            : !fits ? 'Not matched' : !selected || !memory.armed ? 'Waiting'
                : memory.since === undefined || now - memory.since < phrase.holdMs ? 'Confirming' : cooldown ? 'Cooldown' : 'Confirming';
        context.inspections.set(phrase.id, { status });
        const eligible = fits && memory.armed && memory.since !== undefined
            && now - memory.since >= phrase.holdMs && !cooldown && (!phrase.movementDirection || context.movementCanSpeak());
        if (eligible) context.entryPriorities?.set(phrase.id, phrase.priority);
        else context.entryPriorities?.delete(phrase.id);
        return eligible && context.emit && Boolean(phrase.additionalActions?.length || !context.audioVariants
            || context.audioVariants(phrase, `${phrase.id}:say:0`).length > 0);
    }, (context) => snapshotConditions(context.entryConditions.get(phrase.id) ?? []), snapshotConditions(unevaluatedConditions(phrase.conditions)));
    entry.reset = (state) => {
        const memory = memories.get(state);
        if (!memory || phrase.referenceCue) return;
        memory.since = memory.clearSince = undefined;
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

function trackPhraseConditions(phrase: PhraseDefinition, action: ReturnType<typeof getPhraseActions>[number], context?: PhraseContext): ConditionSnapshot[] {
    const entry = context?.entryConditions.get(phrase.id) ?? unevaluatedConditions(phrase.conditions);
    const conditions = context?.actionConditions.get(action.id) ?? unevaluatedConditions(action.conditions ?? []);
    return snapshotConditions(conditions.length
        ? [new PhraseConditionGroup(entry), new PhraseConditionGroup(conditions)] : entry);
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
        ], { id: `${phrase.id}:inside-long-straight`, priority: phrase.priority });
}

function createInsideCornerClosure(phrase: PhraseDefinition): Closure<PhraseContext> {
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
        ], { id: 'one-tight-slow-corner:inside-corner', priority: phrase.priority });
}

export function createPhraseRoot(phrases: readonly PhraseDefinition[] = PHRASE_DEFINITIONS): Closure<PhraseContext> {
    const nodes = phrases.flatMap<PhraseClosure | TrackGuidePhrase>((phrase) => {
        const entry = phraseEntry(phrase);
        const oncePerRange = phrase.referenceCue === 'cornerExit' || phrase.referenceCue === 'steering' || phrase.referenceCue === 'easeBrake';
        const fields = { 'Hold for': `${phrase.holdMs / 1000} s`,
            ...(oncePerRange ? {} : { Cooldown: `${(phrase.referenceCue ? REFERENCE_CUE_COOLDOWN_MS : COOLDOWN_MS) / 1000} s` }),
            'Clear period': isTurnIn(phrase) ? 'Until the driver leaves the mapped corner'
                : oncePerRange ? 'Until a different reference turn range'
                : phrase.referenceCue === 'brake' ? 'Until leaving the measured brake zone' : `${REARM_MS / 1000} s` };
        const actions = getPhraseActions(phrase);
        const say = actions.map((action) => {
            const readyAudio = new WeakMap<State<PhraseContext>, readonly PhraseInstance[]>();
            const prepareAudio = (context: PhraseContext) => {
                context = context.forPhrase?.(phrase, action.id) ?? context;
                const instances = action.phrase.createAudioInstances(action.instanceSettings
                    ? [action.instanceSettings(context, action.phrase)] : [{}, ...(action.audioVariations ?? [])]);
                return context.audioVariants?.(phrase, action.id, instances).map(({ instance }) => instance) ?? instances;
            };
            return new Action<PhraseContext>('say phrase', action.sentence, (context, state) => {
                const allowed = readyAudio.get(state) ?? [];
                const selected = action.phrase.selectAudioInstance(allowed);
                if (!selected) return;
                context.publish(phrase, selected);
            }, describeCondition((context, state) => {
                readyAudio.delete(state);
                const forecast = context.forPhrase?.(phrase, action.id) ?? context;
                if (!measureAction(forecast, action.id, action.conditions)
                    || (action.exitConditions && conditionsMatch(action.exitConditions.map((condition) => condition.evaluate(context.inputs))))) {
                    context.scheduleCandidates?.delete(action.id);
                    return false;
                }
                // Each phrase supplies its own prediction; scheduling never inspects its enclosing scopes.
                context.scheduleCandidates?.set(action.id, {
                    id: action.id, kind: 'phrase', label: action.sentence, priority: action.priority,
                    category: getPhraseCategory(phrase), predictedAt: context.now + (forecast.predictionOffsetMs ?? 0),
                    durationMs: context.speechLeadMs?.(action.id) ?? 0,
                    durationEstimated: context.speechLeadMs === undefined,
                    ...context.scheduleAudio?.(phrase, action.id),
                });
                if (isSpeechPending(context, state) || state.current !== state.closure || !context.emit) return false;
                const allowed = prepareAudio(context);
                readyAudio.set(state, allowed);
                return allowed.length > 0;
            },
                action.conditions ? (context) => snapshotConditions(context.actionConditions.get(action.id) ?? []) : 'Always eligible',
                action.conditions ? snapshotConditions(unevaluatedConditions(action.conditions)) : undefined),
            { id: action.id, kind: 'phrase', priority: action.priority, inspect: (context, state) => ({ fields: {
                'Preparation time': context.speechLeadMs ? `${context.speechLeadMs(action.id) / 1000} s` : 'Awaiting prepared speech',
                ...(context.audioVariants ? { 'Allowed audio variants': readyAudio.get(state)?.length ?? 0 } : {}),
            } }), fields: {
                'Audio variants': action.phrase.createAudioInstances([
                    {}, ...(action.preloadSettings ?? []), ...(action.audioVariations ?? []),
                ]).length,
                'Reaction allowance': `${action.reactionAllowanceMs} ms`,
                'Phrase exit': 'The selected audio finishes, or the phrase exit condition matches',
                ...(action.phrase.randomAlternatives.length > 1 ? { 'Phrase selection': 'Random (equal chance)',
                    ...Object.fromEntries(action.phrase.randomAlternatives.slice(1).map((sentence, index) => [`Random alternative ${index + 1}`, sentence])) } : {}),
            } });
        });
        if (phrase.referenceCue) return say.map((action, index) => new TrackGuidePhrase(phrase, entry,
            phrase.name, action.description, action.run, describeCondition((context, state) => {
                const ready = entry(context, state);
                if (!ready && !context.entryPriorities?.has(phrase.id)) return false;
                context = context.forPhrase?.(phrase) ?? context;
                return !(phrase.exitConditions && conditionsMatch(phrase.exitConditions.map((condition) => condition.evaluate(context.inputs))))
                    && action.condition(context, state);
            },
            (context) => trackPhraseConditions(phrase, actions[index], context), trackPhraseConditions(phrase, actions[index])), {
                ...action.metadata, fields: { ...fields, ...action.metadata.fields },
                inspect: (context, state) => ({ ...context.inspections.get(phrase.id), ...action.metadata.inspect?.(context, state) }),
            }));
        const parentSay = phrase.id === 'one-tight-slow-corner' ? say.slice(1, 2)
            : phrase.id === 'second-apex' ? say.slice(0, 2) : say;
        const children = phrase.id === 'one-tight-slow-corner'
            ? [createInsideLongStraightClosure(phrase, [...say.slice(2), say[0]]), createInsideCornerClosure(phrase)]
            : phrase.id === 'second-apex' ? [createInsideLongStraightClosure(phrase, say.slice(2))] : [];
        return [new PhraseClosure(phrase, phrase.name, phrase.description, entry, [
            ...(phrase.exitConditions ? [Action.exitToParent<PhraseContext>(describeCondition(
                (context) => conditionsMatch(phrase.exitConditions!.map((condition) => condition.evaluate(context.inputs))),
                (context) => snapshotConditions(phrase.exitConditions!.map((condition) => condition.evaluate(context.inputs))),
                snapshotConditions(unevaluatedConditions(phrase.exitConditions))), { priority: Number.MAX_SAFE_INTEGER })] : []),
            ...children, ...parentSay,
            Action.exitToRoot(describeCondition<PhraseContext>((context, state) =>
                parentSay.some((action) => state.getExecution(action).status === 'completed'
                    && !context.speechPending?.(action.metadata.id!)),
                'The phrase releases after the final spoken sound, or when its exit condition matches')),
        ], { id: phrase.id, preparationTimeMs: phrase.preparationTimeMs,
            inspect: (context) => context.inspections.get(phrase.id) ?? {},
            fields,
            priority: phrase.priority,
        })];
    });
    const guides = nodes.filter((node): node is TrackGuidePhrase => node instanceof TrackGuidePhrase);
    const trackGuide: TrackGuideClosure = new TrackGuideClosure('Track guide',
        'Evaluate every track cue independently using its prepared speech duration, then yield when no cue is ready.',
        describeCondition((context, state) => {
            if (context.trackGuideEnabled === false) { trackGuide.readyPriority = 0; return false; }
            // Evaluate every cue so confirmation holds continue independently of priority.
            const ready = guides.filter((guide) => guide.entry(context, state));
            trackGuide.readyPriority = Math.max(0, ...guides.filter((guide) =>
                ready.includes(guide) || context.entryPriorities?.has(guide.phrase.id)).map((guide) => guide.priority));
            return ready.length > 0;
        }, 'Track guide is enabled and a cue is ready'), [
            ...guides, Action.exitToRoot<PhraseContext>(),
        ], { id: TRACK_GUIDE_ID, priority: Math.max(0, ...guides.map((guide) => guide.priority)) });
    return new PhraseRootClosure('root',
        'Select an eligible track or overtaking cue. Track cues share a group and use independent audio lookahead.',
        describeCondition(() => true, 'Always eligible'), [
            ...(guides.length ? [trackGuide] : []), ...nodes.filter((node) => node instanceof PhraseClosure),
        ]);
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
    private currentDeceleration?: number;
    private currentBrakeReleasing = false;
    private brakeReleasing = false;
    private decelerationSample?: { speed: number; brake?: number; receivedAt: number };
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
    // Retain original car behaviour while caution rebuilds only the target speed/pedals.
    private referenceBaseline: TopLapReferenceEntry | null = null;
    private possibleBrakingDeceleration?: number;
    private carPerformance?: ReferenceCarPerformance;
    private referenceLap: TopLapReferenceEntry | null = null;
    private drivingAggression = REFERENCE_DRIVING_AGGRESSION;
    private trackGuideEnabled = true;
    private sessionGeneration?: number;
    private streamGeneration?: number;
    private referenceMovement = new ReferenceMovement();
    private mapContext = createPhraseMapContext(null);
    private consecutiveCornerPositioning = createConsecutiveCornerPositioning(null, null);
    private sectionId?: string;
    private game?: string;
    private opponentMotion = new PhraseOpponentMotion();
    private cornerSummaries: readonly DriverCornerSummary[] = [];
    private phraseClosures = new Map<string, Closure<PhraseContext>>();
    private speechDurations?: ReadonlyMap<string, number>;
    private speechActions = new Map<string, ReturnType<typeof getPhraseActions>[number]>();
    private preparedAudio = new Map<string, readonly PhraseAudioVariant[]>();
    private randomAudioPools = new Map<string, readonly PhraseAudioVariant[]>();
    private speechLeads = new Map<string, number>();

    constructor(
        private readonly phrases: readonly PhraseDefinition[] = PHRASE_DEFINITIONS,
        createRoot: (phrases: readonly PhraseDefinition[]) => Closure<PhraseContext> = createPhraseRoot,
    ) {
        this.root = createRoot(phrases);
        phrases.flatMap(getPhraseActions).forEach((action) => this.speechActions.set(action.id, action));
        const register = (node: Closure<PhraseContext> | Action<PhraseContext>) => {
            if (node instanceof Closure) {
                if (node.metadata.id) this.phraseClosures.set(node.metadata.id, node);
                node.children.forEach(register);
            }
        };
        register(this.root);
        this.state = this.createState();
    }

    private createState(): State<PhraseContext> {
        return new State(this.root, (node, context, matches) => node instanceof PhraseClosure || node instanceof TrackGuideClosure
            ? context : context.forPreparation?.(node.preparationTimeMs, matches) ?? context);
    }

    /** Enable playback tracking; only prepared audio identities supply scheduling durations. */
    setSpeechDurations(durations: ReadonlyMap<string, number>): void {
        this.speechDurations = new Map(Array.from(durations).filter(([, duration]) => Number.isFinite(duration) && duration >= 0));
        this.preparedAudio.clear();
        this.randomAudioPools.clear();
        this.speechActions.forEach((action) => {
            const variants = action.phrase.createAudioInstances([
                {}, ...(action.preloadSettings ?? []), ...(action.audioVariations ?? []),
            ]).flatMap((instance) => {
                const durationMs = this.speechDurations!.get(instance.cacheKey);
                return durationMs === undefined ? [] : [{ instance, durationMs }];
            });
            this.preparedAudio.set(action.id, variants);
            const randomKeys = new Set(action.phrase.createAudioInstances([
                {}, ...(action.instanceSettings ? action.preloadSettings ?? [] : action.audioVariations ?? []),
            ]).map(({ cacheKey }) => cacheKey));
            this.randomAudioPools.set(action.id, variants.filter(({ instance }) => randomKeys.has(instance.cacheKey)));
        });
        this.speechLeads.clear();
        this.events = this.events.map((event) => event.speech && event.instance
            ? { ...event, speech: { ...event.speech, durationMs: this.speechDurations!.get(event.instance.cacheKey),
                preparationTimeMs: this.speechLeadMs(event.instance.id) } } : event);
    }

    private speechLeadMs(actionId: string): number {
        const cached = this.speechLeads.get(actionId);
        if (cached !== undefined) return cached;
        const action = this.speechActions.get(actionId);
        const durations = this.preparedAudio.get(actionId)?.map(({ durationMs }) => durationMs) ?? [];
        // Search the phrase's entry window without selecting an audio variant.
        const lead = Math.max(0, ...durations) + (action?.reactionAllowanceMs ?? PHRASE_REACTION_ALLOWANCE_MS);
        this.speechLeads.set(actionId, lead);
        return lead;
    }

    speechRequested(eventId: number, now: number): PhraseSnapshot {
        this.events = this.events.map((event) => event.id === eventId && event.speech && event.speech.requestedAt === undefined
            && event.speech.finishedAt === undefined
            ? { ...event, speech: { ...event.speech, requestedAt: now } } : event);
        return this.evaluate(now);
    }

    playbackStarted(eventId: number, now: number): PhraseSnapshot {
        this.events = this.events.map((event) => event.id === eventId && event.speech && event.speech.playbackStartedAt === undefined
            && event.speech.finishedAt === undefined
            ? { ...event, speech: { ...event.speech, playbackStartedAt: now } } : event);
        return this.evaluate(now);
    }

    speechStarted(eventId: number, now: number): PhraseSnapshot {
        this.events = this.events.map((event) => event.id === eventId && event.speech && event.speech.startedAt === undefined
            && event.speech.releaseAt === undefined
            ? { ...event, speech: { ...event.speech, startedAt: now } } : event);
        return this.evaluate(now);
    }

    speechFinished(eventId: number, now: number, cancelled = false, outcome?: PhraseSpeechTiming['outcome']): PhraseSnapshot {
        this.events = this.events.map((event) => event.id === eventId && event.speech
            && event.speech.finishedAt === undefined
            ? { ...event, speech: { ...event.speech, finishedAt: now, outcome: outcome ?? (cancelled ? 'cancelled' : 'completed'),
                releaseAt: event.speech.releaseAt ?? now } } : event);
        return this.evaluate(now);
    }

    private speechPending(actionId: string): boolean {
        const event = this.events.slice().reverse().find((event) => event.instance?.id === actionId);
        return Boolean(event?.speech && event.speech.releaseAt === undefined);
    }

    receiveMap(map: CircuitMapDto | null, now: number, referenceLap: TopLapReferenceEntry | null = null): PhraseSnapshot {
        const referenceChanged = referenceLap !== this.sourceReferenceLap || map !== this.sourceMap;
        if (referenceLap !== this.sourceReferenceLap) {
            // Keep source identity only to validate Live Map's fresh lateral-position predictions.
            this.sourceReferenceLap = referenceLap;
            this.referenceBaseline = copyPhraseReference(referenceLap);
            this.possibleBrakingDeceleration = referenceBrakingDeceleration(this.referenceBaseline);
            this.carPerformance = referenceCarPerformance(this.referenceBaseline);
            this.phrases.filter((phrase) => phrase.referenceCue).forEach((phrase) => this.memory.delete(phrase.id));
            this.referenceMovement.reset();
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
        if (referenceChanged) {
            this.referenceLap = copyPhraseReference(this.referenceBaseline, this.drivingAggression, this.mapContext.cornerBounds);
            this.consecutiveCornerPositioning = createConsecutiveCornerPositioning(this.map, this.referenceLap);
            this.memory.forEach((_, id) => { if (id.startsWith(TURN_IN_MEMORY_PREFIX)) this.memory.delete(id); });
        }
        return this.evaluate(now);
    }

    setDrivingAggression(value: number, now: number): PhraseSnapshot {
        const aggression = normalizeDrivingAggression(value);
        if (aggression !== this.drivingAggression) {
            this.drivingAggression = aggression;
            // Always rebuild from the untouched local baseline, never from an already adjusted speed profile.
            this.referenceLap = copyPhraseReference(this.referenceBaseline, aggression, this.mapContext.cornerBounds);
            this.consecutiveCornerPositioning = createConsecutiveCornerPositioning(this.map, this.referenceLap);
            this.memory.clear();
            this.state = this.createState();
            this.events = [];
            this.referenceMovement.reset();
        }
        return this.evaluate(now);
    }

    setTrackGuideEnabled(enabled: boolean, now: number): PhraseSnapshot {
        if (enabled !== this.trackGuideEnabled) {
            this.trackGuideEnabled = enabled;
            if (this.state.path.some((node) => node.metadata.id === TRACK_GUIDE_ID)) this.state.exitToRoot();
            this.phrases.filter((phrase) => phrase.referenceCue).forEach((phrase) => this.memory.delete(phrase.id));
            const guideIds = new Set(this.phrases.filter((phrase) => phrase.referenceCue).map((phrase) => phrase.id));
            this.memory.forEach((_, id) => { if (id.startsWith(TURN_IN_MEMORY_PREFIX)) this.memory.delete(id); });
            this.events = this.events.map((event) => guideIds.has(event.ruleId) && event.speech && event.speech.releaseAt === undefined
                ? { ...event, speech: { ...event.speech, releaseAt: now } } : event);
            this.referenceMovement.reset();
            this.deceleration = this.currentDeceleration = undefined;
            this.decelerationSample = undefined;
            this.brakeReleasing = this.currentBrakeReleasing = false;
        }
        return this.evaluate(now);
    }

    reset(now: number) {
        this.sample = {};
        this.lookahead.reset();
        this.receivedAt = -Infinity;
        this.deceleration = undefined;
        this.currentDeceleration = undefined;
        this.currentBrakeReleasing = false;
        this.brakeReleasing = false;
        this.decelerationSample = undefined;
        this.vision = null;
        this.visionAfter = now;
        this.opponentDistanceM = undefined;
        this.closingOnOpponent = undefined;
        this.memory.clear();
        this.state = this.createState();
        this.events = [];
        this.sectionId = undefined;
        this.game = undefined;
        this.opponentMotion.reset();
        this.cornerSummaries = [];
        this.referenceMovement.reset();
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
            const brakeActive = this.trackGuideEnabled;
            this.currentDeceleration = continuous && brakeActive ? (previousSpeed! - speed!) / 3.6 / (elapsed / 1000) : undefined;
            const previousBrake = this.sample.Physics_brake, currentBrake = event.sample.Physics_brake;
            this.currentBrakeReleasing = continuous && brakeActive && previousBrake !== undefined && currentBrake !== undefined
                && Number.isFinite(previousBrake) && previousBrake >= 0 && previousBrake <= 1
                && Number.isFinite(currentBrake) && currentBrake >= 0 && currentBrake <= 1
                && (previousBrake - currentBrake) / (elapsed / 1000) > BRAKE_RELEASE_RATE_THRESHOLD;
            if (!continuous) {
                const moreBrake = this.memory.get('reference-moreBrake');
                if (moreBrake) moreBrake.since = undefined;
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
        if (!sameCalibration) {
            this.referenceMovement.reset();
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
        // Anticipate reference conditions except More brake, which uses measured motion.
        // Freshness, capture validation and closure clocks still use measured telemetry and real time.
        const predictionFrames = telemetryReady ? this.lookahead.getFrames(this.sample, this.referenceLap) : [];
        const preparationFrame = telemetryReady ? this.lookahead.getFrame(this.sample, this.referenceLap, this.root.preparationTimeMs) : undefined;
        const intendedTelemetry = preparationFrame?.intendedTelemetry ?? {};
        const intendedDeceleration = preparationFrame?.deceleration;
        const intendedBrakeReleasing = preparationFrame?.brakeReleasing;
        const referencePosition = intendedTelemetry.Graphics_normalized_car_position;
        const referenceMapContext = telemetryReady && sameMap ? this.mapContext.at(referencePosition) : {};
        const steeringCorner = telemetryReady && sameMap
            ? referenceMapContext.cornerGeometry ? referenceMapContext : this.mapContext.approaching(referencePosition)
            : undefined;
        const steeringBounds = steeringCorner?.cornerStartPosition !== undefined && steeringCorner.cornerEndPosition !== undefined
            ? { start: steeringCorner.cornerStartPosition, end: steeringCorner.cornerEndPosition } : undefined;
        const activeMeasurements = () => ({
            brake: this.trackGuideEnabled, throttle: this.trackGuideEnabled, steering: this.trackGuideEnabled,
        });
        const readGuidance = () => readReferenceGuidance(telemetryReady ? this.sample : {}, positionVision, this.referenceLap, this.game, now,
            this.sourceReferenceLap, intendedDeceleration ?? (telemetryReady ? this.deceleration : undefined),
            steeringCorner?.cornerGeometry?.turnSign, steeringBounds, intendedBrakeReleasing ?? this.brakeReleasing,
            referencePosition, intendedTelemetry,
            { deceleration: telemetryReady ? this.currentDeceleration : undefined, releasing: this.currentBrakeReleasing },
            activeMeasurements(), this.possibleBrakingDeceleration, this.carPerformance);
        let guidance = readGuidance();
        // Rearm on measured range completion even while a different guide owns the tree.
        const exitPosition = guidance.currentReferencePosition;
        this.memory.forEach((memory, id) => {
            if (memory.referenceExit && exitPosition) {
                const previous = memory.referenceExit;
                const travelled = ((exitPosition.distance - previous.distance + exitPosition.length * 1.5) % exitPosition.length)
                    - exitPosition.length / 2;
                previous.remaining -= travelled;
                previous.distance = exitPosition.distance;
                if (previous.remaining <= 1e-9) {
                    if (id.startsWith(TURN_IN_MEMORY_PREFIX)) this.memory.delete(id);
                    memory.referenceRange = undefined;
                    memory.referenceExit = undefined;
                    memory.lastEmitted = undefined;
                    memory.armed = true;
                }
            }
            if (this.phrases.find((phrase) => phrase.id === id)?.referenceCue === 'brake') {
                if (guidance.currentPedalZone === 'brake') memory.enteredBrakeZone = true;
                else if (guidance.currentPedalZone !== undefined && memory.enteredBrakeZone) {
                    memory.armed = true;
                    memory.enteredBrakeZone = false;
                }
            }
        });
        const driverCornerIds = telemetryReady && sameMap
            ? this.mapContext.cornersAt(this.sample.Graphics_normalized_car_position) : undefined;
        const driverOnStraight = telemetryReady && sameMap
            ? this.mapContext.isStraightAt(referencePosition) : undefined;
        inputs.driverOnStraight = driverOnStraight === undefined ? undefined : driverOnStraight ? 1 : 0;
        inputs.referenceBraking = guidance.braking === undefined ? undefined : guidance.braking ? 1 : 0;
        inputs.referenceBrakeRequired = guidance.brakeRequired === undefined ? undefined : guidance.brakeRequired ? 1 : 0;
        inputs.referencePedalZone = guidance.pedalZone;
        inputs.referenceCurrentPedalZone = guidance.currentPedalZone;
        inputs.referenceSteeringPhase = guidance.steeringPhase ?? (guidance.ready ? 'none' : undefined);
        inputs.referenceCornerEntry = steeringBounds && guidance.ready ? steeringCorner?.phase === 'entry' && guidance.turnInCorner ? 1 : 0 : undefined;
        inputs.referenceCornerPhase = guidance.cornerPhase;
        inputs.referenceCornerInSequence = steeringBounds ? steeringCorner?.sequenceId === undefined ? 0 : 1 : undefined;
        inputs.referenceSequenceRemaining = steeringCorner?.sequenceRemaining;
        inputs.referenceNextCornerSetup = this.consecutiveCornerPositioning(steeringCorner);
        inputs.referenceSteeringDirection = guidance.steeringDirection;
        // Per-cue entry holds use each direction instruction's own audio forecast.
        inputs.referenceBrakeReleaseRate = guidance.brakeReleaseRate;
        const referenceMovementAllowed = this.trackGuideEnabled && this.drivingAggression === 100
            && driverOnStraight === true && guidance.braking === false && guidance.lateralOffset !== undefined;
        if (!this.trackGuideEnabled || this.drivingAggression !== 100 || guidance.lateralOffset === undefined) this.referenceMovement.reset();
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
        // Keep motion history available before any cue becomes eligible.
        if (this.trackGuideEnabled) {
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
                }
            }
        }
        const context: PhraseContext = {
            entryPriorities: new Map(),
            scheduleCandidates: new Map(),
            now, emit, telemetryReady, intendedTelemetry, predictionFrames, inputs, guidance, memory: this.memory,
            trackGuideEnabled: this.trackGuideEnabled,
            speechPending: (actionId) => this.speechPending(actionId),
            ...(this.speechDurations ? { speechLeadMs: (actionId: string) => this.speechLeadMs(actionId),
                speechDurationMs: (instance: PhraseInstance) => this.speechDurations?.get(instance.cacheKey) } : {}),
            referenceMovementAllowed,
            movementCue: () => {
                const cue = this.referenceMovement.update(this.trackGuideEnabled && this.drivingAggression === 100 ? guidance.lateralOffset : undefined, positionVision?.capturedAt);
                if (cue) guidance[cue] = now + 1500;
                return cue;
            },
            movementCanSpeak: () => this.referenceMovement.canSpeak(now),
            driverLongStraightIds: telemetryReady && sameMap
                ? this.mapContext.longStraightsAt(this.sample.Graphics_normalized_car_position) : undefined,
            driverCornerIds,
            entryConditions: new Map(), actionConditions: new Map(), inspections: new Map(),
            publish: (phrase, instance) => {
                const phraseGuidance = context.forPhrase?.(phrase, instance.id).guidance ?? guidance;
                const sentence = instance.speech.text;
                const brakeRelease = phrase.referenceCue === 'easeBrake';
                const durationMs = this.speechDurations?.get(instance.cacheKey);
                const reactionAllowanceMs = this.speechActions.get(instance.id)?.reactionAllowanceMs ?? PHRASE_REACTION_ALLOWANCE_MS;
                const leadMs = this.speechLeadMs(instance.id);
                const referenceDeadline = phrase.referenceCue ? phraseGuidance[phrase.referenceCue] ?? guidance[phrase.referenceCue] : undefined;
                const timedCorner = this.speechDurations && this.speechActions.get(instance.id)?.conditions?.some((condition) =>
                    condition instanceof PhraseCondition && condition.input === 'opponentCornerEtaS');
                const expiresAt = timedCorner && inputs.opponentCornerEtaS !== undefined ? now + inputs.opponentCornerEtaS * 1000
                    : brakeRelease ? now + Math.max(4000, leadMs)
                    : referenceDeadline !== undefined ? Math.max(referenceDeadline, now + leadMs) : undefined;
                this.events = [...this.events, { id: ++this.nextId, ruleId: phrase.id, sentence, instance, timestamp: now,
                    ...(this.speechDurations ? { speech: { durationMs, preparationTimeMs: leadMs, reactionAllowanceMs } } : {}),
                    ...(phrase.referenceCue === 'steering' ? { steeringPhase: phraseGuidance.steeringPhase } : {}),
                    ...(brakeRelease ? { brakeRelease: true } : {}),
                    ...(expiresAt !== undefined ? { expiresAt } : {}) }].slice(-HISTORY_LIMIT);
                if (phrase.movementDirection) this.referenceMovement.spoken(phrase.referenceCue as ReferenceMovementCue, now);
                const memory = this.memory.get(phrase.id)!;
                memory.lastEmitted = now;
                memory.armed = false;
                if (phrase.referenceCue === 'brake' && phraseGuidance.currentPedalZone === 'brake') memory.enteredBrakeZone = true;
                if ((phrase.referenceCue === 'cornerExit' || phrase.referenceCue === 'steering' || phrase.referenceCue === 'easeBrake')
                    && phraseGuidance.steeringRange && phraseGuidance.currentReferencePosition) {
                    const { distance, length } = phraseGuidance.currentReferencePosition;
                    const end = isTurnIn(phrase) ? phraseGuidance.turnInCorner!.end : phraseGuidance.steeringRange.end;
                    memory.referenceExit = { distance,
                        remaining: (end - distance + length) % length };
                    if (isTurnIn(phrase)) {
                        // Keep the whole mapped corner consumed until measured exit, across all of its arcs.
                        this.memory.set(`${TURN_IN_MEMORY_PREFIX}${turnInCornerKey(phraseGuidance)}`,
                            { armed: false, referenceExit: { ...memory.referenceExit } });
                    }
                }
            },
        };
        const forecasts = new Map<string, PhraseContext>();
        const activePath = this.state.path;
        const forecastAt = (timeMs: number): PhraseContext => {
            const horizon = Math.max(0, Math.min(REFERENCE_LOOKAHEAD_MS, timeMs));
            const measurements = activeMeasurements();
            const key = `${horizon}:${measurements.brake}:${measurements.throttle}:${measurements.steering}`;
            const cached = forecasts.get(key);
            if (cached) return cached;
            const frame = telemetryReady ? this.lookahead.getFrame(this.sample, this.referenceLap, horizon) : undefined;
            const intended = frame?.intendedTelemetry ?? {};
            const position = intended.Graphics_normalized_car_position;
            const mapped = telemetryReady && sameMap ? this.mapContext.at(position) : {};
            const corner = mapped.cornerGeometry ? mapped : this.mapContext.approaching(position);
            const bounds = corner?.cornerStartPosition !== undefined && corner.cornerEndPosition !== undefined
                ? { start: corner.cornerStartPosition, end: corner.cornerEndPosition } : undefined;
            const forecast = readReferenceGuidance(telemetryReady ? this.sample : {}, positionVision, this.referenceLap, this.game, now,
                this.sourceReferenceLap, frame?.deceleration ?? this.deceleration,
                corner?.cornerGeometry?.turnSign, bounds, frame?.brakeReleasing ?? this.brakeReleasing,
                position, intended, { deceleration: this.currentDeceleration, releasing: this.currentBrakeReleasing },
                measurements, this.possibleBrakingDeceleration, this.carPerformance);
            const straight = telemetryReady && sameMap ? this.mapContext.isStraightAt(position) : undefined;
            const result: PhraseContext = { ...context, intendedTelemetry: intended, guidance: forecast, predictionOffsetMs: horizon,
                referenceMovementAllowed: this.trackGuideEnabled && this.drivingAggression === 100
                    && straight === true && forecast.braking === false && forecast.lateralOffset !== undefined,
                inputs: { ...inputs, referencePedalZone: forecast.pedalZone,
                    referenceBrakeRequired: forecast.brakeRequired === undefined ? undefined : forecast.brakeRequired ? 1 : 0,
                    referenceBraking: forecast.braking === undefined ? undefined : forecast.braking ? 1 : 0,
                    driverOnStraight: straight === undefined ? undefined : straight ? 1 : 0,
                    referenceSteeringPhase: forecast.steeringPhase ?? (forecast.ready ? 'none' : undefined),
                    referenceCornerEntry: bounds && forecast.ready ? corner?.phase === 'entry' && forecast.turnInCorner ? 1 : 0 : undefined,
                    referenceCornerPhase: forecast.cornerPhase,
                    referenceCornerInSequence: bounds ? corner?.sequenceId === undefined ? 0 : 1 : undefined,
                    referenceSequenceRemaining: corner?.sequenceRemaining,
                    referenceNextCornerSetup: this.consecutiveCornerPositioning(corner),
                    referenceSteeringDirection: forecast.steeringDirection,
                    referenceBrakeReleaseRate: forecast.brakeReleaseRate } };
            forecasts.set(key, result);
            return result;
        };
        // Search the bounded forecast, including now and the exact preparation boundary.
        // A short-lived match inside the window must not be skipped by its endpoint.
        const withinPreparation = (preparationTimeMs: number, matches: (forecast: PhraseContext) => boolean) => {
            const end = Math.max(0, Math.min(REFERENCE_LOOKAHEAD_MS, preparationTimeMs));
            const boundary = forecastAt(end);
            if (matches(boundary)) return boundary;
            for (const frame of predictionFrames) {
                if (frame.offsetMs >= end) break;
                const forecast = forecastAt(frame.offsetMs);
                if (matches(forecast)) return forecast;
            }
            // Retain the endpoint result for descriptions and recorded measurements.
            matches(boundary);
            return boundary;
        };
        context.forPhrase = (phrase, actionId) => {
            // More brake deliberately keeps its current measured-state hold.
            if (phrase.referenceCue === 'moreBrake') return forecastAt(0);
            // The first spoken action determines when its phrase enters. Once entered,
            // speech is immediately eligible; only explicit child conditions may defer it.
            const preparationTimeMs = this.speechDurations
                ? this.speechLeadMs(actionId ?? `${phrase.id}:say:0`) : phrase.preparationTimeMs ?? CLOSURE_PREPARATION_MS;
            // Newly admitted phrases keep their matching entry frame for the first action.
            if (!actionId && activePath.some((node) => node.metadata.id === phrase.id
                || (node instanceof TrackGuidePhrase && node.phrase.id === phrase.id))) return forecastAt(preparationTimeMs);
            const action = actionId ? this.speechActions.get(actionId) : undefined;
            return withinPreparation(preparationTimeMs, (forecast) => {
                const conditions = action?.conditions ?? phrase.conditions;
                return conditionsMatch(conditions.map((condition) => condition.evaluate({ ...forecast.inputs,
                    referenceCue: phrase.movementDirection ? forecast.referenceMovementAllowed && context.movementCue() === phrase.referenceCue
                        ? phrase.referenceCue : forecast.guidance.ready ? 'none' : undefined
                        : phrase.referenceCue && forecast.guidance[phrase.referenceCue] !== undefined ? phrase.referenceCue
                        : forecast.guidance.ready ? 'none' : undefined,
                })));
            });
        };
        context.forPreparation = (preparationTimeMs, matches) => matches
            ? withinPreparation(preparationTimeMs, matches) : forecastAt(preparationTimeMs);
        if (this.speechDurations) {
            const randomAudio = (action: ReturnType<typeof getPhraseActions>[number]) =>
                action.phrase.randomAlternatives.length > 1
                || (!action.instanceSettings && Boolean(action.audioVariations?.length));
            const deadlineAt = (phrase: PhraseDefinition, actionId: string): number | undefined => {
                const action = this.speechActions.get(actionId)!;
                const forecast = context.forPhrase!(phrase, actionId);
                const deadlines: number[] = [];
                const cueDeadline = phrase.referenceCue && forecast.guidance[phrase.referenceCue];
                if (typeof cueDeadline === 'number') deadlines.push(cueDeadline);
                if (action.conditions?.some((condition) => condition instanceof PhraseCondition && condition.input === 'opponentCornerEtaS')
                    && inputs.opponentCornerEtaS !== undefined) deadlines.push(now + inputs.opponentCornerEtaS * 1000);
                // Exit conditions belong to the phrase. Predict them only to size its random audio pool.
                const exits = [phrase.exitConditions, action.exitConditions].filter(
                    (conditions): conditions is readonly PhraseConditionNode[] => conditions !== undefined);
                for (const frame of exits.length ? predictionFrames : []) {
                    const predicted = forecastAt(frame.offsetMs);
                    if (exits.some((conditions) => conditionsMatch(conditions.map((condition) => condition.evaluate({
                        ...predicted.inputs, speed: finite(predicted.intendedTelemetry.Physics_speed_kmh) ?? predicted.inputs.speed,
                    }))))) {
                        deadlines.push(now + frame.offsetMs);
                        break;
                    }
                }
                return deadlines.length ? Math.min(...deadlines) : undefined;
            };
            context.scheduleAudio = (phrase, actionId) => {
                const action = this.speechActions.get(actionId)!;
                return randomAudio(action) ? { audioVariants: this.randomAudioPools.get(actionId) ?? [], deadlineAt: deadlineAt(phrase, actionId),
                    reactionAllowanceMs: action.reactionAllowanceMs } : {};
            };
            context.audioVariants = (phrase, actionId, instances) => {
                const action = this.speechActions.get(actionId)!;
                const variants = instances ? instances.flatMap((instance) => {
                    const durationMs = this.speechDurations!.get(instance.cacheKey);
                    return durationMs === undefined ? [] : [{ instance, durationMs }];
                }) : (randomAudio(action) ? this.randomAudioPools : this.preparedAudio).get(actionId) ?? [];
                if (!randomAudio(action)) return variants;
                const candidate: PhraseScheduleCandidate = {
                    id: actionId, kind: 'phrase', label: action.sentence, priority: action.priority,
                    category: getPhraseCategory(phrase), durationMs: this.speechLeadMs(actionId),
                    audioVariants: variants, deadlineAt: deadlineAt(phrase, actionId), reactionAllowanceMs: action.reactionAllowanceMs,
                };
                const currentId = this.state.current instanceof Action ? this.state.current.metadata.id : undefined;
                const playing = this.events.slice().reverse().find((event) => event.instance?.id === currentId
                    && event.speech && event.speech.releaseAt === undefined);
                const active: PhraseScheduleItem | undefined = playing ? {
                    id: playing.instance!.id, kind: 'phrase', label: playing.sentence, priority: playing.instance?.priority ?? 0,
                    category: 'other', status: 'active', startAt: playing.speech!.startedAt ?? playing.timestamp,
                    endAt: (playing.speech!.startedAt ?? playing.timestamp) + (playing.speech!.durationMs ?? 0),
                } : undefined;
                const candidates = new Map(context.scheduleCandidates);
                candidates.set(actionId, candidate);
                return schedulePhrases(now, Array.from(candidates.values()), active).items.find(({ id }) => id === actionId)?.audioVariants ?? [];
            };
        }
        const movementPhrases = this.phrases.filter((phrase) => phrase.movementDirection);
        const movementAvailable = movementPhrases.some((phrase) => {
            const lead = this.speechDurations ? this.speechLeadMs(`${phrase.id}:say:0`)
                : phrase.preparationTimeMs ?? CLOSURE_PREPARATION_MS;
            return withinPreparation(lead, (forecast) => forecast.referenceMovementAllowed === true).referenceMovementAllowed;
        });
        if (!movementAvailable) {
            this.referenceMovement.reset();
            movementPhrases.forEach((phrase) => {
                const memory = this.memory.get(phrase.id);
                if (memory) { memory.armed = true; memory.since = undefined; }
            });
        }
        const currentAction = this.state.current;
        if (currentAction instanceof Action && this.speechDurations && currentAction.metadata.id && this.speechActions.has(currentAction.metadata.id)) {
            const action = this.speechActions.get(currentAction.metadata.id);
            const exits = [action?.exitConditions,
                currentAction instanceof TrackGuidePhrase ? currentAction.phrase.exitConditions : undefined];
            if (exits.some((conditions) => conditions && conditionsMatch(conditions.map((condition) => condition.evaluate(inputs))))) {
                this.events = this.events.map((event) => event.instance?.id === currentAction.metadata.id && event.speech && event.speech.releaseAt === undefined
                    ? { ...event, speech: { ...event.speech, releaseAt: now } } : event);
            }
            if (!this.speechPending(currentAction.metadata.id) && !(currentAction instanceof TrackGuidePhrase)) this.state.leaveAction();
        }
        // Evaluation descends only through entered parents. Exit actions may return
        // through several levels before the next eligible phrase is selected.
        for (let transitions = 0; transitions <= this.phraseClosures.size + 1; transitions++) {
            const previous = this.state.current;
            const previousClosure = this.state.closure;
            const eventCount = this.nextId;
            this.step(context);
            if (this.nextId !== eventCount || this.state.current === previous || this.state.current instanceof Action) break;
            if (previousClosure === this.root
                || (this.state.current !== this.root && previousClosure.metadata.id !== TRACK_GUIDE_ID
                    && !previousClosure.metadata.id?.startsWith('reference-'))) break;
        }
        const root = this.state.snapshot(context);
        const nodes = new Map<string, NodeSnapshot>();
        const collect = (node: NodeSnapshot) => {
            nodes.set(node.id, node);
            node.children.forEach(collect);
        };
        collect(root);
        this.speechActions.forEach((action) => {
            const node = nodes.get(action.id);
            if (!node) return;
            const event = this.events.slice().reverse().find((item) => item.instance?.id === action.id);
            node.exitConditions = [{ description: 'The selected audio finishes',
                conditionFit: event?.speech ? event.speech.releaseAt !== undefined : null, inputMissing: false }];
            if (action.exitConditions) {
                const conditions = action.exitConditions.map((condition) => condition.evaluate(inputs));
                node.exitConditions = [...node.exitConditions, { description: 'Phrase exit condition', connector: 'or',
                    conditionFit: conditionsMatch(conditions), inputMissing: missingConditionInputs(conditions).length > 0,
                    conditions: snapshotConditions(conditions) }];
            }
            if (node.current && this.speechPending(action.id)) {
                node.status = 'Active';
                node.execution = { status: 'running' };
            }
        });
        // Track guide checks its cues before entry as well as while another cue speaks.
        // Publish those measurements even when no cue is ready to enter the group.
        this.phrases.filter((phrase) => phrase.referenceCue).forEach((phrase) => {
            getPhraseActions(phrase).forEach((action) => {
                const node = nodes.get(action.id);
                if (!node || !context.entryConditions.has(phrase.id)) return;
                node.conditions = trackPhraseConditions(phrase, action, context);
                if (!node.onPath) node.status = context.inspections.get(phrase.id)?.status ?? node.status;
            });
        });
        const closures = this.phrases.flatMap((phrase) => {
            const node = nodes.get(phrase.referenceCue ? `${phrase.id}:say:0` : phrase.id);
            if (!node) return [];
            const conditions = node.conditions.some((condition) => condition.conditionFit !== null)
                ? context.entryConditions.get(phrase.id) ?? unevaluatedConditions(phrase.conditions) : unevaluatedConditions(phrase.conditions);
            const actionConditions = context.actionConditions.get(`${phrase.id}:say:0`);
            const missing = conditionsMatch(conditions) ? [] : missingConditionInputs(conditions);
            const status = node.status as ClosureStatus;
            return [{
                id: phrase.id, name: phrase.name, description: phrase.description, holdMs: phrase.holdMs,
                status, missing, conditions, actionConditions,
                actions: (node.kind === 'phrase' ? [node] : node.children.filter((child) => child.kind !== 'closure'))
                    .map(({ name, description }) => ({ name, description })),
            }];
        });
        const activePhrase = this.phrases.find((phrase) => this.state.path.some((node) => node.metadata.id === phrase.id
            || (node instanceof TrackGuidePhrase && node.phrase.id === phrase.id)));
        const activeGuidance = activePhrase ? context.forPhrase(activePhrase,
            this.state.current instanceof Action ? this.state.current.metadata.id : undefined).guidance : guidance;
        const activeAction = this.state.current instanceof Action && this.state.current.metadata.kind === 'phrase'
            ? this.state.current : undefined;
        const activeEvent = activeAction ? this.events.slice().reverse().find((event) => event.instance?.id === activeAction.metadata.id
            && event.speech && event.speech.releaseAt === undefined) : undefined;
        const activeStart = activeEvent?.speech?.startedAt ?? activeEvent?.timestamp ?? now;
        const activeDefinition = activeEvent ? this.phrases.find((phrase) => phrase.id === activeEvent.ruleId) : undefined;
        const active: PhraseScheduleItem | undefined = activeAction && activeEvent ? {
            id: activeAction.metadata.id!, kind: 'phrase', label: activeEvent.sentence,
            priority: activeAction.priority, category: activeDefinition ? getPhraseCategory(activeDefinition) : 'other',
            status: 'active', startAt: activeStart,
            endAt: Math.max(now, activeStart + (activeEvent.speech?.durationMs ?? 0)),
            durationEstimated: activeEvent?.speech?.durationMs === undefined || activeEvent.speech.startedAt === undefined,
        } : undefined;
        context.scheduleCandidates!.forEach((_candidate, id) => {
            const phrase = nodes.get(id);
            if (phrase?.execution?.status !== 'idle' || phrase.status === 'Not evaluated') context.scheduleCandidates!.delete(id);
        });
        return {
            schedule: schedulePhrases(now, Array.from(context.scheduleCandidates!.values()), active),
            telemetryReady, visionReady, mapReady: sameMap && this.mapContext.ready, mapContext, closures, events: this.events,
            referenceLoaded: this.referenceLap !== null, trackGuideEnabled: this.trackGuideEnabled,
            referenceSteeringPhase: activeGuidance.steeringPhase,
            referenceSteeringDirection: guidance.steeringDirection,
            referenceBrakeReleaseRate: activeGuidance.brakeReleaseRate,
            referenceBrakeReleaseDurationMs: activeGuidance.brakeReleaseDurationMs,
            steeringCorner: this.trackGuideEnabled && guidance.steering !== undefined ? steeringCorner : undefined,
            root,
            state: { current: this.state.current.name, description: this.state.current.description,
                kind: this.state.current instanceof Closure ? 'closure' : this.state.current.metadata.kind ?? 'action', path: this.state.path.map((node) => node.name) },
        };
    }
}
