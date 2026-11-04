export type MovementDirection = 'left' | 'right';
type MovementLevel = 'small' | 'medium' | 'more' | 'enough';
export type ReferenceMovementCue = `move-${MovementDirection}-${MovementLevel}`;

export const REFERENCE_MOVEMENTS = (['left', 'right'] as const).flatMap((direction) => ([
    { level: 'small', name: 'Small movement', sentence: `Move to the ${direction}.`,
        sentenceAlternatives: [`Shift to the ${direction}.`, `Bring it to the ${direction}.`],
        description: 'The reference is 3–7 source units away laterally.' },
    { level: 'medium', name: 'Medium movement', sentence: `Further to the ${direction}.`,
        sentenceAlternatives: [`More to the ${direction}.`, `Move further to the ${direction}.`],
        description: 'The reference is 7–13 source units away laterally.' },
    { level: 'more', name: 'A little bit more', sentence: 'A little bit more.',
        sentenceAlternatives: ['A bit more.', 'Just a little more.'],
        description: 'The driver is moving toward the reference, but would take more than 3 seconds to reach it at this rate.' },
    { level: 'enough', name: 'Enough movement', sentence: "OK, that's enough.",
        sentenceAlternatives: ["That's enough.", 'Good, hold that line.'],
        description: 'After a movement cue, the driver is within 0.35 source units of the reference or has crossed it.' },
] as const).map((movement) => ({ ...movement, direction, cue: `move-${direction}-${movement.level}` as ReferenceMovementCue })));

/** Track lateral progress using distinct, validated position-model captures only. */
export class ReferenceMovement {
    private capturedAt?: number;
    private anchor?: { capturedAt: number; distance: number };
    private closingRate?: number;
    private active?: { direction: MovementDirection; completed: boolean };
    private cue?: ReferenceMovementCue;
    private lastSpoken = -Infinity;

    reset() {
        this.capturedAt = undefined;
        this.anchor = undefined;
        this.closingRate = undefined;
        this.active = undefined;
        this.cue = undefined;
        this.lastSpoken = -Infinity;
    }

    update(lateral: number | undefined, capturedAt: number | undefined): ReferenceMovementCue | undefined {
        if (lateral === undefined || capturedAt === undefined) { this.reset(); return; }
        if (this.capturedAt !== undefined && capturedAt <= this.capturedAt) return this.cue;
        if (this.capturedAt !== undefined && capturedAt - this.capturedAt > 500) this.reset();
        this.capturedAt = capturedAt;
        const distance = Math.abs(lateral);
        const direction = lateral < 0 ? 'left' : 'right';
        this.anchor ??= { capturedAt, distance };
        if (capturedAt - this.anchor.capturedAt >= 750) {
            this.closingRate = (this.anchor.distance - distance) / ((capturedAt - this.anchor.capturedAt) / 1000);
            this.anchor = { capturedAt, distance };
        }
        if (this.active?.completed && distance >= 3) this.active = undefined;
        if (this.active && (distance <= 0.35 || direction !== this.active.direction || this.active.completed)) {
            this.cue = `move-${this.active.direction}-enough`;
        } else if (this.active && this.closingRate !== undefined && this.closingRate > 0.05) {
            this.cue = (distance - 0.35) / this.closingRate > 3 ? `move-${direction}-more` : undefined;
        } else {
            this.cue = distance < 3 ? undefined : `move-${direction}-${distance < 7 ? 'small' : 'medium'}`;
        }
        return this.cue;
    }

    canSpeak(now: number) {
        return this.cue?.endsWith('-enough') || now - this.lastSpoken >= 2000;
    }

    spoken(cue: ReferenceMovementCue, now: number) {
        const direction = cue.startsWith('move-left-') ? 'left' : 'right';
        this.active = { direction, completed: cue.endsWith('-enough') };
        this.lastSpoken = now;
    }
}
