import type { ControllerAction } from './virtual-controller';

export type ControlChanges = Pick<ControllerAction, 'steering' | 'throttle' | 'brake'>;
export const PEDAL_CHANGE_REFERENCE_MS = 100;

class ControlMovement {
    private direction = 0;
    private start = 0;
    private cost = 0;

    record(from: number, to: number, elapsedMs: number, pedalScale?: number) {
        const direction = Math.sign(to - from);
        if (direction === 0) {
            if (elapsedMs > 0) this.direction = 0;
            return 0;
        }

        // Settle the preceding movement once, when a reversal or restart confirms an interruption.
        const interrupted = direction !== this.direction;
        const charge = interrupted ? this.cost : 0;
        if (interrupted) {
            this.start = from;
            this.cost = 0;
        }
        this.direction = direction;
        this.cost = pedalScale === undefined ? (to - this.start) ** 2 : this.cost + (to - from) ** 2 * pedalScale;
        return charge;
    }
}

export class DrivingControlChanges {
    private steering = new ControlMovement();
    private throttle = new ControlMovement();

    record(previous: ControlChanges, current: ControlChanges, elapsedMs: number): ControlChanges {
        // Commands can be acknowledged in the same millisecond.
        const pedalScale = PEDAL_CHANGE_REFERENCE_MS / (elapsedMs > 0 ? elapsedMs : 1);
        return {
            steering: this.steering.record(previous.steering, current.steering, elapsedMs),
            throttle: this.throttle.record(previous.throttle, current.throttle, elapsedMs, pedalScale),
            brake: (current.brake - previous.brake) ** 2 * pedalScale,
        };
    }
}
