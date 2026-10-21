import { DrivingAction, DrivingObservation, DrivingReward, drivingReward, NEUTRAL, ObservationResult } from './driving-observation';
import { DrivingPolicy, PolicyDecision } from './driving-policy';
import type { VirtualController } from './virtual-controller';

export interface TrainingState {
    mode: 'idle' | 'training' | 'evaluation';
    status: string;
    action: DrivingAction;
    steps: number;
    episodeReward: number;
    reward: DrivingReward | null;
}
export const initialTrainingState: TrainingState = {
    mode: 'idle', status: 'Connect the controller and prepare Live Session inputs.', action: NEUTRAL,
    steps: 0, episodeReward: 0, reward: null,
};

export class DrivingTrainingLoop {
    state = { ...initialTrainingState };
    private previous?: { observation: DrivingObservation; decision: PolicyDecision; appliedAt: number; previousAction: DrivingAction };
    private generation = 0;
    private busy = false;
    private timer?: ReturnType<typeof setInterval>;

    constructor(private controller: VirtualController, private read: () => ObservationResult,
        private publish: (state: TrainingState) => void, private now = Date.now) {}

    start(policy: DrivingPolicy, train: boolean) {
        if (this.state.mode !== 'idle' || this.busy) throw new Error('The controller is still processing the previous command.');
        const input = this.read();
        if (!input.observation) throw new Error(input.error);
        if (!policy.matches(input.observation)) throw new Error('Model inputs differ from this game, car, map or segmentation model. Create a new model.');
        this.previous = undefined;
        this.generation++;
        this.state = { ...initialTrainingState, mode: train ? 'training' : 'evaluation', steps: policy.steps,
            status: train ? 'Training • online actor–critic' : 'Driving • fixed policy' };
        this.publish(this.state);
        this.timer = setInterval(() => { void this.tick(policy); }, 100);
        void this.tick(policy);
    }

    async stop(status = 'Stopped. Controller released.'): Promise<void> {
        this.generation++;
        clearInterval(this.timer);
        this.previous = undefined;
        this.state = { ...this.state, mode: 'idle', status, action: NEUTRAL };
        this.publish(this.state);
        try { await this.controller.apply(NEUTRAL); }
        catch (error) {
            this.state = { ...this.state, status: `${status} ${error instanceof Error ? error.message : 'Controller disconnected.'}` };
            this.publish(this.state);
        }
    }

    private async tick(policy: DrivingPolicy) {
        if (this.busy || this.state.mode === 'idle') return;
        this.busy = true;
        const generation = this.generation;
        try {
            const input = this.read();
            if (!input.observation) throw new Error(input.error);
            const observation = input.observation;
            if (!policy.matches(observation)) throw new Error('Game, car, map or segmentation model changed. Start a new episode.');
            const previous = this.previous;
            if (previous && observation.session !== previous.observation.session) throw new Error('Live session changed. Start a new episode.');
            // Refresh the hardware watchdog, but never learn twice from the same image or from a capture predating the action.
            if (previous && observation.capturedAt <= previous.appliedAt) {
                await this.controller.apply(this.state.action);
                return;
            }
            if (previous) {
                const reward = drivingReward(previous.observation, observation, previous.decision.action, previous.previousAction);
                const terminal = observation.referenceDistance > 25;
                if (terminal) reward.total = -10;
                if (this.state.mode === 'training') policy.learn(previous.decision, reward.total, observation.features,
                    (observation.capturedAt - previous.observation.capturedAt) / 1000, terminal);
                this.state = { ...this.state, steps: policy.steps, reward, episodeReward: this.state.episodeReward + reward.total };
                if (terminal) throw new Error('Car is too far from the reference line. Reposition it and start a new episode.');
            }
            if (observation.referenceDistance > 25) throw new Error('Position the car within 25 m of the reference line before starting.');
            const decision = policy.decide(observation.features, this.state.mode === 'training');
            await this.controller.apply(decision.action);
            if (generation !== this.generation) return;
            this.previous = { observation, decision, appliedAt: this.now(), previousAction: this.state.action };
            this.state = { ...this.state, action: decision.action };
            this.publish(this.state);
        } catch (error) {
            if (generation === this.generation) await this.stop(error instanceof Error ? error.message : 'Training failed.');
        } finally { this.busy = false; }
    }
}
