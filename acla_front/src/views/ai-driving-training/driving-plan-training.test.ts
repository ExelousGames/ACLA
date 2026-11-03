import { DrivingPlanTraining } from './driving-plan-training';
import { DrivingPolicy } from './driving-policy';
import { NEUTRAL } from './driving-observation';
import { DrivingMemory } from './driving-memory';
import { trainingFixture } from './training-fixtures';

function setup() {
    const { observation } = trainingFixture();
    const policy = new DrivingPolicy(observation.context, observation.features.length);
    const learn = jest.spyOn(policy, 'learnPlannedActionsAsync').mockResolvedValue(undefined);
    return { policy, learn, training: new DrivingPlanTraining(), features: new DrivingMemory().encode(observation) };
}
const shifts = { up: true, down: true };

it('pairs all six later time slots with their observed action and original input snapshot', async () => {
    const { policy, learn, training, features } = setup();
    const original = { features: features.features.slice(), cameras: features.cameras.slice() };
    await training.recordAsync(policy, features, NEUTRAL, shifts, 1000);
    features.features[0] = 1;
    for (let step = 1; step <= 6; step++) {
        const action = { throttle: step / 6, brake: 0, steering: -step / 6, shiftUp: step === 3 };
        await training.recordAsync(policy, features, action, shifts, 1000 + step * 500);
        expect(learn).toHaveBeenCalledWith(original, [{ step, action, shifts }]);
    }
    expect(learn.mock.calls.filter(([input]) => input.features[0] === original.features[0])).toHaveLength(6);
});

it('waits for a time slot, tolerates small sampling offsets and skips missing targets', async () => {
    const { policy, learn, training, features } = setup();
    const original = { features: features.features.slice(), cameras: features.cameras.slice() };
    await training.recordAsync(policy, features, NEUTRAL, shifts, 1000);
    features.features[0] = 1; // Distinguish this initial plan from overlapping later plans.
    await training.recordAsync(policy, features, NEUTRAL, shifts, 1499);
    expect(learn).not.toHaveBeenCalled();
    const action = { throttle: 0.8, brake: 0.1, steering: -0.5 };
    await training.recordAsync(policy, features, action, shifts, 1600);
    expect(learn).toHaveBeenCalledWith(original, [{ step: 1, action, shifts }]);
    learn.mockClear();
    await training.recordAsync(policy, features, NEUTRAL, shifts, 2900);
    expect(learn).not.toHaveBeenCalled(); // No sample within 250 ms of a due slot.
    await training.recordAsync(policy, features, action, shifts, 3000);
    expect(learn).toHaveBeenCalledWith(original, [{ step: 4, action, shifts }]);
    expect(learn.mock.calls.filter(([input]) => input.features[0] === original.features[0])
        .some(([, targets]) => targets.some(({ step }) => step === 2 || step === 3))).toBe(false);
});

it('discards pending time slots on reset', async () => {
    const { policy, learn, training, features } = setup();
    await training.recordAsync(policy, features, NEUTRAL, shifts, 1000);
    training.reset();
    await training.recordAsync(policy, features, NEUTRAL, shifts, 1500);
    expect(learn).not.toHaveBeenCalled();
});

it('counts submitted replays, excluding pending, cancelled and failed samples', async () => {
    const { policy, learn, training, features } = setup();
    const submitted = jest.fn();
    let active = true;
    const record = (at: number) => training.recordAsync(policy, features, NEUTRAL, shifts, at, () => active, submitted);
    await record(1000);
    await record(1100);
    expect(submitted).not.toHaveBeenCalled();
    learn.mockImplementationOnce(async () => { active = false; });
    await record(1600); // Both plans are due; only the first is submitted before cancellation.
    expect(learn).toHaveBeenCalledTimes(1);
    expect(submitted).toHaveBeenCalledTimes(1);
    active = true;
    learn.mockRejectedValueOnce(new Error('GPU lost'));
    await expect(record(2100)).rejects.toThrow('GPU lost');
    expect(submitted).toHaveBeenCalledTimes(1);
});
