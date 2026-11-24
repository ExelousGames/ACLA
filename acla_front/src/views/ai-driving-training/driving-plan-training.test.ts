import { DrivingPlanTraining } from './driving-plan-training';
import { DrivingPolicy } from './driving-policy';
import { DrivingMemory } from './driving-memory';
import { trainingFixture } from './training-fixtures';

function setup() {
    const { observation } = trainingFixture();
    const policy = new DrivingPolicy(observation.context, observation.features.length);
    const learn = jest.spyOn(policy, 'learnPlannedActionsAsync').mockResolvedValue(undefined);
    return { policy, learn, training: new DrivingPlanTraining(), features: new DrivingMemory().encode(observation) };
}
const neutral = { pedal: 0, steering: 0 };
const shifts = { up: true, down: true };

it('pairs all 30 later 100 ms time slots with their observed action and original input snapshot', async () => {
    const { policy, learn, training, features } = setup();
    const original = { ...features, features: features.features.slice(), cameras: features.cameras.slice() };
    await training.recordAsync(policy, features, neutral, shifts, 1000);
    features.features[0] = 1;
    for (let step = 1; step <= 30; step++) {
        const action = { pedal: step / 30, steering: -step / 30, shiftUp: step === 3 };
        await training.recordAsync(policy, features, action, shifts, 1000 + step * 100);
        expect(learn).toHaveBeenCalledWith(original, [{ step, action, shifts }]);
    }
    expect(learn.mock.calls.filter(([input]) => input.features[0] === original.features[0])).toHaveLength(30);
    expect(learn.mock.calls[0][0].recurrent).toBe(original.recurrent);
});

it('waits for a time slot, tolerates small sampling offsets and skips missing targets', async () => {
    const { policy, learn, training, features } = setup();
    const original = { ...features, features: features.features.slice(), cameras: features.cameras.slice() };
    await training.recordAsync(policy, features, neutral, shifts, 1000);
    features.features[0] = 1; // Distinguish this initial plan from overlapping later plans.
    await training.recordAsync(policy, features, neutral, shifts, 1099);
    expect(learn).not.toHaveBeenCalled();
    const action = { pedal: -0.1, steering: -0.5 };
    await training.recordAsync(policy, features, action, shifts, 1110);
    expect(learn).toHaveBeenCalledWith(original, [{ step: 1, action, shifts }]);
    learn.mockClear();
    await training.recordAsync(policy, features, neutral, shifts, 1361);
    expect(learn).not.toHaveBeenCalled(); // No sample within 50 ms of a due slot.
    await training.recordAsync(policy, features, action, shifts, 1400);
    expect(learn).toHaveBeenCalledWith(original, [{ step: 4, action, shifts }]);
    expect(learn.mock.calls.filter(([input]) => input.features[0] === original.features[0])
        .some(([, targets]) => targets.some(({ step }) => step === 2 || step === 3))).toBe(false);
});

it.each([
    { edge: 'the scheduled time', origin: 481.1, frame: 481.2, boundaryMs: 0, direction: -1 },
    { edge: 'the late limit', origin: 481.2, frame: 481.35, boundaryMs: 50, direction: 1 },
])('keeps brake and steering labels when frame time rounds outside $edge', async ({ origin, frame, boundaryMs, direction }) => {
    const { policy, learn, training, features } = setup();
    const recordedAt = (seconds: number) => (0.236614 + seconds) * 1000;
    const originMs = recordedAt(origin), frameMs = recordedAt(frame);
    expect(Math.sign(frameMs - (originMs + 100) - boundaryMs)).toBe(direction);

    await training.recordAsync(policy, features, neutral, shifts, originMs);
    const action = { pedal: -0.55, steering: 0.4 };
    await training.recordAsync(policy, features, action, shifts, frameMs);
    expect(learn).toHaveBeenCalledWith(features, [{ step: 1, action, shifts }]);

    await training.recordAsync(policy, features, neutral, shifts, frameMs + 1);
    expect(learn).toHaveBeenCalledTimes(1);
});

it('discards pending time slots on reset', async () => {
    const { policy, learn, training, features } = setup();
    await training.recordAsync(policy, features, neutral, shifts, 1000);
    training.reset();
    await training.recordAsync(policy, features, neutral, shifts, 1100);
    expect(learn).not.toHaveBeenCalled();
});

it('counts submitted replays, excluding pending, cancelled and failed samples', async () => {
    const { policy, learn, training, features } = setup();
    const submitted = jest.fn();
    let active = true;
    const record = (at: number) => training.recordAsync(policy, features, neutral, shifts, at, () => active, submitted);
    await record(1000);
    await record(1050);
    expect(submitted).not.toHaveBeenCalled();
    learn.mockImplementationOnce(async () => { active = false; });
    await record(1150); // Both plans are due; only the first is submitted before cancellation.
    expect(learn).toHaveBeenCalledTimes(1);
    expect(submitted).toHaveBeenCalledTimes(1);
    active = true;
    learn.mockRejectedValueOnce(new Error('GPU lost'));
    await expect(record(1250)).rejects.toThrow('GPU lost');
    expect(submitted).toHaveBeenCalledTimes(1);
});
