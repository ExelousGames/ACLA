import { DrivingControlOutput } from './driving-control-output';
import { NEUTRAL } from './driving-observation';

const action = { throttle: 0.8, brake: 0, steering: 0.3, shiftUp: true };
const setup = () => {
    const controller = { connect: jest.fn(), disconnect: jest.fn(), apply: jest.fn().mockResolvedValue(undefined) };
    return { controller, output: new DrivingControlOutput(controller) };
};

it('serializes both drivers and releases outgoing paddles before the new action', async () => {
    const { controller, output } = setup();
    const ai = output.acquire('ai');
    await ai.apply(action);
    const human = output.acquire('manual');
    expect(await ai.apply(action)).toBe(false);
    const manualAction = { throttle: 0, brake: 0.7, steering: -0.4 };
    expect(await human.apply(manualAction)).toBe(true);
    expect(controller.apply.mock.calls.map(([value]) => value)).toEqual([action, NEUTRAL, manualAction]);
});

it.each(['manual', 'test'] as const)('discards queued AI commands when %s takes ownership during a write', async (owner) => {
    const { controller, output } = setup();
    let acknowledge!: () => void;
    controller.apply.mockImplementationOnce(() => new Promise<void>((resolve) => { acknowledge = resolve; }));
    const ai = output.acquire('ai');
    const pending = ai.apply(action);
    await Promise.resolve();
    const queued = ai.apply({ ...action, throttle: 1 });
    const human = output.acquire(owner);
    const replacement = human.apply({ ...NEUTRAL, brake: 1 });
    expect(controller.apply).toHaveBeenCalledTimes(1);
    acknowledge();
    expect(await pending).toBe(false);
    expect(await queued).toBe(false);
    expect(await replacement).toBe(true);
    expect(controller.apply.mock.calls.map(([value]) => value)).toEqual([action, NEUTRAL, { ...NEUTRAL, brake: 1 }]);
});

it('makes neutral the final write on Stop even with delayed acknowledgement and queued commands', async () => {
    const { controller, output } = setup();
    let acknowledge!: () => void;
    controller.apply.mockImplementationOnce(() => new Promise<void>((resolve) => { acknowledge = resolve; }));
    const driver = output.acquire('manual');
    const pending = driver.apply(action);
    await Promise.resolve();
    const queued = driver.apply(action);
    const stopped = output.release();
    acknowledge();
    expect(await pending).toBe(false);
    expect(await queued).toBe(false);
    await stopped;
    expect(await driver.apply(action)).toBe(false);
    expect(controller.apply.mock.calls.map(([value]) => value)).toEqual([action, NEUTRAL]);
});

it('revokes an earlier lease even when the same driver starts again', async () => {
    const { controller, output } = setup();
    const old = output.acquire('ai');
    const current = output.acquire('ai');
    expect(await old.apply(action)).toBe(false);
    await current.apply(NEUTRAL);
    expect(controller.apply).toHaveBeenCalledTimes(1);
});

it('still releases after a rejected write and never sends a handoff action if neutral fails', async () => {
    const { controller, output } = setup();
    await output.acquire('ai').apply(action);
    controller.apply.mockRejectedValueOnce(new Error('Device failure'));
    await expect(output.acquire('manual').apply(action)).rejects.toThrow('Device failure');
    await output.release();
    expect(controller.apply.mock.calls.map(([value]) => value)).toEqual([action, NEUTRAL, NEUTRAL]);
});
