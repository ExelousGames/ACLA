import { ACC_STATUS } from 'data/live-analysis/live-map-data';
import { DrivingImpactMonitor } from './driving-impact-monitor';
import { drivingReward } from './driving-observation';
import { trainingFixture } from './training-fixtures';

function setup(game = 'acc', kph = 80) {
    const fixture = trainingFixture();
    const session = JSON.stringify([fixture.live.sessionGeneration, fixture.live.streamGeneration, game, 'GT3', 'monza']);
    const monitor = new DrivingImpactMonitor();
    const observation = (at: number) => ({ ...fixture.observation, session, capturedAt: at, measuredAt: at });
    const frame = (at: number, forward = kph / 3.6, right = 0, brake = 0) => ({ ...fixture.live,
        game: game as 'acc' | 'iracing', receivedAt: at, currentTelemetry: { ...fixture.live.currentTelemetry,
            Physics_local_velocity_x: right, Physics_local_velocity_y: game === 'iracing' ? 0 : forward,
            Physics_local_velocity_z: game === 'iracing' ? forward : 0, Physics_brake: brake } });
    return { monitor, observation, frame };
}

it.each(['acc', 'iracing'])('retains a brief bounce between slow AI frames for %s at 20 and 80 km/h', (game) => {
    for (const kph of [20, 80]) {
        const { monitor, observation, frame } = setup(game, kph);
        const speed = kph / 3.6;
        monitor.observe(frame(1000));
        monitor.observe(frame(1100, speed * 0.6, speed * 0.3));
        // The bounce settles before the next AI image; endpoints alone miss it.
        for (const at of [1200, 1400, 1600, 1800, 2000]) monitor.observe(frame(at));
        const impact = monitor.between(observation(1000), observation(1800))!;
        expect(impact).toMatchObject({ available: true, detected: true });
        expect(impact.severity).toBeGreaterThan(0);
        const reward = drivingReward(observation(1000), observation(1800), undefined, impact);
        expect(reward.impact).toBeLessThan(0);
        expect(reward.total).toBeCloseTo(reward.timing + reward.line + reward.approach + reward.progress + reward.impact);
        expect(monitor.between(observation(1800), observation(2000))).toMatchObject({ available: true, detected: false });
    }
});

it('excludes collisions newer than the image and does not consume evidence when queried', () => {
    const { monitor, observation, frame } = setup();
    for (const at of [1000, 1200, 1400]) monitor.observe(frame(at));
    monitor.observe(frame(1500, 0));
    expect(monitor.between(observation(1000), observation(1400))).toMatchObject({ available: true, detected: false });
    const impact = monitor.between(observation(1400), observation(1500));
    expect(impact?.detected).toBe(true);
    expect(monitor.between(observation(1400), observation(1500))).toEqual(impact);
});

it('selects the strongest impact instead of multiplying a bounce by the telemetry frame count', () => {
    const { monitor, observation, frame } = setup();
    monitor.observe(frame(1000, 30));
    monitor.observe(frame(1100, 27));
    monitor.observe(frame(1200, 0));
    monitor.observe(frame(1300, 0));
    const impact = monitor.between(observation(1000), observation(1300));
    expect(impact).toMatchObject({ detected: true, confidence: 1, severity: 1 });
    expect(drivingReward(observation(1000), observation(1300), undefined, impact).impact).toBe(-5);
});

it('accumulates an impact spread across fast samples instead of subtracting noise from every tiny change', () => {
    const { monitor, observation, frame } = setup('acc', 20);
    for (let step = 0; step <= 10; step++) monitor.observe(frame(1000 + step * 20, (20 / 3.6) * (1 - step / 10)));
    expect(monitor.between(observation(1000), observation(1200))?.detected).toBe(true);
});

it('uses braking between camera frames even when the pedal is released at both endpoints', () => {
    const { monitor, observation, frame } = setup('acc', 20);
    monitor.observe(frame(1000));
    for (let step = 1; step <= 5; step++) monitor.observe(frame(1000 + step * 100, (20 / 3.6) * (1 - step / 5), 0, 0.5));
    monitor.observe(frame(1600, 0));
    expect(monitor.between(observation(1000), observation(1600))).toMatchObject({ available: true, detected: false });
});

it('accumulates fast telemetry updates until the minimum detection interval', () => {
    const { monitor, observation, frame } = setup('acc', 20);
    monitor.observe(frame(1000));
    monitor.observe(frame(1010, 0));
    monitor.observe(frame(1020, 0));
    expect(monitor.between(observation(1000), observation(1020))?.detected).toBe(true);
    monitor.observe(frame(1020)); // Duplicate notifications do not generate another event.
    monitor.observe(frame(1040, 0));
    expect(monitor.between(observation(1020), observation(1040))?.detected).toBe(false);
});

it('reports a telemetry gap as unavailable even when later telemetry is healthy', () => {
    const { monitor, observation, frame } = setup();
    monitor.observe(frame(1000));
    monitor.observe(frame(1800, 0));
    monitor.observe(frame(1900, 0));
    expect(monitor.between(observation(1000), observation(1900))).toMatchObject({ available: false, detected: false });
});

it.each(['stream', 'session', 'car', 'track', 'paused', 'invalid', 'clock'])('discards old impact evidence after a %s boundary', (boundary) => {
    const { monitor, observation, frame } = setup();
    monitor.observe(frame(1000));
    monitor.observe(frame(1100, 0));
    const changed = frame(1200);
    if (boundary === 'stream') changed.streamGeneration++;
    if (boundary === 'session') changed.sessionGeneration++;
    if (boundary === 'car') changed.currentTelemetry.Static_car_model = 'other car';
    if (boundary === 'track') changed.currentTelemetry.Static_track = 'other track';
    if (boundary === 'paused') changed.telemetryStatus = ACC_STATUS.ACC_PAUSE;
    if (boundary === 'invalid') changed.currentTelemetry.Physics_local_velocity_x = NaN;
    if (boundary === 'clock') changed.receivedAt = 900;
    monitor.observe(changed);
    expect(monitor.between(observation(1000), observation(1200))).toBeUndefined();
});

it.each(['acc', 'iracing'])('ignores vertical bumps in %s telemetry', (game) => {
    const { monitor, observation, frame } = setup(game);
    monitor.observe(frame(1000));
    const bump = frame(1100);
    bump.currentTelemetry[game === 'iracing' ? 'Physics_local_velocity_y' : 'Physics_local_velocity_z'] = 30;
    monitor.observe(bump);
    expect(monitor.between(observation(1000), observation(1100))).toMatchObject({ available: true, detected: false });
});

it('expires old evidence when the bounded history fills', () => {
    const { monitor, observation, frame } = setup();
    monitor.observe(frame(1000));
    monitor.observe(frame(1020, 0));
    for (let i = 2; i <= 520; i++) monitor.observe(frame(1000 + i * 20, 0));
    expect(monitor.between(observation(1000), observation(1040))).toBeUndefined();
    expect(monitor.between(observation(11000), observation(11400))).toMatchObject({ available: true, detected: false });
});
