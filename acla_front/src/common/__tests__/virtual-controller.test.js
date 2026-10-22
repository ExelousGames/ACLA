const { EventEmitter } = require('events');
const { PassThrough, Writable } = require('stream');
const { registerVirtualController } = require('../../../electron/virtual-controller');

function setup(platform = 'win32') {
    const handlers = {};
    const app = new EventEmitter();
    app.getAppPath = () => '/app';
    app.getPath = jest.fn(() => '/redirected/documents');
    const installAccPreset = jest.fn().mockResolvedValue({ name: 'ACLA AI Training - vJoy 3', path: '/preset.json', deviceId: 3 });
    const contents = new EventEmitter();
    contents.mainFrame = {};
    contents.isDestroyed = () => false;
    contents.getBackgroundThrottling = () => true;
    contents.setBackgroundThrottling = jest.fn();
    const child = new EventEmitter();
    const writes = [];
    child.stdin = new Writable({ write(chunk, _encoding, callback) { writes.push(JSON.parse(chunk.toString())); callback(); } });
    child.stdout = new PassThrough(); child.stderr = new PassThrough(); child.kill = jest.fn();
    const spawnProcess = jest.fn(() => child);
    registerVirtualController({ app, ipcMain: { handle: (key, handler) => { handlers[key] = handler; } },
        getMainWindow: () => ({ isDestroyed: () => false, webContents: contents }), getPythonExecutable: () => 'python',
        spawnProcess, installAccPreset, platform });
    const event = { sender: contents, senderFrame: contents.mainFrame };
    return { handlers, event, contents, child, writes, spawnProcess, installAccPreset, app,
        connect: async (options) => {
            const connected = handlers['virtual-controller-connect'](event, options);
            child.stdout.write(JSON.stringify({ type: 'ready' }) + '\n');
            await connected;
        } };
}

beforeEach(() => jest.useFakeTimers());
afterEach(() => { jest.runOnlyPendingTimers(); jest.useRealTimers(); });

it('restricts IPC to the main frame and reports startup failures', async () => {
    const run = setup();
    await expect(run.handlers['virtual-controller-connect']({ sender: {} })).rejects.toThrow(/main workspace/);
    await expect(run.handlers['virtual-controller-connect']({ ...run.event, senderFrame: {} })).rejects.toThrow(/main workspace/);
    expect(run.spawnProcess).not.toHaveBeenCalled();
    const result = run.handlers['virtual-controller-connect'](run.event);
    run.child.stdout.write(JSON.stringify({ type: 'error', error: 'Install vJoy.' }) + '\n');
    await expect(result).rejects.toThrow('Install vJoy');
});

it('installs an ACC preset only for the main frame using the Windows Documents path', async () => {
    const run = setup();
    const install = run.handlers['virtual-controller-install-acc-preset'];
    await expect(install({ sender: {} }, 3)).rejects.toThrow(/main workspace/);
    await expect(install({ ...run.event, senderFrame: {} }, 3)).rejects.toThrow(/main workspace/);
    for (const id of [0, 17, 1.5, '3', null]) await expect(install(run.event, id)).rejects.toThrow(/device ID/);
    expect(run.installAccPreset).not.toHaveBeenCalled();
    await expect(install(run.event, 3)).resolves.toEqual({ name: 'ACLA AI Training - vJoy 3', path: '/preset.json', deviceId: 3 });
    expect(run.app.getPath).toHaveBeenCalledWith('documents');
    expect(run.installAccPreset).toHaveBeenCalledWith('/redirected/documents', 3);
    expect(run.spawnProcess).not.toHaveBeenCalled();
});

it('rejects a preset for a different connected device or an unsupported platform', async () => {
    const run = setup(); await run.connect({ deviceId: 3 });
    await expect(run.handlers['virtual-controller-install-acc-preset'](run.event, 1)).rejects.toThrow(/connected vJoy device/);
    expect(run.installAccPreset).not.toHaveBeenCalled();
    run.handlers['virtual-controller-disconnect'](run.event);
    const other = setup('linux');
    await expect(other.handlers['virtual-controller-install-acc-preset'](other.event, 1)).rejects.toThrow(/Windows/);
    expect(other.installAccPreset).not.toHaveBeenCalled();
});

it('routes wheel output to vJoy and requires disconnect before changing devices', async () => {
    const run = setup();
    await run.connect({ backend: 'vjoy', deviceId: 3 });
    expect(run.spawnProcess).toHaveBeenCalledWith('python',
        expect.arrayContaining(['--backend', 'vjoy', '--device-id', '3']), expect.anything());
    await run.handlers['virtual-controller-connect'](run.event, { backend: 'vjoy', deviceId: 3 });
    expect(run.spawnProcess).toHaveBeenCalledTimes(1);
    await expect(run.handlers['virtual-controller-connect'](run.event, { backend: 'vjoy', deviceId: 4 })).rejects.toThrow(/Disconnect/);
    run.handlers['virtual-controller-disconnect'](run.event);
});

it.each([null, { backend: 'xbox' }, { backend: 'moza' }, { backend: 'vjoy', deviceId: 0 },
    { backend: 'vjoy', deviceId: 17 }, { backend: 'vjoy', deviceId: 1.5 },
    { backend: 'vjoy', deviceId: '1' }])('rejects invalid output options before starting a process: %j', async (options) => {
    const run = setup();
    await expect(run.handlers['virtual-controller-connect'](run.event, options)).rejects.toThrow(/controller output/);
    expect(run.spawnProcess).not.toHaveBeenCalled();
});

it('defaults to vJoy, waits for device acknowledgement and uses a hidden child process', async () => {
    const run = setup(); await run.connect();
    expect(run.spawnProcess).toHaveBeenCalledWith('python',
        ['-u', expect.stringContaining('virtual_controller.py'), '--backend', 'vjoy', '--device-id', '1'],
        expect.objectContaining({ windowsHide: true }));
    expect(run.contents.setBackgroundThrottling).toHaveBeenCalledWith(false);
    const action = { throttle: 0.4, brake: 0, steering: -0.2 };
    const applied = run.handlers['virtual-controller-apply'](run.event, action);
    expect(run.writes).toEqual([{ id: 1, action }]);
    run.child.stdout.write(JSON.stringify({ type: 'applied', id: 1 }) + '\n');
    await applied;
    run.handlers['virtual-controller-disconnect'](run.event);
    expect(run.child.stdin.writableEnded).toBe(true);
    expect(run.contents.setBackgroundThrottling).toHaveBeenLastCalledWith(true);
});

it.each([NaN, Infinity, 1.2, -0.1])('rejects invalid pedal value %s and closes the device', async (value) => {
    const run = setup(); await run.connect();
    await expect(run.handlers['virtual-controller-apply'](run.event, { throttle: value, brake: 0, steering: 0 })).rejects.toThrow(/finite/);
    expect(run.child.stdin.writableEnded).toBe(true);
});

it.each(['shiftUp', 'shiftDown'])('forwards %s presses and neutral releases to the wheel bridge', async (key) => {
    const run = setup(); await run.connect();
    const neutral = { throttle: 0, brake: 0, steering: 0 };
    for (const action of [{ ...neutral, [key]: true }, neutral]) {
        const applied = run.handlers['virtual-controller-apply'](run.event, action);
        const command = run.writes[run.writes.length - 1];
        expect(command.action).toEqual(action);
        run.child.stdout.write(JSON.stringify({ type: 'applied', id: command.id }) + '\n');
        await applied;
    }
    run.handlers['virtual-controller-disconnect'](run.event);
});

it.each([{ shiftUp: 1 }, { shiftDown: 0 }, { shiftUp: null }, { shiftDown: 'true' },
    { shiftUp: true, shiftDown: true }, { unknown: false }])('rejects invalid shift controls %j and closes the device', async (buttons) => {
    const run = setup(); await run.connect();
    await expect(run.handlers['virtual-controller-apply'](run.event,
        { throttle: 0, brake: 0, steering: 0, ...buttons })).rejects.toThrow(/shift buttons/);
    expect(run.writes).toEqual([]);
    expect(run.child.stdin.writableEnded).toBe(true);
});

it('rejects pending commands on renderer loss, and times out an unresponsive device', async () => {
    const run = setup(); await run.connect();
    const action = { throttle: 0.1, brake: 0, steering: 0 };
    const pending = run.handlers['virtual-controller-apply'](run.event, action);
    const rejection = expect(pending).rejects.toThrow(/disconnected/);
    run.contents.emit('render-process-gone');
    await rejection;
    const second = setup(); await second.connect();
    const timedOut = second.handlers['virtual-controller-apply'](second.event, action);
    const timeoutRejection = expect(timedOut).rejects.toThrow(/responding/);
    jest.advanceTimersByTime(1000);
    await timeoutRejection;
    expect(second.child.stdin.writableEnded).toBe(true);
});
