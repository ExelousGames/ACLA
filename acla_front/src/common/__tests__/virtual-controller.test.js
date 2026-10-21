const { EventEmitter } = require('events');
const { PassThrough, Writable } = require('stream');
const { registerVirtualController } = require('../../../electron/virtual-controller');

function setup() {
    const handlers = {};
    const app = new EventEmitter();
    app.getAppPath = () => '/app';
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
        spawnProcess, platform: 'win32' });
    const event = { sender: contents, senderFrame: contents.mainFrame };
    return { handlers, event, contents, child, writes, spawnProcess,
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
