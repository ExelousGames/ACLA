const { registerScreenCapture } = require('../../../electron/screen-capture');

function setup() {
    const handlers = {};
    const frame = {};
    let handler;
    let navigate;
    const contents = { mainFrame: frame, on: (_, callback) => { navigate = callback; }, session: {
        setDisplayMediaRequestHandler: (callback) => { handler = callback; },
    } };
    const window = { isDestroyed: () => false, webContents: contents };
    const sources = [{ id: 'window:42', name: 'Simulator' }, { id: 'screen:0', name: 'Screen 1' }];
    const getSources = jest.fn().mockResolvedValue(sources);
    registerScreenCapture({ ipcMain: { handle: (name, callback) => { handlers[name] = callback; } },
        desktopCapturer: { getSources }, getMainWindow: () => window })(window);
    return { handlers, frame, event: { sender: contents, senderFrame: frame }, sources, getSources,
        request: (options, callback) => handler({ frame, videoRequested: true, ...options }, callback),
        navigate: () => navigate() };
}

it('lists sources for the main frame and grants only the explicitly selected source once', async () => {
    const capture = setup();
    expect(await capture.handlers['screen-capture-sources'](capture.event)).toEqual(capture.sources);
    await capture.handlers['screen-capture-select'](capture.event, 'window:42');
    const callback = jest.fn();
    await capture.request({}, callback);
    expect(callback).toHaveBeenLastCalledWith({ video: capture.sources[0] });
    await capture.request({}, callback);
    expect(callback).toHaveBeenLastCalledWith({});
});

it('rejects foreign renderers, subframes, and unavailable sources', async () => {
    const capture = setup();
    await expect(capture.handlers['screen-capture-sources']({ sender: {} })).rejects.toThrow('main workspace');
    await expect(capture.handlers['screen-capture-select']({ ...capture.event, senderFrame: {} }, 'window:42')).rejects.toThrow('main workspace');
    await expect(capture.handlers['screen-capture-select'](capture.event, 'window:missing')).rejects.toThrow('no longer available');
    await capture.handlers['screen-capture-select'](capture.event, 'window:42');
    const callback = jest.fn();
    await capture.request({ frame: {} }, callback);
    expect(callback).toHaveBeenCalledWith({});
});

it('clears selections on navigation and handles disappearing windows', async () => {
    const capture = setup();
    const callback = jest.fn();
    await capture.handlers['screen-capture-select'](capture.event, 'window:42');
    capture.navigate();
    await capture.request({}, callback);
    expect(callback).toHaveBeenLastCalledWith({});
    await capture.handlers['screen-capture-select'](capture.event, 'window:42');
    capture.getSources.mockResolvedValue([]);
    await capture.request({}, callback);
    expect(callback).toHaveBeenLastCalledWith({});
});

it('does not grant capture if the main frame changes during source lookup', async () => {
    const capture = setup();
    await capture.handlers['screen-capture-select'](capture.event, 'window:42');
    let resolveSources;
    capture.getSources.mockReturnValue(new Promise((resolve) => { resolveSources = resolve; }));
    const callback = jest.fn();
    const request = capture.request({}, callback);
    capture.event.sender.mainFrame = {};
    resolveSources(capture.sources);
    await request;
    expect(callback).toHaveBeenCalledWith({});
});
