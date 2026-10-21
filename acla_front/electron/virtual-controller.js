const { spawn } = require('child_process');
const readline = require('readline');
const path = require('path');
const { installAccControlPreset } = require('./acc-control-preset');

function validAction(action) {
  return action && Object.keys(action).length === 3 && ['throttle', 'brake', 'steering'].every((key) =>
    typeof action[key] === 'number' && Number.isFinite(action[key]) && action[key] <= 1
      && action[key] >= (key === 'steering' ? -1 : 0));
}

function registerVirtualController({ app, ipcMain, getMainWindow, getPythonExecutable, spawnProcess = spawn,
  installAccPreset = installAccControlPreset, platform = process.platform }) {
  let active = null;
  const owner = (event) => {
    const window = getMainWindow();
    if (!window || window.isDestroyed() || event.sender !== window.webContents
      || event.senderFrame !== window.webContents.mainFrame) throw new Error('Virtual control is available only in the main workspace.');
    return window.webContents;
  };
  const close = (connection, reason = 'Virtual controller disconnected.') => {
    if (!connection || connection.closed) return;
    connection.closed = true;
    if (active === connection) active = null;
    clearTimeout(connection.startTimer);
    connection.rejectReady(new Error(reason));
    connection.pending.forEach(({ reject, timer }) => { clearTimeout(timer); reject(new Error(reason)); });
    connection.pending.clear();
    connection.contents.removeListener('destroyed', connection.onClose);
    connection.contents.removeListener('render-process-gone', connection.onClose);
    connection.contents.removeListener('did-start-navigation', connection.onNavigate);
    if (!connection.contents.isDestroyed()) connection.contents.setBackgroundThrottling(connection.backgroundThrottling);
    connection.lines?.close();
    // EOF makes Python reset and destroy the device. The independent Python watchdog also covers a main-process crash.
    connection.child.stdin.end();
    const timer = setTimeout(() => { connection.child.kill(); }, 800);
    timer.unref?.();
    connection.child.once('exit', () => clearTimeout(timer));
  };
  ipcMain.handle('virtual-controller-connect', async (event, options = {}) => {
    const contents = owner(event);
    if (platform !== 'win32') throw new Error('Virtual controllers currently require Windows.');
    const backend = options?.backend ?? 'vjoy';
    const deviceId = options?.deviceId ?? 1;
    if (!options || typeof options !== 'object' || Array.isArray(options)
      || backend !== 'vjoy' || !Number.isInteger(deviceId) || deviceId < 1 || deviceId > 16) {
      throw new Error('Invalid controller output. Use a vJoy steering wheel with a device ID from 1 to 16.');
    }
    if (active) {
      if (active.backend !== backend || active.deviceId !== deviceId) throw new Error('Disconnect before changing controller output.');
      return active.ready;
    }
    const script = app.isPackaged ? path.join(process.resourcesPath, 'py-scripts', 'virtual_controller.py')
      : path.join(app.getAppPath(), 'src', 'py-scripts', 'virtual_controller.py');
    const child = spawnProcess(getPythonExecutable(), ['-u', script, '--backend', backend, '--device-id', String(deviceId)],
      { windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] });
    const connection = { child, contents, backend, deviceId, pending: new Map(), sequence: 0, closed: false, started: false,
      backgroundThrottling: contents.getBackgroundThrottling() };
    // The simulator owns foreground focus while the controller is active.
    contents.setBackgroundThrottling(false);
    active = connection;
    connection.ready = new Promise((resolve, reject) => { connection.resolveReady = resolve; connection.rejectReady = reject; });
    connection.onClose = () => close(connection);
    connection.onNavigate = (_event, _url, _inPlace, isMainFrame) => { if (isMainFrame) close(connection); };
    contents.on('destroyed', connection.onClose);
    contents.on('render-process-gone', connection.onClose);
    contents.on('did-start-navigation', connection.onNavigate);
    connection.startTimer = setTimeout(() => close(connection, 'Virtual controller startup timed out.'), 10000);
    connection.lines = readline.createInterface({ input: child.stdout });
    connection.lines.on('line', (line) => {
      if (connection.closed) return;
      let message;
      try { message = JSON.parse(line); } catch { close(connection, 'Invalid virtual controller response.'); return; }
      if (message.type === 'ready') {
        clearTimeout(connection.startTimer);
        connection.started = true;
        connection.resolveReady();
      } else if (message.type === 'error') close(connection, message.error || 'Virtual controller failed.');
      else if (message.type === 'applied') {
        const pending = connection.pending.get(message.id);
        if (pending) { clearTimeout(pending.timer); connection.pending.delete(message.id); pending.resolve(); }
      }
    });
    let stderr = '';
    child.stderr.on('data', (chunk) => { stderr = (stderr + chunk.toString()).slice(-2000); });
    child.on('error', (error) => close(connection, error.message));
    child.stdin.on('error', (error) => close(connection, error.message));
    child.on('exit', () => close(connection, stderr.trim() || 'Virtual controller process ended. Reconnect to continue.'));
    return connection.ready;
  });
  ipcMain.handle('virtual-controller-apply', async (event, action) => {
    owner(event);
    if (!validAction(action)) {
      close(active, 'Invalid controller inputs.');
      throw new Error('Controller inputs must be finite: pedals 0–1 and steering −1–1.');
    }
    const connection = active;
    if (!connection?.started || connection.closed) throw new Error('Connect the virtual controller first.');
    if (connection.pending.size >= 4) { close(connection, 'Controller command queue overflow.'); throw new Error('Controller is not responding.'); }
    const id = ++connection.sequence;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => close(connection, 'Virtual controller stopped responding.'), 1000);
      connection.pending.set(id, { resolve, reject, timer });
      connection.child.stdin.write(JSON.stringify({ id, action }) + '\n', (error) => { if (error) close(connection, error.message); });
    });
  });
  ipcMain.handle('virtual-controller-disconnect', (event) => { owner(event); close(active); });
  ipcMain.handle('virtual-controller-install-acc-preset', async (event, deviceId) => {
    owner(event);
    if (platform !== 'win32') throw new Error('ACC control presets currently require Windows.');
    if (!Number.isInteger(deviceId) || deviceId < 1 || deviceId > 16) throw new Error('Select a vJoy device ID from 1 to 16.');
    if (active && active.deviceId !== deviceId) throw new Error('Select the connected vJoy device for the ACC preset.');
    return installAccPreset(app.getPath('documents'), deviceId);
  });
  app.on('before-quit', () => close(active));
}

module.exports = { registerVirtualController, validAction };
