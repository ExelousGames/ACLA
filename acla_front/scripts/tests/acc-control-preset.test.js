const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { buildAccControlPreset, installAccControlPreset } = require('../../electron/acc-control-preset');
const { registerVirtualController } = require('../../electron/virtual-controller');

// ACC's saved control schema, with synthetic device GUIDs.
function controls() {
    return {
        version: 0, configurationName: 'My wheel', carControllerMode: 1,
        keyboardSettings: { mouseSteering: false }, gamepadSettings: { deadZone: 0.1 },
        commandDevices: [
            { productName: 'vJoy Device 3', instanceGuid: '{33333333-1111-2222-3333-444444444444}',
                productId: '3199013428-0-0-00807368867368', raceCommandAxisList: [],
                brakeGamma: 2, steerScale: -1, steerLock: 360, combinedPedals: 1 },
            { productName: 'Physical wheel', instanceGuid: '{99999999-1111-2222-3333-444444444444}',
                raceCommandButtonList: [{ instantActionCode: 'GearUp', buttonIndex: 4 }],
                raceCommandAxisList: ['Steer', 'Gas', 'Brake', 'Clutch'].map((axisCode, axisIndex) => ({ axisCode, axisIndex })) },
            { productName: 'vJoy Device', instanceGuid: '{11111111-1111-2222-3333-444444444444}',
                productId: '3199013428-0-0-00807368867368', raceCommandAxisList: [{ axisCode: 'Gas', axisIndex: 5 }] },
        ],
    };
}

test('binds the selected device using ACC identity, independent of device order', () => {
    const original = controls();
    const before = JSON.stringify(original);
    const preset = buildAccControlPreset(original, 3);
    assert.equal(preset.carControllerMode, 0);
    assert.equal(preset.configurationName, 'ACLA AI Training - vJoy 3');
    const wheel = preset.commandDevices[0];
    assert.equal(wheel.instanceGuid, original.commandDevices[0].instanceGuid);
    assert.equal(wheel.productId, original.commandDevices[0].productId);
    assert.deepEqual(wheel.raceCommandAxisList, [
        { axisIndex: 0, axisCode: 'Steer', minRange: -1, maxRange: 1, isCombined: false },
        { axisIndex: 1, axisCode: 'Gas', minRange: -1, maxRange: 1, isCombined: false },
        { axisIndex: 2, axisCode: 'Brake', minRange: -1, maxRange: 1, isCombined: false },
    ]);
    assert.equal(wheel.brakeGamma, 1);
    assert.equal(wheel.steerScale, 1);
    assert.equal(wheel.steerLock, 900);
    assert.equal(wheel.combinedPedals, 0);
    assert.equal(wheel.forceFeedbackGain, 0);
    assert.deepEqual(wheel.raceCommandButtonList, ['GearUp', 'GearDown'].map((instantActionCode, buttonIndex) => ({
        buttonIndex, powIndex: -1, powValue: 0, keyName: 'None', gamepadButtonName: 'None',
        instantActionCode, extendedActionCode: 'None', extendedTime: 1,
        pinkieInstanceActionCode: 'None', pinkieExtendedActionCode: 'None',
    })));
    assert.deepEqual(preset.commandDevices[1].raceCommandAxisList, [{ axisCode: 'Clutch', axisIndex: 3 }]);
    assert.deepEqual(preset.commandDevices[2].raceCommandAxisList, []);
    assert.deepEqual(preset.commandDevices[1].raceCommandButtonList, original.commandDevices[1].raceCommandButtonList);
    assert.deepEqual(preset.keyboardSettings, original.keyboardSettings);
    assert.deepEqual(preset.gamepadSettings, original.gamepadSettings);
    assert.equal(JSON.stringify(original), before);
});

test('replaces conflicting shift bindings on the selected device and preserves unrelated buttons', () => {
    const original = controls();
    const look = { buttonIndex: 5, instantActionCode: 'LookLeft' };
    original.commandDevices[0].raceCommandButtonList = [
        { buttonIndex: 0, instantActionCode: 'PitLimiter' },
        { buttonIndex: 1, instantActionCode: 'LookRight' },
        { buttonIndex: 3, instantActionCode: 'GearUp' },
        { buttonIndex: 4, instantActionCode: 'GearDown' }, look,
    ];
    const before = JSON.stringify(original);
    const preset = buildAccControlPreset(original, 3);
    assert.deepEqual(preset.commandDevices[0].raceCommandButtonList.map(({ buttonIndex, instantActionCode }) =>
        ({ buttonIndex, instantActionCode })), [look,
        { buttonIndex: 0, instantActionCode: 'GearUp' }, { buttonIndex: 1, instantActionCode: 'GearDown' }]);
    assert.deepEqual(preset.commandDevices[1].raceCommandButtonList, original.commandDevices[1].raceCommandButtonList);
    assert.equal(JSON.stringify(original), before);
    assert.deepEqual(buildAccControlPreset(preset, 3), preset);
});

test('recognizes the unnumbered first device without falling back to it for other IDs', () => {
    const preset = buildAccControlPreset(controls(), 1);
    assert.equal(preset.commandDevices[2].raceCommandAxisList.length, 3);
    assert.equal(preset.commandDevices[0].raceCommandAxisList.length, 0);
    assert.throws(() => buildAccControlPreset(controls(), 2), /Enable it in Configure vJoy/);
    for (const deviceId of [0, 17, 1.5, '1', null, '../controls']) {
        assert.throws(() => buildAccControlPreset(controls(), deviceId), /device ID/);
    }
});

test('rejects missing, ambiguous or invalid ACC device identities', () => {
    const original = controls();
    original.commandDevices.push({ ...original.commandDevices[0] });
    assert.throws(() => buildAccControlPreset(original, 3), /uniquely identified/);
    original.commandDevices.pop();
    delete original.commandDevices[0].instanceGuid;
    assert.throws(() => buildAccControlPreset(original, 3), /uniquely identified/);
    for (const invalid of [null, {}, { commandDevices: [null] }, { commandDevices: [{}] }]) {
        assert.throws(() => buildAccControlPreset(invalid, 1), /unsupported format/);
    }
});

async function workspace(t, data = JSON.stringify(controls())) {
    const documents = await fs.mkdtemp(path.join(os.tmpdir(), 'acla-acc-preset-'));
    t.after(() => fs.rm(documents, { recursive: true, force: true }));
    const config = path.join(documents, 'Assetto Corsa Competizione', 'Config');
    await fs.mkdir(config, { recursive: true });
    const activePath = path.join(config, 'controls.json');
    await fs.writeFile(activePath, data);
    return { documents, activePath };
}

test('ACC preset button handler installs axes and both shifts on the selected device without changing active controls', async (t) => {
    const data = '\uFEFF' + JSON.stringify(controls());
    const { documents, activePath } = await workspace(t, data);
    const handlers = {};
    const contents = { mainFrame: {} };
    registerVirtualController({
        app: { getPath: (name) => { assert.equal(name, 'documents'); return documents; }, on() {} },
        ipcMain: { handle: (name, handler) => { handlers[name] = handler; } },
        getMainWindow: () => ({ isDestroyed: () => false, webContents: contents }),
        platform: 'win32',
    });
    const install = () => handlers['virtual-controller-install-acc-preset']({
        sender: contents, senderFrame: contents.mainFrame,
    }, 3);
    const result = await install();
    assert.equal(result.path, path.join(documents, 'Assetto Corsa Competizione', 'Customs', 'Controls', `${result.name}.json`));
    assert.equal(result.deviceId, 3);
    const saved = JSON.parse(await fs.readFile(result.path, 'utf8'));
    assert.equal(saved.configurationName, result.name);
    assert.equal(saved.commandDevices[0].raceCommandAxisList.length, 3);
    assert.deepEqual(saved.commandDevices[0].raceCommandButtonList.map(({ buttonIndex, instantActionCode }) =>
        ({ buttonIndex, instantActionCode })), [
        { buttonIndex: 0, instantActionCode: 'GearUp' },
        { buttonIndex: 1, instantActionCode: 'GearDown' },
    ]);
    assert.equal(saved.commandDevices[2].raceCommandButtonList, undefined);
    assert.equal(await fs.readFile(activePath, 'utf8'), data);
    assert.deepEqual(await install(), result);
});

test('preserves an existing user preset and exports with a distinct name', async (t) => {
    const { documents } = await workspace(t);
    const first = await installAccControlPreset(documents, 1);
    const userData = '{"configurationName":"User edited training preset"}';
    await fs.writeFile(first.path, userData);
    const second = await installAccControlPreset(documents, 1);
    assert.equal(second.name, 'ACLA AI Training - vJoy 1 (2)');
    assert.equal(await fs.readFile(first.path, 'utf8'), userData);
    assert.equal(JSON.parse(await fs.readFile(second.path, 'utf8')).configurationName, second.name);
    assert.deepEqual(await installAccControlPreset(documents, 1), second);
});

test('reports missing or malformed ACC controls before creating a preset', async (t) => {
    const { documents } = await workspace(t, '{unfinished');
    await assert.rejects(installAccControlPreset(documents, 1), /could not be read/);
    await assert.rejects(installAccControlPreset(path.join(documents, 'missing'), 1), /not found/);
    await assert.rejects(fs.access(path.join(documents, 'Assetto Corsa Competizione', 'Customs')), { code: 'ENOENT' });
});
