const fs = require('fs/promises');
const path = require('path');

const DRIVING_AXES = ['Steer', 'Gas', 'Brake'];
const SHIFT_BUTTONS = ['GearUp', 'GearDown'];

function validateDeviceId(deviceId) {
  if (!Number.isInteger(deviceId) || deviceId < 1 || deviceId > 16) {
    throw new Error('Select a vJoy device ID from 1 to 16.');
  }
}

function buildAccControlPreset(controls, deviceId) {
  validateDeviceId(deviceId);
  if (!controls || !Array.isArray(controls.commandDevices)
    || controls.commandDevices.some((device) => !device || !Array.isArray(device.raceCommandAxisList)
      || (device.raceCommandButtonList !== undefined && !Array.isArray(device.raceCommandButtonList)))) {
    throw new Error('ACC controls have an unsupported format. Save your controls in ACC and try again.');
  }
  // Use ACC's own device identity; GUIDs vary between Windows installations.
  // vJoy names its first device "vJoy Device", then "vJoy Device 2", etc.
  const matches = controls.commandDevices.filter((device) => {
    const match = /^vJoy Device(?: (\d+))?$/i.exec(device.productName);
    return match && Number(match[1] || 1) === deviceId;
  });
  if (matches.length !== 1 || !/^\{[\da-f-]{36}\}$/i.test(matches[0].instanceGuid)) {
    throw new Error(`ACC has not uniquely identified vJoy device ${deviceId}. Enable it in Configure vJoy, restart ACC, save your controls, then try again.`);
  }
  const preset = JSON.parse(JSON.stringify(controls));
  preset.configurationName = `ACLA AI Training - vJoy ${deviceId}`;
  preset.carControllerMode = 0; // ACC Wheel input mode.
  for (const device of preset.commandDevices) {
    // A physical wheel must not compete for the three AI-controlled inputs.
    device.raceCommandAxisList = device.raceCommandAxisList.filter((axis) => !DRIVING_AXES.includes(axis.axisCode));
    if (device.instanceGuid !== matches[0].instanceGuid) continue;
    Object.assign(device, {
      isGamePad: 0, isPedals: 0, forceFeedbackGain: 0,
      steerScale: 1, steerLock: 900, steerLinearity: 1, brakeGamma: 1, combinedPedals: 0,
      // DirectInput X/Y/Z are indices 0/1/2. The bridge sends pedals low -> high.
      raceCommandAxisList: DRIVING_AXES.map((axisCode, axisIndex) => ({
        axisIndex, axisCode, minRange: -1, maxRange: 1, isCombined: false,
      })),
      // ACC uses zero-based DirectInput button indices; vJoy SetBtn uses 1/2.
      raceCommandButtonList: [
        ...(device.raceCommandButtonList || []).filter((button) =>
          ![0, 1].includes(button.buttonIndex) && !SHIFT_BUTTONS.includes(button.instantActionCode)),
        ...SHIFT_BUTTONS.map((instantActionCode, buttonIndex) => ({
          buttonIndex, powIndex: -1, powValue: 0, keyName: 'None', gamepadButtonName: 'None',
          instantActionCode, extendedActionCode: 'None', extendedTime: 1,
          pinkieInstanceActionCode: 'None', pinkieExtendedActionCode: 'None',
        })),
      ],
    });
  }
  return preset;
}

async function installAccControlPreset(documentsPath, deviceId) {
  validateDeviceId(deviceId);
  const accDirectory = path.join(documentsPath, 'Assetto Corsa Competizione');
  let controls;
  try {
    controls = JSON.parse((await fs.readFile(path.join(accDirectory, 'Config', 'controls.json'), 'utf8')).replace(/^\uFEFF/, ''));
  } catch (error) {
    if (error.code === 'ENOENT') throw new Error('ACC controls were not found. Start ACC, save your controls, then try again.');
    if (error instanceof SyntaxError) throw new Error('ACC controls could not be read. Save your controls in ACC and try again.');
    throw error;
  }
  const preset = buildAccControlPreset(controls, deviceId);
  const directory = path.join(accDirectory, 'Customs', 'Controls');
  await fs.mkdir(directory, { recursive: true });
  const baseName = preset.configurationName;
  // Reuse an identical export, but never overwrite an existing user preset.
  for (let suffix = 1; ; suffix++) {
    preset.configurationName = baseName + (suffix === 1 ? '' : ` (${suffix})`);
    const presetPath = path.join(directory, `${preset.configurationName}.json`);
    const data = JSON.stringify(preset, null, '\t') + '\n';
    try {
      await fs.writeFile(presetPath, data, { encoding: 'utf8', flag: 'wx' });
    } catch (error) {
      if (error.code !== 'EEXIST') throw error;
      if (await fs.readFile(presetPath, 'utf8') !== data) continue;
    }
    return { name: preset.configurationName, path: presetPath, deviceId };
  }
}

module.exports = { buildAccControlPreset, installAccControlPreset };
