const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const DRIVER_FILENAME = 'ViGEmBus_1.22.0_x64_x86_arm64.exe';
const DRIVER_URL = `https://github.com/nefarius/ViGEmBus/releases/download/v1.22.0/${DRIVER_FILENAME}`;
// Pinned upstream release; checksum also published in Microsoft's winget manifest:
// https://github.com/microsoft/winget-pkgs/blob/master/manifests/v/ViGEm/ViGEmBus/1.22.0/ViGEm.ViGEmBus.installer.yaml
const DRIVER_SHA256 = '89220a7865076b342892f98865f3499fb7c4cfd673159e89d352c360fd014c6a';

function verifyDriver(contents) {
    if (crypto.createHash('sha256').update(contents).digest('hex') !== DRIVER_SHA256) {
        throw new Error('ViGEmBus installer checksum mismatch. Remove .venv/controller-driver and rebuild.');
    }
}

// electron-builder invokes this even when called directly, outside npm's prebuild.
module.exports = async function prepareControllerDriver(context) {
    if (context.electronPlatformName !== 'win32') return;
    const directory = path.join(context.packager.projectDir, '.venv', 'controller-driver');
    const destination = path.join(directory, DRIVER_FILENAME);
    if (fs.existsSync(destination)) {
        verifyDriver(fs.readFileSync(destination));
        return;
    }

    console.log('Downloading the bundled ViGEmBus controller driver...');
    const response = await fetch(DRIVER_URL, { signal: AbortSignal.timeout(120000) });
    if (!response.ok) throw new Error(`Unable to download ViGEmBus: HTTP ${response.status}`);
    const contents = Buffer.from(await response.arrayBuffer());
    verifyDriver(contents);
    fs.mkdirSync(directory, { recursive: true });
    fs.writeFileSync(destination, contents);
};
