#!/usr/bin/env node
// Download at build time so the Windows installer works without network access.
const fs = require('node:fs/promises');
const path = require('node:path');
const { createHash } = require('node:crypto');

const release = {
    version: '2.2.2.0',
    url: 'https://github.com/BrunnerInnovation/vJoy/releases/download/v2.2.2.0/vJoySetup_v2.2.2.0_Win10_Win11.exe',
    sha256: 'ef569a3105cd301b89580f18f60c66b339e95296acf2c0dfcaf4b4bbf8ab68fe',
};

function verifyInstaller(contents) {
    if (createHash('sha256').update(contents).digest('hex') !== release.sha256) {
        throw new Error('vJoy installer checksum mismatch. Remove assets/vjoy/vJoySetup.exe and run npm run setup:vjoy again.');
    }
}

async function prepareVjoy(context, fetchInstaller = fetch) {
    if (context.electronPlatformName !== 'win32') return;
    // electron-builder's Arch.x64 = 1. The signed driver has no ARM64/x86 build.
    if (context.arch !== 1) throw new Error('The bundled vJoy driver requires a Windows x64 build.');
    const directory = path.join(context.packager.projectDir, 'assets', 'vjoy');
    const installer = path.join(directory, 'vJoySetup.exe');
    let contents;
    try {
        contents = await fs.readFile(installer);
    } catch (error) {
        if (error.code !== 'ENOENT') throw error;
    }
    if (contents) {
        verifyInstaller(contents);
        return;
    }

    console.log(`Downloading signed vJoy ${release.version} for the Windows installer...`);
    const response = await fetchInstaller(release.url, { signal: AbortSignal.timeout(120000) });
    if (!response.ok) throw new Error(`vJoy download failed: HTTP ${response.status}`);
    contents = Buffer.from(await response.arrayBuffer());
    verifyInstaller(contents);
    await fs.mkdir(directory, { recursive: true });
    // Only a complete, verified download is published to the packaging directory.
    const temporary = `${installer}.tmp`;
    await fs.writeFile(temporary, contents);
    await fs.rename(temporary, installer);
}

module.exports = prepareVjoy;
module.exports.release = release;
module.exports.verifyInstaller = verifyInstaller;

if (require.main === module) {
    prepareVjoy({ electronPlatformName: 'win32', arch: 1, packager: { projectDir: path.resolve(__dirname, '..') } })
        .catch((error) => { console.error(error.message); process.exitCode = 1; });
}
