#!/usr/bin/env node
const { spawnSync } = require('node:child_process');
const path = require('node:path');
const prepareVjoy = require('./prepare-vjoy');

const projectRoot = path.resolve(__dirname, '..');

function run(command, args, options = {}) {
    const result = spawnSync(command, args, { cwd: projectRoot, stdio: 'inherit', ...options });
    if (result.error) throw result.error;
    if (result.status !== 0) {
        throw new Error(`Setup failed (${result.signal || result.status}): ${command} ${args.join(' ')}`);
    }
}

async function setupAll() {
    // npm supplies its CLI path even before node_modules exists. Calling it with
    // Node avoids Windows .cmd spawning/quoting issues and includes build tools.
    if (!process.env.npm_execpath) throw new Error('Run this script with npm run setup:all.');
    if (process.platform === 'win32' && (process.arch !== 'x64'
        || (process.env.PROCESSOR_ARCHITEW6432 || process.env.PROCESSOR_ARCHITECTURE) === 'ARM64')) {
        throw new Error('vJoy requires Windows x64 and an x64 Node.js installation.');
    }
    console.log('Installing Node dependencies...');
    run(process.execPath, [process.env.npm_execpath, 'install', '--include=dev']);
    run(process.execPath, [path.join(__dirname, 'setup-python.js'), '--mode=dev']);
    run(process.execPath, [path.join(__dirname, 'setup-python.js'), '--mode=vision']);
    run(process.execPath, [path.join(__dirname, 'prepare-vision-runtime.js')]);
    run(process.execPath, [path.join(__dirname, 'setup-vision-models.js')], {
        env: {
            ...process.env,
            VISION_PYTHON: path.join(projectRoot, '.venv', 'track-vision',
                process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python'),
        },
    });

    if (process.platform === 'win32') {
        await prepareVjoy({ electronPlatformName: 'win32', arch: 1, packager: { projectDir: projectRoot } });
        run('powershell.exe', ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
            path.join(__dirname, 'install-vjoy.ps1'), '-InstallerPath',
            path.join(projectRoot, 'assets', 'vjoy', 'vJoySetup.exe')]);
    } else {
        console.log('Skipping the vJoy driver: it is only supported on Windows x64.');
    }
    console.log('\nDependency setup complete. Start the desktop app with npm run start:electron.');
}

module.exports = setupAll;
if (require.main === module) {
    setupAll().catch((error) => { console.error(error.message); process.exitCode = 1; });
}
