const assert = require('node:assert/strict');
const { test } = require('node:test');
const { execFileSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '../..');

function fixture({ platform = 'win32', arch = 'x64', failAt = -1, downloadError = false } = {}) {
    const calls = [];
    const module = { exports: {} };
    vm.runInNewContext(fs.readFileSync(path.join(root, 'scripts/setup-all.js'), 'utf8'), {
        __dirname: path.join(root, 'scripts'), module,
        console: { log() {} },
        process: { platform, arch, execPath: 'node', env: {
            npm_execpath: 'npm-cli.js', VISION_PYTHON: 'unrelated-python', PYTHON: 'custom-python',
        } },
        require: (name) => ({
            'node:path': path,
            'node:child_process': { spawnSync: (command, args, options) => {
                calls.push({ command, args: Array.from(args), options });
                return { status: calls.length - 1 === failAt ? 1 : 0 };
            } },
            './prepare-vjoy': async (context) => {
                calls.push({ download: context });
                if (downloadError) throw new Error('vJoy installer checksum mismatch');
            },
        }[name]),
    });
    return { calls, setup: module.exports };
}

test('one command bootstraps packages before exporting models and installing the verified driver', async () => {
    assert.equal(require('../../package.json').scripts['setup:all'], 'node scripts/setup-all.js');
    const { setup, calls } = fixture();
    await setup();
    assert.deepEqual(calls[0].args, ['npm-cli.js', 'install', '--include=dev']);
    assert.deepEqual(calls.slice(1, 5).map(({ args }) => [path.basename(args[0]), ...args.slice(1)]), [
        ['setup-python.js', '--mode=dev'], ['setup-python.js', '--mode=vision'],
        ['prepare-vision-runtime.js'], ['setup-vision-models.js'],
    ]);
    assert.equal(calls[4].options.env.VISION_PYTHON, path.join(root, '.venv/track-vision/Scripts/python.exe'));
    assert.equal(calls[4].options.env.PYTHON, 'custom-python');
    assert.equal(calls[5].download.packager.projectDir, root);
    assert.equal(calls[6].command, 'powershell.exe');
    assert.ok(calls[6].args.includes(path.join(root, 'scripts/install-vjoy.ps1')));
    assert.equal(calls[6].args.at(-1), path.join(root, 'assets/vjoy/vJoySetup.exe'));
    assert.ok(calls.filter(({ command }) => command).every(({ options }) => options.cwd === root));
});

test('a package, model or driver failure stops setup instead of reporting success', async () => {
    for (const failAt of [0, 1, 2, 3, 4, 6]) {
        const { setup, calls } = fixture({ failAt });
        await assert.rejects(setup(), /Setup failed/);
        assert.equal(calls.length, failAt + 1);
    }
    const { setup, calls } = fixture({ downloadError: true });
    await assert.rejects(setup(), /checksum mismatch/);
    assert.equal(calls.length, 6); // A failed verification never launches the installer.
});

test('non-Windows setup uses its managed vision Python and never downloads or installs vJoy', async () => {
    for (const platform of ['linux', 'darwin']) {
        const { setup, calls } = fixture({ platform });
        await setup();
        assert.equal(calls.length, 5);
        assert.equal(calls[4].options.env.VISION_PYTHON, path.join(root, '.venv/track-vision/bin/python'));
    }
});

test('unsupported Windows architectures fail before changing dependencies', async () => {
    const { setup, calls } = fixture({ arch: 'arm64' });
    await assert.rejects(setup(), /Windows x64/);
    assert.equal(calls.length, 0);
});

for (const scenario of ['existing', 'install', 'reboot3010', 'reboot8', 'failed', 'missing-sdk', 'cancelled']) {
    test(`vJoy installation: ${scenario}`, { skip: process.platform !== 'win32' }, () => {
        execFileSync('powershell.exe', ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
            path.join(__dirname, 'install-vjoy.test.ps1'), '-Scenario', scenario], { stdio: 'pipe' });
    });
}
