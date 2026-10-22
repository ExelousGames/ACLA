const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '../..');

test('managed Python environments keep desktop and vision requirements separate', () => {
    for (const [mode, directory, requirementsFile] of [
        ['dev', 'py-scripts', 'src/py-scripts/requirements.txt'],
        ['prod', 'py-scripts-prod', 'src/py-scripts/requirements.txt'],
        ['vision', 'track-vision', 'scripts/vision-requirements.txt'],
    ]) {
        const calls = [];
        vm.runInNewContext(fs.readFileSync(path.join(root, 'scripts/setup-python.js'), 'utf8'), {
            __dirname: path.join(root, 'scripts'),
            process: { platform: 'win32', argv: ['node', 'setup-python.js', `--mode=${mode}`], env: { TEST_ENV: 'preserved' } },
            console: { log() {} },
            require: (name) => ({
                fs: { existsSync: () => true, writeFileSync() {} }, path,
                child_process: { spawnSync: (command, args, options) => {
                    calls.push({ command, args, options });
                    return { status: 0, stdout: Buffer.from('Python 3.10.9'), stderr: Buffer.alloc(0) };
                } },
            }[name]),
        });
        const install = calls.find(({ args }) => args.includes('-r'));
        assert.equal(install.command, path.join(root, '.venv', directory, 'Scripts/python.exe'));
        assert.deepEqual(Array.from(install.args), ['-m', 'pip', 'install', '-r', path.join(root, requirementsFile)]);
        const requirements = fs.readFileSync(install.args.at(-1), 'utf8');
        assert.doesNotMatch(requirements, /vgamepad|vigem|controller-requirements/i);
    }
    const config = require('../../package.json').build;
    assert.equal(config.beforePack, './scripts/prepare-vjoy.js');
    assert.equal(config.nsis?.include, 'assets/installer.nsh');
    assert.equal(config.nsis.perMachine, true);
    assert.equal(config.nsis.oneClick, false);
    assert.deepEqual(config.win.target, [{ target: 'nsis', arch: ['x64'] }]);
    assert.ok(config.win.extraResources.some(({ from, to, filter }) => from === 'assets/vjoy' && to === 'vjoy'
        && filter.includes('vJoySetup.exe') && filter.includes('LICENSE.txt')));
    assert.ok(config.extraResources.some(({ from, to }) => from === '.venv/py-scripts-prod' && to === 'python-env'));
    assert.ok(config.extraResources.some(({ from, to }) => from === 'src/py-scripts' && to === 'py-scripts'));
});
