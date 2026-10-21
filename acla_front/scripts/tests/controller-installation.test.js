const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { spawnSync } = require('node:child_process');

const root = path.resolve(__dirname, '../..');
const expectedHash = '89220a7865076b342892f98865f3499fb7c4cfd673159e89d352c360fd014c6a';

function driverBuild({ cached = false, valid = true, status = 200 } = {}) {
    const calls = { downloads: [], writes: [] };
    const sandbox = {
        module: { exports: {} }, Buffer, AbortSignal, console: { log() {} },
        fetch: async (...args) => {
            calls.downloads.push(args);
            return { ok: status === 200, status, arrayBuffer: async () => Buffer.from('driver') };
        },
        require: (name) => ({
            fs: {
                existsSync: () => cached,
                readFileSync: () => Buffer.from('driver'),
                mkdirSync() {},
                writeFileSync: (...args) => calls.writes.push(args),
            },
            path,
            crypto: { createHash: (algorithm) => {
                assert.equal(algorithm, 'sha256');
                return { update: (contents) => {
                    assert.equal(contents.toString(), 'driver');
                    return { digest: () => valid ? expectedHash : 'invalid' };
                } };
            } },
        }[name]),
    };
    vm.runInNewContext(fs.readFileSync(path.join(root, 'scripts/prepare-controller-driver.js'), 'utf8'), sandbox);
    return { prepare: sandbox.module.exports, calls };
}

const context = { electronPlatformName: 'win32', packager: { projectDir: root } };

test('Windows packaging downloads the pinned driver and reuses a verified cache', async () => {
    const run = driverBuild();
    await run.prepare(context);
    assert.match(run.calls.downloads[0][0], /releases\/download\/v1\.22\.0\/ViGEmBus_1\.22\.0_x64_x86_arm64.exe$/);
    assert.equal(run.calls.writes.length, 1);
    assert.equal(run.calls.writes[0][0], path.join(root, '.venv/controller-driver/ViGEmBus_1.22.0_x64_x86_arm64.exe'));
    const cached = driverBuild({ cached: true });
    await cached.prepare(context);
    assert.equal(cached.calls.downloads.length, 0);
    assert.equal(cached.calls.writes.length, 0);
});

test('packaging fails for a corrupt download, corrupt cache, or unavailable release', async () => {
    for (const options of [{ valid: false }, { cached: true, valid: false }, { status: 503 }]) {
        const run = driverBuild(options);
        await assert.rejects(run.prepare(context), /checksum mismatch|HTTP 503/);
        assert.equal(run.calls.writes.length, 0);
    }
});

test('non-Windows packaging does not provision a controller driver', async () => {
    const run = driverBuild();
    await run.prepare({ ...context, electronPlatformName: 'darwin' });
    assert.equal(run.calls.downloads.length, 0);
    assert.equal(run.calls.writes.length, 0);
});

test('both managed Python environments install controller requirements without launching the old driver setup', () => {
    for (const mode of ['dev', 'prod']) {
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
        assert.equal(install.command, path.join(root, '.venv', mode === 'prod' ? 'py-scripts-prod' : 'py-scripts', 'Scripts/python.exe'));
        assert.equal(install.options.env.VGAMEPAD_SKIP_VIGEMBUS_INSTALL, 'true');
        assert.equal(install.options.env.TEST_ENV, 'preserved');
        const requirements = fs.readFileSync(install.args.at(-1), 'utf8');
        const included = requirements.match(/^-r (.+)$/m)[1].trim();
        const controllerRequirements = fs.readFileSync(path.resolve(path.dirname(install.args.at(-1)), included), 'utf8');
        // This release is hosted on GitHub, not PyPI. An index version pin cannot resolve it.
        assert.match(controllerRequirements, /^vgamepad @ https:\/\/github\.com\/yannbouteiller\/vgamepad\/archive\/refs\/tags\/v0\.1\.3\.zip#sha256=[a-f0-9]{64} ; sys_platform == 'win32'$/m);
    }
    const config = require('../../package.json').build;
    assert.equal(config.beforePack, 'scripts/prepare-controller-driver.js');
    assert.equal(config.nsis.include, 'assets/installer.nsh');
    assert.ok(config.extraResources.some(({ from, to }) => from === '.venv/py-scripts-prod' && to === 'python-env'));
});

// Execute the real installation script with OS boundaries mocked. No driver runs.
const cases = [
    { name: 'existing working driver', ready: 'always', code: 0, launches: false },
    { name: 'fresh installation', ready: 'after', code: 0, launches: true },
    { name: 'restart requested', driverCode: 3010, code: 3010, launches: true },
    { name: 'restart initiated', driverCode: 1641, code: 3010, launches: true },
    { name: 'driver setup failure', driverCode: 1603, code: 1603, launches: true },
    { name: 'administrator approval declined', cancelled: true, code: 1223, launches: true },
    { name: 'missing payload', missing: true, code: 1, launches: false },
    { name: 'driver absent after reported success', code: 1, launches: true },
    { name: 'device detection failure', detectionFailed: true, code: 1, launches: false },
];
for (const scenario of cases) {
    test(`driver installation handles ${scenario.name}`, { skip: process.platform !== 'win32' }, () => {
        const harness = `
            $script:queries = 0
            function Get-CimInstance {
                param($ClassName, $Filter)
                if ($ClassName -ne 'Win32_PnPEntity' -or $Filter -ne "Service = 'ViGEmBus'") { throw 'Invalid device query' }
                if ($env:DETECTION_FAILED -eq 'true') { throw 'WMI unavailable' }
                $script:queries++
                if ($env:READY -eq 'always' -or ($env:READY -eq 'after' -and $script:queries -gt 1)) {
                    [pscustomobject]@{ ConfigManagerErrorCode = 0 }
                } else {
                    [pscustomobject]@{ ConfigManagerErrorCode = 28 }
                }
            }
            function Test-Path { param($LiteralPath, $PathType) return $env:MISSING -ne 'true' }
            function Start-Process {
                param($FilePath, $ArgumentList, $Verb, $WindowStyle, [switch]$Wait, [switch]$PassThru)
                Write-Output 'DRIVER_SETUP_LAUNCHED' | Out-Host
                if ($FilePath -ne 'C:\\test path\\ViGEmBus.exe' -or $Verb -ne 'RunAs' -or $WindowStyle -ne 'Hidden' -or -not $Wait -or -not $PassThru -or ($ArgumentList -join ' ') -ne '/exenoui /qn /norestart') { throw 'Invalid setup invocation' }
                if ($env:CANCELLED -eq 'true') { throw [System.ComponentModel.Win32Exception]::new(1223) }
                [pscustomobject]@{ ExitCode = [int]$env:DRIVER_CODE }
            }
            & $env:INSTALL_SCRIPT -InstallerPath 'C:\\test path\\ViGEmBus.exe'
            exit $LASTEXITCODE
        `;
        const result = spawnSync('powershell.exe', ['-NoLogo', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command', harness], {
            encoding: 'utf8', windowsHide: true,
            env: { ...process.env, INSTALL_SCRIPT: path.join(root, 'assets/install-controller-driver.ps1'),
                READY: scenario.ready || '', DRIVER_CODE: String(scenario.driverCode || 0),
                CANCELLED: String(!!scenario.cancelled), MISSING: String(!!scenario.missing), DETECTION_FAILED: String(!!scenario.detectionFailed) },
        });
        assert.ifError(result.error);
        assert.equal(result.status, scenario.code, result.stdout + result.stderr);
        assert.equal(result.stdout.includes('DRIVER_SETUP_LAUNCHED'), scenario.launches, result.stdout);
    });
}
