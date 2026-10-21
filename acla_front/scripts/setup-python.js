#!/usr/bin/env node
/*
 * Ensure a Python virtual environment exists for the desktop Electron scripts
 * and install all required packages.
 *
 * Usage:
 *   node scripts/setup-python.js --mode=dev
 *   node scripts/setup-python.js --mode=prod
 */

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

if (process.platform === 'darwin') {
    console.log("macOS detected. Skipping Python setup since this device is only used for building the UI.");
    process.exit(0);
}

const projectRoot = path.resolve(__dirname, '..');
const requirementsPath = path.join(projectRoot, 'src', 'py-scripts', 'requirements.txt');

if (!fs.existsSync(requirementsPath)) {
    console.error(`Unable to find Python requirements at ${requirementsPath}`);
    process.exit(1);
}

const args = process.argv.slice(2);
const modeArg = args.find((arg) => arg.startsWith('--mode='));
const mode = modeArg ? modeArg.split('=')[1] : 'dev';
const isProd = mode === 'prod' || mode === 'production';

const envDir = path.join(
    projectRoot,
    '.venv',
    isProd ? 'py-scripts-prod' : 'py-scripts'
);

const pythonCandidates = [process.env.PYTHON, 'python3.10', 'python3.9', 'python3.8', 'python3.11', 'python3', 'python', 'py'];

function runCommand(command, commandArgs, options = {}) {
    const printable = `${command} ${commandArgs.join(' ')}`.trim();
    console.log(`\n$ ${printable}`);
    const result = spawnSync(command, commandArgs, {
        stdio: 'inherit',
        ...options,
    });

    if (result.error) {
        throw result.error;
    }

    if (result.status !== 0) {
        throw new Error(`Command failed with exit code ${result.status}: ${printable}`);
    }

    return result;
}

function resolveSystemPython() {
    for (const candidate of pythonCandidates) {
        if (!candidate) continue;
        try {
            const result = spawnSync(candidate, ['--version'], {
                stdio: 'pipe',
            });
            if (result.error || result.status !== 0) {
                continue;
            }
            const version = result.stdout.toString().trim() || result.stderr.toString().trim();
            console.log(`Using Python interpreter '${candidate}' (${version})`);
            return candidate;
        } catch (error) {
            continue; // Try next candidate
        }
    }
    console.error(
        'Unable to locate a working Python interpreter. Set the PYTHON environment variable to a valid python executable.'
    );
    process.exit(1);
}

function getVenvPython(envPath) {
    if (process.platform === 'win32') {
        return path.join(envPath, 'Scripts', 'python.exe');
    }
    return path.join(envPath, 'bin', 'python');
}

function ensureVenv(systemPython, envPath) {
    const venvConfig = path.join(envPath, 'pyvenv.cfg');
    if (fs.existsSync(venvConfig)) {
        console.log(`Virtual environment already present at ${envPath}`);
        return;
    }

    console.log(`Creating virtual environment at ${envPath}`);
    fs.mkdirSync(path.dirname(envPath), { recursive: true });
    runCommand(systemPython, ['-m', 'venv', envPath]);
}

function installRequirements(envPython) {
    runCommand(envPython, ['-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel']);
    runCommand(envPython, ['-m', 'pip', 'install', '-r', requirementsPath]);
}

function writeManifest(envPath) {
    const manifest = {
        mode,
        envDir: path.relative(projectRoot, envPath),
        updatedAt: new Date().toISOString(),
    };
    const manifestPath = path.join(envPath, 'manifest.json');
    fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
}

function main() {
    const systemPython = resolveSystemPython();
    ensureVenv(systemPython, envDir);
    const envPython = getVenvPython(envDir);

    if (!fs.existsSync(envPython)) {
        console.error(`Unable to locate python inside virtual environment: ${envPython}`);
        process.exit(1);
    }

    installRequirements(envPython);
    writeManifest(envDir);

    console.log('\n✅ Python environment ready');
    console.log(`   Mode: ${mode}`);
    console.log(`   Location: ${envDir}`);
    console.log(`   Interpreter: ${envPython}`);
}

main();
