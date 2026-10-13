const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const localPython = path.resolve(__dirname, '../.venv/track-vision', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const python = process.env.VISION_PYTHON || (fs.existsSync(localPython) ? localPython : 'python');
const result = spawnSync(python, [path.join(__dirname, 'export-vision-models.py')], { stdio: 'inherit' });
if (result.error) console.error(result.error.message);
if (result.status !== 0) console.error('Install scripts/vision-requirements.txt in your Python environment, then retry. VISION_PYTHON can select an interpreter.');
process.exitCode = result.status ?? 1;
