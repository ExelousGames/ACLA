# ACLA Frontend & Desktop Companion

This package contains the React UI and the Electron desktop shell that executes all Python automation located in `src/py-scripts/`.

## Prerequisites

- Node.js 18+
- npm 9+
- Python 3.10 or newer (system interpreter used to create virtual environments)
- FFmpeg (required for Whisper-based speech recognition; optional otherwise)

## Python environments for desktop scripts

All Electron-only Python entry points live under `src/py-scripts/`. We manage their dependencies with
virtual environments so that development and production builds are reproducible.

| Mode | Virtual environment location | How it is created |
| --- | --- | --- |
| Development | `acla_front/.venv/py-scripts` | `npm run start:electron` (via pre script) or `npm run setup:python -- --mode=dev` |
| Production (packaged app) | `acla_front/.venv/py-scripts-prod` (bundled as `resources/python-env`) | `npm run build:electron` (via pre script) or `npm run setup:python -- --mode=prod` |

The managed environments install everything from `src/py-scripts/requirements.txt` including
scikit-learn, pandas, speech recognition, and audio tooling.

### One-time manual setup

```bash
# Install Node dependencies
npm install

# Prepare the development Python environment (optional – start:electron runs it automatically)
npm run setup:python -- --mode=dev
```

> **Tip:** Set `PYTHON=/path/to/python3` if your system Python command is named differently.

### Run the desktop app in development

```bash
npm run start:electron
```

The pre-script will:

1. Create/refresh `.venv/py-scripts`
2. Install every package in `src/py-scripts/requirements.txt`
3. Run the enhanced speech setup script
4. Launch the React dev server and Electron shell

### Build the production bundle

```bash
npm run build:electron
```

This command:

1. Creates/refreshes the production env in `.venv/py-scripts-prod`
2. Installs requirements inside that env
3. Runs the speech dependency installer
4. Builds the React bundle and packages Electron via `electron-builder`

The packaged app ships with the Python runtime under `resources/python-env`. The Electron
main process automatically selects that interpreter; override it with the environment variable
`ACLA_PYTHON_PATH=/absolute/path/to/python` if needed.

### Additional scripts

- `npm run setup:python` — Run the Python environment bootstrap in on-demand mode.
- `npm run setup-speech` — Install/verify speech recognition dependencies (used by the pre-scripts).
- `npm run electron` — Launch Electron pointing at the compiled build output.

### Troubleshooting

- Delete `acla_front/.venv/` and re-run `npm run setup:python -- --mode=dev` if packages become inconsistent.
- If FFmpeg or system audio drivers are missing, the speech setup script will print manual installation hints.
- On Windows, ensure that Visual C++ Build Tools are installed for PyAudio.
