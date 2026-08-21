const fs = require('fs');
const os = require('os');
const path = require('path');
const {
    FIELD_TYPES,
    STANDARD_TELEMETRY_FIELDS,
    validateSourceFrame,
    validateStandardTelemetrySample,
} = require('../../../electron/recording/telemetry-contract');
const {
    recordingStartFailure,
    validateRecordingStartConfig,
} = require('../../../electron/recording/recording-protocol');
const {
    RecordingSessionManager,
    workerLifecycleLogMessage,
} = require('../../../electron/recording/recording-session-manager');
const { RecordingWriter } = require('../../../electron/recording/workers/writer-worker');
const { RecordingView } = require('../../../electron/recording/workers/view-worker');
const { RecordedFileReader } = require('../../../electron/recording/workers/recorded-file-reader-worker');

describe('shared recording architecture', () => {
    it('uses clear console messages for recording worker lifecycle events', () => {
        expect(workerLifecycleLogMessage('reader', 'started')).toBe('[recording] collect worker started.');
        expect(workerLifecycleLogMessage('reader', 'ended')).toBe('[recording] collect worker ended.');
        expect(workerLifecycleLogMessage('writer', 'started')).toBe('[recording] write worker started.');
        expect(workerLifecycleLogMessage('writer', 'ended')).toBe('[recording] write worker ended.');
        expect(workerLifecycleLogMessage('view', 'started')).toBe('[recording] view worker started.');
        expect(workerLifecycleLogMessage('view', 'ended')).toBe('[recording] view worker ended.');
    });

    it('enforces the exhaustive standard flat telemetry contract', () => {
        expect(STANDARD_TELEMETRY_FIELDS).toHaveLength(240);
        expect(Object.keys(FIELD_TYPES)).toHaveLength(240);
        expect(validateStandardTelemetrySample({
            Physics_speed_kmh: 120.5,
            Graphics_completed_lap: 2,
            Graphics_status: 1,
            Static_track: 'Spa',
        })).toEqual(expect.objectContaining({ ok: true }));
        expect(validateStandardTelemetrySample({ speedKph: 120.5 })).toEqual(expect.objectContaining({ ok: false }));
        expect(validateStandardTelemetrySample({ Graphics_car_id: [1, 2] })).toEqual(expect.objectContaining({ ok: false }));
        expect(validateSourceFrame({ game: 'acc', sample: { Graphics_status: 0 } }, 'acc'))
            .toEqual(expect.objectContaining({ ok: true }));
        expect(validateSourceFrame({ game: 'iracing', sample: { Graphics_status: 0 } }, 'acc'))
            .toEqual(expect.objectContaining({ ok: false }));
    });

    it('matches every field name and type in the documented 240-field catalog', () => {
        const catalog = fs.readFileSync(path.join(__dirname, '../../../tmp/telemetry-fields.md'), 'utf8');
        const documented = Object.fromEntries(Array.from(
            catalog.matchAll(/^(Physics|Graphics|Static)_[A-Za-z0-9_]+\s{2,}([^\r\n]+)$/gm),
            (match) => {
                const field = match[0].trim().split(/\s+/)[0];
                const documentedType = match[2].trim().replace(/\s+\(.+\)$/, '');
                const type = documentedType === 'integer[60]'
                    ? 'integer-array'
                    : documentedType.startsWith('array<') ? 'coordinates' : documentedType;
                return [field, type];
            },
        ));

        expect(Object.keys(documented)).toHaveLength(240);
        expect(FIELD_TYPES).toEqual(documented);
    });

    it('returns transport-safe discriminated failures for malformed and unknown games', () => {
        expect(validateRecordingStartConfig({})).toEqual({
            ok: false,
            error: expect.objectContaining({ type: 'malformed-recording-game' }),
        });
        expect(validateRecordingStartConfig({ game: 'not-a-game' })).toEqual({
            ok: false,
            error: expect.objectContaining({ type: 'unknown-recording-game' }),
        });
        expect(validateRecordingStartConfig({ game: 'acc' })).toBeNull();
    });

    it('keeps manager construction and unsupported resolution side-effect free', async () => {
        const fork = jest.fn();
        const channel = jest.fn();
        const getMainWindow = jest.fn();
        const getReaderLaunchConfig = jest.fn().mockResolvedValue(
            recordingStartFailure('unsupported-recording-game', 'Coming soon.'),
        );
        const manager = new RecordingSessionManager({
            utilityProcess: { fork },
            MessageChannelMain: channel,
            getMainWindow,
            getReaderLaunchConfig,
            recordingDirectory: path.join(os.tmpdir(), 'acla-recording-manager-test'),
        });

        expect(fork).not.toHaveBeenCalled();
        expect(channel).not.toHaveBeenCalled();
        expect(getMainWindow).not.toHaveBeenCalled();
        expect(getReaderLaunchConfig).not.toHaveBeenCalled();

        await expect(manager.startSession({ game: 'iracing', ownerWebContentsId: 7 })).resolves.toEqual({
            ok: false,
            error: expect.objectContaining({ type: 'unsupported-recording-game' }),
        });
        expect(fork).not.toHaveBeenCalled();
        expect(channel).not.toHaveBeenCalled();
        expect(getMainWindow).not.toHaveBeenCalled();
    });

    it('writes only the unchanged sample and keeps commit metadata in memory', async () => {
        const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'acla-writer-'));
        const progress = [];
        const parent = [];
        const writer = new RecordingWriter({
            game: 'acc',
            recordingDirectory: directory,
            progressPort: { postMessage: (message) => progress.push(message) },
            parentSend: (message) => parent.push(message),
        });
        try {
            const filePath = await writer.open();
            const samples = [
                { Physics_speed_kmh: 100, Graphics_status: 2 },
                { Physics_speed_kmh: 101, Graphics_status: 3 },
            ];
            samples.forEach((sample) => writer.acceptFrame({ game: 'acc', sample }));
            const result = await writer.end();
            const rows = fs.readFileSync(filePath, 'utf8').trim().split('\n').map(JSON.parse);

            expect(rows).toEqual(samples);
            expect(rows[0]).not.toHaveProperty('game');
            expect(rows[0]).not.toHaveProperty('sequence');
            expect(result).toEqual({ filePath, writtenSamples: 2 });
            expect(progress).toEqual(expect.arrayContaining([
                expect.objectContaining({ type: 'committed', fromSequence: 1, toSequence: 2, committedCount: 2 }),
                expect.objectContaining({ type: 'final', writtenSamples: 2 }),
            ]));
            expect(parent).toEqual([
                expect.objectContaining({ type: 'committed', fromSequence: 1, toSequence: 2, committedCount: 2 }),
                expect.objectContaining({ type: 'finalized', writtenSamples: 2 }),
            ]);
        } finally {
            fs.rmSync(directory, { recursive: true, force: true });
        }
    });

    it('publishes every live frame immediately without batching', () => {
        const updates = [];
        const view = new RecordingView({
            game: 'acc',
            updatesPort: { postMessage: (message) => updates.push(message) },
            parentSend: jest.fn(),
        });
        const samples = [
            { Physics_speed_kmh: 100, Graphics_status: 2 },
            { Physics_speed_kmh: 101, Graphics_status: 2 },
        ];

        view.acceptFrame({ game: 'acc', sample: samples[0] });
        expect(updates).toEqual([expect.objectContaining({
            type: 'frame',
            sample: samples[0],
            sequence: 1,
        })]);

        view.acceptFrame({ game: 'acc', sample: samples[1] });
        expect(updates).toEqual([
            expect.objectContaining({ type: 'frame', sample: samples[0], sequence: 1 }),
            expect.objectContaining({ type: 'frame', sample: samples[1], sequence: 2 }),
        ]);
    });

    it('holds each saved-file chunk until its consumer acknowledges it', async () => {
        const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'acla-reader-'));
        const filePath = path.join(directory, 'acc-flow-control.jsonl');
        const rows = Array.from({ length: 251 }, (_, index) => ({ Physics_speed_kmh: index }));
        fs.writeFileSync(filePath, `${rows.map(JSON.stringify).join('\n')}\n`, 'utf8');
        const events = [];
        const parentEvents = [];
        let reader;
        const eventPort = {
            postMessage: (message) => {
                events.push(message);
                if (message.type === 'chunk') {
                    setTimeout(() => reader.acknowledgeChunk(message.chunkIndex), 0);
                }
            },
            close: jest.fn(),
        };
        reader = new RecordedFileReader({
            readId: 'flow-control-read',
            filePath,
            game: 'acc',
            purpose: 'consume',
            recordingDirectory: directory,
            eventPort,
            parentSend: (message) => parentEvents.push(message),
        });

        try {
            await reader.start();
            expect(events.filter((event) => event.type === 'chunk').map((event) => event.rows.length))
                .toEqual([250, 1]);
            expect(events.at(-1)).toEqual(expect.objectContaining({ type: 'complete', rowCount: 251 }));
            expect(parentEvents).toEqual([expect.objectContaining({ type: 'complete' })]);
        } finally {
            fs.rmSync(directory, { recursive: true, force: true });
        }
    });

    it('reads exactly a fixed committed prefix and ignores an incomplete trailing append', async () => {
        const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'acla-active-reader-'));
        const filePath = path.join(directory, 'acc-active.jsonl');
        const committedRows = [
            { Physics_speed_kmh: 100 },
            { Physics_speed_kmh: 101 },
        ];
        fs.writeFileSync(
            filePath,
            `${committedRows.map(JSON.stringify).join('\n')}\n{"Physics_speed_kmh":`,
            'utf8',
        );
        const events = [];
        let reader;
        const eventPort = {
            postMessage: (message) => {
                events.push(message);
                if (message.type === 'chunk') {
                    setTimeout(() => reader.acknowledgeChunk(message.chunkIndex), 0);
                }
            },
            close: jest.fn(),
        };
        reader = new RecordedFileReader({
            readId: 'active-prefix-read',
            filePath,
            game: 'acc',
            purpose: 'consume',
            recordingDirectory: directory,
            rowLimit: 2,
            eventPort,
            parentSend: jest.fn(),
        });

        try {
            await reader.start();
            expect(events.filter((event) => event.type === 'chunk').flatMap((event) => event.rows))
                .toEqual(committedRows);
            expect(events.at(-1)).toEqual(expect.objectContaining({ type: 'complete', rowCount: 2 }));
            expect(events.some((event) => event.type === 'error')).toBe(false);
        } finally {
            fs.rmSync(directory, { recursive: true, force: true });
        }
    });

    it('authorizes active-file snapshots only for the owning renderer, game, and file', () => {
        const directory = path.join(os.tmpdir(), 'acla-active-auth');
        const activeFilePath = path.join(directory, 'acc-active.jsonl');
        const manager = new RecordingSessionManager({
            utilityProcess: { fork: jest.fn() },
            MessageChannelMain: jest.fn(),
            getMainWindow: jest.fn(),
            getReaderLaunchConfig: jest.fn(),
            recordingDirectory: directory,
        });
        manager.active = {
            status: 'running',
            ownerWebContentsId: 7,
            game: 'acc',
            activeFilePath,
            committedRowCount: 12,
        };

        expect(manager.getActiveRecordedFileReadLimit({
            ownerWebContentsId: 7,
            game: 'acc',
            filePath: activeFilePath,
        })).toBe(12);
        expect(() => manager.getActiveRecordedFileReadLimit({
            ownerWebContentsId: 8,
            game: 'acc',
            filePath: activeFilePath,
        })).toThrow('another renderer');
        expect(() => manager.getActiveRecordedFileReadLimit({
            ownerWebContentsId: 7,
            game: 'iracing',
            filePath: activeFilePath,
        })).toThrow('game');
        expect(() => manager.getActiveRecordedFileReadLimit({
            ownerWebContentsId: 7,
            game: 'acc',
            filePath: path.join(directory, 'another.jsonl'),
        })).toThrow('active writer file');
    });
});
