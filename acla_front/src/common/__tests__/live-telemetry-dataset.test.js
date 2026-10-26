/** @jest-environment node */
const fs = require('fs');
const os = require('os');
const path = require('path');
const { EventEmitter } = require('events');
const { PassThrough } = require('stream');
const {
    LIVE_TELEMETRY_DATASET,
    assertLiveTelemetryRow,
    validateLiveTelemetryRow,
} = require('../../data/live-telemetry-dataset');
const { AccPythonReader } = require('../../../electron/recording/readers/acc/acc-python-reader');
const { IRacingReader } = require('../../../electron/recording/readers/iracing/iracing-reader');
const { RecordingWriter } = require('../../../electron/recording/workers/writer-worker');
const { RecordingView } = require('../../../electron/recording/workers/view-worker');
const { RecordedFileReader } = require('../../../electron/recording/workers/recorded-file-reader-worker');

const invalidRows = [
    {},
    { available: false },
    { Physics_speed_kmh: 100, speedKph: 120 },
    { Graphics_completed_laps: 1 },
    { Graphics: { completed_laps: 1 } },
    { Physics: { speed_kmh: 100 } },
    { Static: { track: 'monza' } },
    { Statics: { track: 'monza' } },
    { Physics_gear: 1.5 },
    { Physics_speed_kmh: null },
    { Physics_speed_kmh: Infinity },
    { Physics_pit_limiter_on: 1 },
    { Graphics_rain_tyres: true },
    { Graphics_car_coordinates: JSON.stringify(Array(60).fill({ x: 0, y: 0, z: 0 })) },
    { Graphics_car_id: [1, 2] },
    ...[[], null, '0.5', { '-1': 0.5 }, { '01': 0.5 }, { car: 0.5 }, { '1.5': 0.5 },
        { '9007199254740992': 0.5 }, { 0: -1 }, { 0: 1.01 }, { 0: NaN }, { 0: Infinity },
        { 0: null }, { 0: '0.5' }, { 0: { position: 0.5 } }]
        .map((value) => ({ Graphics_normalized_positions: value })),
];

describe('live telemetry dataset boundary', () => {
    it.each(['acc', 'iracing'])('stamps %s reader frames before live delivery and preserves their times on disk', async (game) => {
        const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'acla-session-time-'));
        const child = new EventEmitter();
        child.stdin = new PassThrough();
        child.stdout = new PassThrough();
        child.stderr = new PassThrough();
        child.kill = () => child.emit('close', 0, null);
        child.stdin.on('finish', () => setImmediate(() => child.emit('close', 0, null)));
        const lines = new EventEmitter();
        lines.close = jest.fn();
        let now = 1000;
        const Reader = game === 'acc' ? AccPythonReader : IRacingReader;
        const reader = new Reader({
            runtime: 'python', pythonExecutable: 'python', scriptDirectory: directory, scriptName: 'capture.py',
        }, { spawn: () => child, createInterface: () => lines, now: () => now });
        const writer = new RecordingWriter({
            game, recordingDirectory: directory, progressPort: { postMessage: jest.fn() }, parentSend: jest.fn(),
        });
        const events = [];
        const updates = [];
        const view = new RecordingView({
            game, updatesPort: { postMessage: (message) => updates.push(message) }, parentSend: jest.fn(),
        });
        let filePath;
        try {
            filePath = await writer.open();
            now = 2500;
            const started = reader.start((event) => {
                events.push(event);
                if (event.type === 'frame') view.acceptFrame(event.frame);
            });
            if (game === 'iracing') {
                child.stdout.write('{"type":"ready"}\n');
                await started;
            }
            const elapsedSeconds = [0.125, 1.25, 4, 5.5];
            for (const [tick, time] of [12.125, 12.125, undefined, 0].entries()) {
                now = 2500 + elapsedSeconds[tick] * 1000;
                const status = [2, 3, 0, 2][tick];
                if (game === 'acc') {
                    lines.emit('line', JSON.stringify({ type: 'sample',
                        sharedMemory: {
                            Physics: { velocity: { x: -4, y: 2, z: -9 } },
                            Graphics: { status, car_coordinates: Array(60).fill({ x: 10, y: 20, z: -30 }) },
                        },
                        broadcasting: { positions: {}, sessionTime: time === undefined ? null : time * 1000 },
                    }));
                } else {
                    child.stdout.write(`${JSON.stringify({ type: 'sample', tick,
                        values: { IsOnTrack: status !== 0, IsReplayPlaying: false, SessionTime: time } })}\n`);
                }
                await Promise.resolve();
                if (tick === 1) {
                    // Reconnecting the simulator must not restart the reader's clock.
                    if (game === 'acc') lines.emit('line', '{"available":false}');
                    else child.stdout.write('{"type":"disconnected"}\n');
                    await Promise.resolve();
                }
            }
            await started;
            await reader.stop();
            // Delayed writer receipt and disk flushing must preserve the reader's times.
            now = 15000;
            events.forEach((event) => writer.acceptFrame(event.frame));
            await writer.end();
            expect(events.map((event) => event.type)).toEqual(['frame', 'frame', 'frame', 'frame']);
            expect(events.map((event) => event.frame.sample.Graphics_analyst_session_time)).toEqual(elapsedSeconds);
            expect(updates.map((update) => update.sample)).toEqual(events.map((event) => event.frame.sample));
            const rows = fs.readFileSync(filePath, 'utf8').trim().split('\n').map(JSON.parse);
            expect(rows).toEqual(events.map((event) => event.frame.sample));
            if (game === 'acc') {
                expect(rows[0]).toMatchObject({
                    Physics_velocity_x: -4, Physics_velocity_y: -9, Physics_velocity_z: 2,
                    Graphics_car_coordinates: Array(60).fill({ x: 10, y: -30, z: 20 }),
                });
            }
            expect(rows[0].Graphics_game_time).toBe(12.125);
            expect(rows[1].Graphics_game_time).toBe(game === 'acc' ? undefined : 12.125);
            expect(rows[2]).not.toHaveProperty('Graphics_game_time');
            expect(rows[3].Graphics_game_time).toBe(0);
            expect(rows[0]).not.toHaveProperty('SessionTime');
        } finally {
            await reader.stop();
            await writer.end();
            if (filePath) fs.rmSync(filePath, { force: true });
            fs.rmdirSync(directory);
        }
    });

    it.each(invalidRows)('rejects invalid rows in validation, writer and live view: %o', (sample) => {
        expect(validateLiveTelemetryRow(sample).ok).toBe(false);
        expect(() => assertLiveTelemetryRow(sample)).toThrow(TypeError);
        const updates = { postMessage: jest.fn() };
        const writer = new RecordingWriter({
            game: 'acc', recordingDirectory: os.tmpdir(), progressPort: updates, parentSend: jest.fn(),
        });
        const view = new RecordingView({ game: 'acc', updatesPort: updates, parentSend: jest.fn() });
        writer.acceptFrame({ game: 'acc', sample });
        view.acceptFrame({ game: 'acc', sample });
        expect(writer.failed).toBe(true);
        expect(writer.sequence).toBe(0);
        expect(view.receivedSequence).toBe(0);
        expect(updates.postMessage).not.toHaveBeenCalledWith(expect.objectContaining({ type: 'frame' }));
    });

    it.each(['acc', 'iracing'])('preserves all registered fields through %s downstream consumers', async (game) => {
        const values = {
            boolean: false, integer: 0, number: 0.25, string: '',
            coordinates: Array.from({ length: 60 }, (_, i) => ({ x: i, y: i + 1, z: -i - 1 })),
            'integer-array': Array.from({ length: 60 }, (_, i) => i),
            'normalized-positions': { 0: 0, 63: 1, 1052: 0.25 },
        };
        const sample = Object.fromEntries(Object.entries(LIVE_TELEMETRY_DATASET)
            .map(([field, type]) => [field, values[type]]));
        expect(assertLiveTelemetryRow(sample)).toBe(sample);
        const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'acla-dataset-'));
        try {
            const updates = { postMessage: jest.fn() };
            const writer = new RecordingWriter({
                game, recordingDirectory: directory, progressPort: updates, parentSend: jest.fn(),
            });
            const view = new RecordingView({ game, updatesPort: updates, parentSend: jest.fn() });
            const filePath = await writer.open();
            writer.acceptFrame({ game, sample });
            view.acceptFrame({ game, sample });
            await writer.end();
            expect(fs.readFileSync(filePath, 'utf8')).toBe(`${JSON.stringify(sample)}\n`);
            expect(updates.postMessage).toHaveBeenCalledWith(expect.objectContaining({ type: 'frame', sample }));

            const rows = [];
            let reader;
            reader = new RecordedFileReader({
                readId: 'dataset-roundtrip', filePath, game, purpose: 'consume', recordingDirectory: directory,
                parentSend: jest.fn(),
                eventPort: {
                    postMessage: (event) => {
                        if (event.type === 'chunk') {
                            rows.push(...event.rows);
                            setImmediate(() => reader.acknowledgeChunk(event.chunkIndex));
                        }
                    },
                    close: jest.fn(),
                },
            });
            await reader.start();
            expect(rows).toEqual([sample]);
        } finally {
            fs.rmSync(directory, { recursive: true, force: true });
        }
    });

    it('fails file reads on legacy rows without publishing a successful completion', async () => {
        const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'acla-invalid-dataset-'));
        const filePath = path.join(directory, 'acc-invalid.jsonl');
        fs.writeFileSync(filePath, '{"Physics_speed_kmh":100}\n{"speedKph":120}\n');
        const events = [];
        try {
            const reader = new RecordedFileReader({
                readId: 'invalid-dataset', filePath, game: 'acc', purpose: 'consume', recordingDirectory: directory,
                parentSend: jest.fn(),
                eventPort: { postMessage: (event) => events.push(event), close: jest.fn() },
            });
            await reader.start();
            expect(events).toContainEqual(expect.objectContaining({ type: 'error', row: 2 }));
            expect(events.some((event) => event.type === 'complete')).toBe(false);
        } finally {
            fs.rmSync(directory, { recursive: true, force: true });
        }
    });

    it('adapts raw ACC captures to dataset rows and keeps control packets outside them', async () => {
        const child = new EventEmitter();
        child.stdout = new PassThrough();
        child.stderr = new PassThrough();
        child.kill = () => child.emit('close', 0, null);
        const lines = new EventEmitter();
        lines.close = jest.fn();
        const reader = new AccPythonReader({
            runtime: 'python', pythonExecutable: 'python', scriptDirectory: os.tmpdir(), scriptName: 'capture.py',
        }, { spawn: () => child, createInterface: () => lines });
        const emit = jest.fn();
        const started = reader.start(emit);
        lines.emit('line', '{"available":false}');
        expect(emit).not.toHaveBeenCalled();
        lines.emit('line', JSON.stringify({ type: 'sample',
            sharedMemory: { Physics: { speed_kmh: 120 }, Graphics: { status: 2 } },
            broadcasting: { positions: { 1052: 0.25 }, sessionTime: 12125 },
        }));
        await started;
        expect(emit).toHaveBeenCalledWith({ type: 'frame', frame: {
            game: 'acc', sample: { Physics_speed_kmh: 120, Graphics_status: 2, Graphics_normalized_positions: { 1052: 0.25 },
                Graphics_game_time: 12.125, Graphics_analyst_session_time: expect.any(Number) },
        } });
        lines.emit('line', JSON.stringify({ type: 'sample',
            sharedMemory: { Physics: { speed_kmh: 121, speedKph: 121 } }, broadcasting: {},
        }));
        expect(emit.mock.calls.filter(([event]) => event.type === 'frame')).toHaveLength(1);
        expect(emit).toHaveBeenLastCalledWith(expect.objectContaining({ type: 'fatal' }));
        await reader.stop();
    });
});
