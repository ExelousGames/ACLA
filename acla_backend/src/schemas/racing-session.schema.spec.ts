import { model } from 'mongoose';
import { RacingSessionSchema } from './racing-session.schema';

describe('Game-neutral racing sessions', () => {
    const SessionModel = model('GameNeutralRacingSession', RacingSessionSchema);
    const payload = {
        session_name: 'Test Session',
        map: 'Test Circuit',
        car_name: 'Test Car',
        user_id: 'user-1',
    };

    it.each(['acc', 'ac', 'iracing', 'iracing_live', 'iracing_recorded', 'forza', 'custom-simulator'])(
        'validates and preserves %s as source metadata', (game) => {
            const document = new SessionModel({ ...payload, game_recorded_from: game });
            expect(document.validateSync()).toBeUndefined();
            expect(document.game_recorded_from).toBe(game);
        },
    );

    it.each([0, 1])('allows a session with %s videos', (count) => {
        const video = { name: 'video.mp4', size: 4, mimeType: 'video/mp4', startedAt: 1, chunkFileIds: [] };
        const document = new SessionModel({ ...payload, game_recorded_from: 'acc', videos: Array(count).fill(video) });
        expect(document.validateSync()).toBeUndefined();
    });

    it('rejects storing multiple videos in a session', () => {
        const video = { name: 'video.mp4', size: 4, mimeType: 'video/mp4', startedAt: 1, chunkFileIds: [] };
        const document = new SessionModel({ ...payload, game_recorded_from: 'acc', videos: [video, video] });
        expect(document.validateSync()?.errors.videos.message).toBe('Only one video is allowed per session.');
    });

    it('requires an explicit game instead of assigning a default', () => {
        const document = new SessionModel(payload);
        expect(document.game_recorded_from).toBeUndefined();
        expect(document.validateSync()?.errors.game_recorded_from).toBeDefined();
    });
});
