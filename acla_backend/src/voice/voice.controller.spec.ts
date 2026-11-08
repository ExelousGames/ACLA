import { INestApplication } from '@nestjs/common';
import { JwtModule, JwtService } from '@nestjs/jwt';
import { Test } from '@nestjs/testing';
import axios from 'axios';
import * as request from 'supertest';
import { App } from 'supertest/types';
import { jwtConstants } from '../common/constants';
import { JwtStrategy } from '../common/jwt.strategy';
import { UserInfoService } from '../modules/user-info/user-info.service';
import { VoiceGateway } from './voice.gateway';
import { VoiceModule } from './voice.module';

describe('Live Phrase voice API', () => {
    let app: INestApplication<App>;
    let token: string;
    let post: jest.SpyInstance;
    const previousAiServiceUrl = process.env.AI_SERVICE_URL;
    const aiServiceUrl = 'http://ai-service.test:8000';
    const preview = {
        phonemes: 'aˈɪ b!',
        tokens: Array.from('aˈɪ b!', (symbol, index) => ({ index, symbol, token_id: index + 10 })),
        sample_rate: 24000,
        max_phonemes: 510,
    };
    const synthesis = {
        text: 'Aye, brake!',
        voice: 'af_bella',
        speed: 1.2,
        language: 'en-gb',
        prosody: {
            expected_phonemes: preview.phonemes,
            curves: [{
                start: 0,
                end: 3,
                timing: [{ position: 0, value: 0.5 }, { position: 1, value: 2 }],
                energy: [{ position: 0, value: -2 }, { position: 1, value: 2 }],
                pitch: [{ position: 0, value: 0 }, { position: 1, value: 3 }],
            }],
        },
    };

    beforeAll(async () => {
        process.env.AI_SERVICE_URL = aiServiceUrl;
        const module = await Test.createTestingModule({
            imports: [
                VoiceModule,
                JwtModule.register({ global: true, secret: jwtConstants.secret }),
            ],
            providers: [
                JwtStrategy,
                { provide: UserInfoService, useValue: { findOneWithPermissions: jest.fn().mockResolvedValue({}) } },
            ],
        })
            .overrideProvider(VoiceGateway)
            .useValue({})
            .compile();
        app = module.createNestApplication();
        await app.init();
        token = module.get(JwtService).sign({ id: 'user-1', username: 'driver' });
    });

    beforeEach(() => {
        post = jest.spyOn(axios, 'post');
    });

    afterEach(() => jest.restoreAllMocks());

    afterAll(async () => {
        await app?.close();
        if (previousAiServiceUrl === undefined) delete process.env.AI_SERVICE_URL;
        else process.env.AI_SERVICE_URL = previousAiServiceUrl;
    });

    it.each(['/voice/phonemes', '/voice/synthesize/prosody'])(
        'requires a valid JWT for %s before contacting the AI service',
        async (path) => {
            await request(app.getHttpServer()).post(path).send(synthesis).expect(401);
            await request(app.getHttpServer()).post(path)
                .set('Authorization', 'Bearer invalid').send(synthesis).expect(401);
            expect(post).not.toHaveBeenCalled();
        },
    );

    it.each([{ text: 'Aye, brake!' }, { text: ' Aye, brake! ', language: 'en-gb' }])(
        'returns all indexed phonemes and forwards the preview payload unchanged: %p',
        async (body) => {
            post.mockResolvedValue({ data: preview });
            const response = await request(app.getHttpServer()).post('/voice/phonemes')
                .set('Authorization', `Bearer ${token}`).send(body).expect(200);
            expect(response.body).toEqual(preview);
            expect(post).toHaveBeenCalledWith(`${aiServiceUrl}/voice/phonemes`, body);
        },
    );

    it.each([synthesis, { text: synthesis.text, prosody: { expected_phonemes: preview.phonemes } }])(
        'forwards prosody controls and optional defaults and returns unmodified WAV bytes: %p',
        async (body) => {
            const wav = Buffer.concat([Buffer.from('RIFF'), Buffer.from([0x00, 0xff, 0x80, 0x01]), Buffer.from('WAVE')]);
            post.mockResolvedValue({ data: wav });
            const response = await request(app.getHttpServer()).post('/voice/synthesize/prosody')
                .set('Authorization', `Bearer ${token}`).send(body).expect(200);
            expect(post).toHaveBeenCalledWith(`${aiServiceUrl}/voice/synthesize/prosody`, body, { responseType: 'arraybuffer' });
            expect(response.headers['content-type']).toBe('audio/wav');
            expect(response.headers['content-length']).toBe(String(wav.length));
            expect(response.headers['content-disposition']).toBe('inline; filename="speech.wav"');
            expect(response.headers['cache-control']).toBe('no-store');
            expect(response.body).toEqual(wav);
        },
    );

    it.each([
        { path: '/voice/phonemes', status: 400, detail: 'text must be non-empty', body: { text: ' ' } },
        { path: '/voice/phonemes', status: 422, detail: [{ loc: ['body', 'text'], msg: 'Field required', type: 'missing' }], body: {} },
        { path: '/voice/synthesize/prosody', status: 400, detail: 'Phoneme sequence changed', body: synthesis },
        { path: '/voice/synthesize/prosody', status: 422, detail: [{ loc: ['body', 'prosody'], msg: 'Field required', type: 'missing' }], body: { text: 'Brake!' } },
        { path: '/voice/synthesize/prosody', status: 500, detail: 'Prosody synthesis failed', body: synthesis },
    ])('preserves AI service status and JSON errors for $path ($status)', async ({ path, status, detail, body }) => {
        const errorBody = { detail };
        post.mockRejectedValue({
            isAxiosError: true,
            response: { status, data: path.endsWith('/prosody') ? Buffer.from(JSON.stringify(errorBody)) : errorBody },
        });
        const response = await request(app.getHttpServer()).post(path)
            .set('Authorization', `Bearer ${token}`).send(body).expect(status);
        expect(response.headers['content-type']).toMatch(/application\/json/);
        expect(response.body).toEqual(errorBody);
    });

    it.each(['/voice/phonemes', '/voice/synthesize/prosody'])(
        'returns 503 when the AI service cannot be reached for %s',
        async (path) => {
            post.mockRejectedValue({ isAxiosError: true, code: 'ECONNREFUSED' });
            await request(app.getHttpServer()).post(path)
                .set('Authorization', `Bearer ${token}`).send(synthesis).expect(503);
        },
    );

    it('handles non-JSON upstream failures without returning an audio buffer as an error', async () => {
        post.mockRejectedValue({
            isAxiosError: true,
            response: { status: 502, data: Buffer.from('<html>Bad gateway</html>') },
        });
        const response = await request(app.getHttpServer()).post('/voice/synthesize/prosody')
            .set('Authorization', `Bearer ${token}`).send(synthesis).expect(502);
        expect(response.body).toEqual({ statusCode: 502, message: 'AI Service prosody synthesis failed' });
    });
});
