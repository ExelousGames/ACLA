import { Test, TestingModule } from '@nestjs/testing';
import { RacingSessionController } from './racing-session.controller';
import { RacingSessionService } from './racing-session.service';
import { UserSessionAiModelService } from '../user-session-ai-model/user-session-ai-model.service';
import { AiServiceClient } from 'src/shared/ai/ai-service.client';
import { UserInfoService } from '../user-info/user-info.service';
import { ForbiddenException } from '@nestjs/common';

describe('RacingSessionController', () => {
  let controller: RacingSessionController;
  let racingSessionService: any;
  let aiServiceClient: any;

  beforeEach(async () => {
    racingSessionService = {
      listUserSessionsForAnalysis: jest.fn(),
      getSessionTelemetryForClassification: jest.fn(),
    };
    aiServiceClient = {
      classifySegments: jest.fn(),
    };

    const module: TestingModule = await Test.createTestingModule({
      controllers: [RacingSessionController],
      providers: [
        { provide: RacingSessionService, useValue: racingSessionService },
        { provide: UserSessionAiModelService, useValue: {} },
        { provide: AiServiceClient, useValue: aiServiceClient },
        { provide: UserInfoService, useValue: {} },
      ],
    }).compile();

    controller = module.get<RacingSessionController>(RacingSessionController);
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });

  it('blocks analysis metadata for another user', async () => {
    await expect(
      controller.initializeUserSessionsAnalysis(
        { user: { userId: 'user-1', username: 'driver@example.com' } },
        { userId: 'user-2' },
      ),
    ).rejects.toBeInstanceOf(ForbiddenException);
  });

  it('allows the AI service account to request analysis metadata for a target user', async () => {
    process.env.AI_SERVICE_USERNAME = 'ai@example.com';
    racingSessionService.listUserSessionsForAnalysis.mockResolvedValue([]);

    await expect(
      controller.initializeUserSessionsAnalysis(
        { user: { userId: 'service-user', username: 'ai@example.com' } },
        { userId: 'user-1', sessionLimit: 10 },
      ),
    ).resolves.toEqual({ userId: 'user-1', totalSessions: 0, sessions: [] });
    expect(racingSessionService.listUserSessionsForAnalysis).toHaveBeenCalledWith('user-1', 10);
  });

  it('loads and forwards a saved session for segment classification', async () => {
    racingSessionService.getSessionTelemetryForClassification.mockResolvedValue({
      sessionId: 'session-1',
      trackName: 'Brands Hatch',
      carName: 'BMW',
      telemetryData: [{ speed: 120 }],
    });
    aiServiceClient.classifySegments.mockResolvedValue({
      status: 'success',
      session_id: 'session-1',
      samples_analyzed: 1,
      segment_count: 1,
      segments: [{
        id: 'segment-1',
        labels: ['EA'],
        main_label_id: 'EA',
        start_index: 0,
        end_index: 1,
        sub_labels: [],
        sub_segments: [{ start_index: 0, end_index: 1, labels: [] }],
      }],
    });

    await expect(
      controller.classifySessionSegments(
        { user: { userId: 'user-1' } },
        { session_id: 'session-1' },
      ),
    ).resolves.toEqual({
      status: 'success',
      session_id: 'session-1',
      samples_analyzed: 1,
      segment_count: 1,
      segments: [{
        id: 'segment-1',
        labels: ['EA'],
        main_label_id: 'EA',
        start_index: 0,
        end_index: 1,
        sub_labels: [],
        sub_segments: [{ start_index: 0, end_index: 1, labels: [] }],
      }],
    });

    expect(racingSessionService.getSessionTelemetryForClassification).toHaveBeenCalledWith('user-1', 'session-1');
    expect(aiServiceClient.classifySegments).toHaveBeenCalledWith({
      session_id: 'session-1',
      telemetry_data: [{ speed: 120 }],
      track_name: 'Brands Hatch',
      car_name: 'BMW',
    });
  });
});
