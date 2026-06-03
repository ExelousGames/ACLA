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

  beforeEach(async () => {
    racingSessionService = {
      listUserSessionsForAnalysis: jest.fn(),
    };

    const module: TestingModule = await Test.createTestingModule({
      controllers: [RacingSessionController],
      providers: [
        { provide: RacingSessionService, useValue: racingSessionService },
        { provide: UserSessionAiModelService, useValue: {} },
        { provide: AiServiceClient, useValue: {} },
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
        { userId: 'user-1' },
      ),
    ).resolves.toEqual({ userId: 'user-1', totalSessions: 0, sessions: [] });
  });
});
