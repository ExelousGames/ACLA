import { Test, TestingModule } from '@nestjs/testing';
import { UserInfoController } from './user-info.controller';
import { UserInfoService } from './user-info.service';
import { AuthService } from 'src/shared/auth/auth.service';
import { BadRequestException } from '@nestjs/common';
import { AuthorizationService } from 'src/shared/authorization/authorization.service';
import { UserSummaryAnalysisService } from './user-summary-analysis.service';

describe('UserInfoController', () => {
  let controller: UserInfoController;
  let userInfoService: {
    getUserSummary: jest.Mock;
    updateUserSummary: jest.Mock;
  };
  let userSummaryAnalysisService: {
    enqueue: jest.Mock;
    getStatus: jest.Mock;
  };

  beforeEach(async () => {
    userInfoService = {
      getUserSummary: jest.fn(),
      updateUserSummary: jest.fn(),
    };
    userSummaryAnalysisService = {
      enqueue: jest.fn(),
      getStatus: jest.fn(),
    };

    const module: TestingModule = await Test.createTestingModule({
      controllers: [UserInfoController],
      providers: [
        { provide: UserInfoService, useValue: userInfoService },
        { provide: UserSummaryAnalysisService, useValue: userSummaryAnalysisService },
        { provide: AuthService, useValue: {} },
        {
          provide: AuthorizationService,
          useValue: {
            hasPermissions: jest.fn().mockReturnValue(true),
            hasAnyRole: jest.fn().mockReturnValue(true),
            hasAllRoles: jest.fn().mockReturnValue(true),
          },
        },
      ],
    }).compile();

    controller = module.get<UserInfoController>(UserInfoController);
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });

  it('gets the authenticated user summary', async () => {
    userInfoService.getUserSummary.mockResolvedValue({ pace: 'steady' });

    await expect(controller.getUserSummary({ user: { userId: 'user-1' } })).resolves.toEqual({
      summary: { pace: 'steady' },
    });
    expect(userInfoService.getUserSummary).toHaveBeenCalledWith('user-1');
  });

  it('updates the authenticated user summary', async () => {
    userInfoService.updateUserSummary.mockResolvedValue({ braking: 'late' });

    await expect(
      controller.updateUserSummary({ user: { userId: 'user-1' } }, { summary: { braking: 'late' } }),
    ).resolves.toEqual({ summary: { braking: 'late' } });
    expect(userInfoService.updateUserSummary).toHaveBeenCalledWith('user-1', { braking: 'late' });
  });

  it('rejects update without summary data', async () => {
    await expect(
      controller.updateUserSummary({ user: { userId: 'user-1' } }, {} as any),
    ).rejects.toBeInstanceOf(BadRequestException);
  });

  it('queues analysis for the authenticated user', async () => {
    userSummaryAnalysisService.enqueue.mockResolvedValue({ id: 'job-1', status: 'queued' });

    await expect(controller.analyzeAllUserSessions({ user: { userId: 'user-1' } })).resolves.toEqual({
      id: 'job-1',
      status: 'queued',
    });
    expect(userSummaryAnalysisService.enqueue).toHaveBeenCalledWith('user-1');
  });

  it('gets analysis status for the authenticated user', async () => {
    userSummaryAnalysisService.getStatus.mockResolvedValue({ id: 'job-1', status: 'running' });

    await expect(controller.getAnalyzeAllUserSessionsStatus({ user: { userId: 'user-1' } })).resolves.toEqual({
      id: 'job-1',
      status: 'running',
    });
    expect(userSummaryAnalysisService.getStatus).toHaveBeenCalledWith('user-1');
  });
});
