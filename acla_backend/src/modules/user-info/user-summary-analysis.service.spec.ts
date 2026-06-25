import { ConflictException } from '@nestjs/common';
import { Test, TestingModule } from '@nestjs/testing';
import { getModelToken } from '@nestjs/mongoose';
import { UserSummaryAnalysisService } from './user-summary-analysis.service';
import { UserInfoService } from './user-info.service';
import { AiServiceClient } from 'src/shared/ai/ai-service.client';
import { UserSummaryAnalysisJob } from 'src/schemas/user-summary-analysis-job.schema';

const execResult = (value: any) => ({ exec: jest.fn().mockResolvedValue(value) });

describe('UserSummaryAnalysisService', () => {
  let service: UserSummaryAnalysisService;
  let jobModel: any;
  let userInfoService: any;
  let aiServiceClient: any;

  beforeEach(async () => {
    jobModel = {
      findOne: jest.fn(),
      create: jest.fn(),
      findOneAndUpdate: jest.fn(),
      findByIdAndUpdate: jest.fn(),
    };
    userInfoService = {
      getUserSummary: jest.fn(),
      updateUserSummary: jest.fn(),
    };
    aiServiceClient = {
      analyzeUserSessions: jest.fn(),
    };

    const module: TestingModule = await Test.createTestingModule({
      providers: [
        UserSummaryAnalysisService,
        { provide: getModelToken(UserSummaryAnalysisJob.name), useValue: jobModel },
        { provide: UserInfoService, useValue: userInfoService },
        { provide: AiServiceClient, useValue: aiServiceClient },
      ],
    }).compile();

    service = module.get<UserSummaryAnalysisService>(UserSummaryAnalysisService);
  });

  it('enqueues a new job', async () => {
    jobModel.findOne.mockReturnValue({ sort: jest.fn().mockReturnValue(execResult(null)) });
    jobModel.create.mockResolvedValue({
      _id: { toString: () => 'job-1' },
      userId: 'user-1',
      status: 'queued',
      progress: {},
      createdAt: new Date('2026-01-01'),
      updatedAt: new Date('2026-01-01'),
    });
    jobModel.findOneAndUpdate.mockReturnValue(execResult(null));

    await expect(service.enqueue('user-1')).resolves.toMatchObject({
      id: 'job-1',
      userId: 'user-1',
      status: 'queued',
      sessionLimit: 10,
    });
    expect(jobModel.create).toHaveBeenCalledWith(expect.objectContaining({
      sessionLimit: 10,
    }));
  });

  it('denies duplicate active jobs for the same user', async () => {
    jobModel.findOne.mockReturnValue({ sort: jest.fn().mockReturnValue(execResult({ status: 'running' })) });

    await expect(service.enqueue('user-1')).rejects.toBeInstanceOf(ConflictException);
  });

  it('returns active status before latest completed status', async () => {
    const active = {
      _id: { toString: () => 'job-active' },
      userId: 'user-1',
      status: 'running',
      progress: { message: 'Analyzing sessions' },
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    jobModel.findOne.mockReturnValueOnce({ sort: jest.fn().mockReturnValue(execResult(active)) });

    await expect(service.getStatus('user-1')).resolves.toMatchObject({
      id: 'job-active',
      status: 'running',
    });
  });

  it('merges completed analysis into user summary', async () => {
    const job = { _id: 'job-1', userId: 'user-1' };
    jobModel.findOneAndUpdate.mockReturnValue(execResult(job));
    aiServiceClient.analyzeUserSessions.mockResolvedValue({
      sessionAnalysis: { sessionsAnalyzed: 2, sessionsSkipped: 1, totalTelemetryRows: 50 },
    });
    userInfoService.getUserSummary.mockResolvedValue({ manual: true });
    userInfoService.updateUserSummary.mockResolvedValue({});
    jobModel.findByIdAndUpdate.mockReturnValue(execResult({}));

    await service.processNextJob();

    expect(aiServiceClient.analyzeUserSessions).toHaveBeenCalledWith({
      user_id: 'user-1',
      session_limit: 10,
    });
    expect(userInfoService.updateUserSummary).toHaveBeenCalledWith('user-1', {
      manual: true,
      sessionAnalysis: { sessionsAnalyzed: 2, sessionsSkipped: 1, totalTelemetryRows: 50 },
    });
    expect(jobModel.findByIdAndUpdate).toHaveBeenCalledWith('job-1', expect.objectContaining({
      status: 'completed',
    }));
  });

  it('marks a job failed when AI analysis fails', async () => {
    const job = { _id: 'job-1', userId: 'user-1' };
    jobModel.findOneAndUpdate.mockReturnValue(execResult(job));
    aiServiceClient.analyzeUserSessions.mockRejectedValue(new Error('classifier unavailable'));
    jobModel.findByIdAndUpdate.mockReturnValue(execResult({}));

    await service.processNextJob();

    expect(jobModel.findByIdAndUpdate).toHaveBeenCalledWith('job-1', expect.objectContaining({
      status: 'failed',
      error: 'classifier unavailable',
    }));
  });
});
