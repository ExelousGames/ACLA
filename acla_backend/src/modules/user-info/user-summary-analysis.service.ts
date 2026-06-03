import { ConflictException, Injectable, Logger, OnModuleInit } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { AiServiceClient } from 'src/shared/ai/ai-service.client';
import {
    UserSummaryAnalysisJob,
    UserSummaryAnalysisJobDocument,
} from 'src/schemas/user-summary-analysis-job.schema';
import { UserInfoService } from './user-info.service';

@Injectable()
export class UserSummaryAnalysisService implements OnModuleInit {
    private readonly logger = new Logger(UserSummaryAnalysisService.name);
    private isProcessing = false;

    constructor(
        @InjectModel(UserSummaryAnalysisJob.name)
        private readonly jobModel: Model<UserSummaryAnalysisJobDocument>,
        private readonly userInfoService: UserInfoService,
        private readonly aiServiceClient: AiServiceClient,
    ) { }

    onModuleInit() {
        const interval = setInterval(() => {
            void this.processNextJob();
        }, 5000);

        if (typeof (interval as any).unref === 'function') {
            (interval as any).unref();
        }

        void this.processNextJob();
    }

    async enqueue(userId: string): Promise<Record<string, any>> {
        const existing = await this.jobModel
            .findOne({ userId, status: { $in: ['queued', 'running'] } })
            .sort({ createdAt: -1 })
            .exec();

        if (existing) {
            throw new ConflictException('User summary analysis is already queued or running for this user');
        }

        try {
            const job = await this.jobModel.create({
                userId,
                status: 'queued',
                progress: { message: 'Queued' },
                result: null,
                error: null,
                createdAt: new Date(),
                updatedAt: new Date(),
                startedAt: null,
                completedAt: null,
            });

            void this.processNextJob();
            return this.toDto(job);
        } catch (error) {
            if (error?.code === 11000) {
                throw new ConflictException('User summary analysis is already queued or running for this user');
            }
            throw error;
        }
    }

    async getStatus(userId: string): Promise<Record<string, any> | null> {
        const activeJob = await this.jobModel
            .findOne({ userId, status: { $in: ['queued', 'running'] } })
            .sort({ createdAt: -1 })
            .exec();

        if (activeJob) {
            return this.toDto(activeJob);
        }

        const job = await this.jobModel
            .findOne({ userId })
            .sort({ createdAt: -1 })
            .exec();

        return job ? this.toDto(job) : null;
    }

    async processNextJob(): Promise<void> {
        if (this.isProcessing) {
            return;
        }

        this.isProcessing = true;
        try {
            const job = await this.jobModel.findOneAndUpdate(
                { status: 'queued' },
                {
                    status: 'running',
                    startedAt: new Date(),
                    updatedAt: new Date(),
                    progress: { message: 'Analyzing sessions' },
                },
                { new: true, sort: { createdAt: 1 } },
            ).exec();

            if (!job) {
                return;
            }

            await this.runJob(job);
        } finally {
            this.isProcessing = false;
        }
    }

    private async runJob(job: UserSummaryAnalysisJobDocument): Promise<void> {
        try {
            const response = await this.aiServiceClient.analyzeUserSessions({ user_id: job.userId });
            const sessionAnalysis = response.sessionAnalysis || {};
            const existingSummary = await this.userInfoService.getUserSummary(job.userId);
            const mergedSummary = {
                ...existingSummary,
                sessionAnalysis,
            };

            await this.userInfoService.updateUserSummary(job.userId, mergedSummary);

            await this.jobModel.findByIdAndUpdate(job._id, {
                status: 'completed',
                progress: {
                    message: 'Completed',
                    sessionsAnalyzed: sessionAnalysis.sessionsAnalyzed ?? 0,
                    sessionsSkipped: sessionAnalysis.sessionsSkipped ?? 0,
                    totalTelemetryRows: sessionAnalysis.totalTelemetryRows ?? 0,
                },
                result: {
                    sessionsAnalyzed: sessionAnalysis.sessionsAnalyzed ?? 0,
                    sessionsSkipped: sessionAnalysis.sessionsSkipped ?? 0,
                    totalTelemetryRows: sessionAnalysis.totalTelemetryRows ?? 0,
                },
                error: null,
                completedAt: new Date(),
                updatedAt: new Date(),
            }).exec();
        } catch (error) {
            this.logger.error(`User summary analysis failed for ${job.userId}: ${error?.message || error}`);
            await this.jobModel.findByIdAndUpdate(job._id, {
                status: 'failed',
                progress: { message: 'Failed' },
                error: error?.message || String(error),
                completedAt: new Date(),
                updatedAt: new Date(),
            }).exec();
        }
    }

    private toDto(job: UserSummaryAnalysisJobDocument): Record<string, any> {
        return {
            id: job._id.toString(),
            userId: job.userId,
            status: job.status,
            progress: job.progress || {},
            result: job.result || null,
            error: job.error || null,
            createdAt: job.createdAt,
            updatedAt: job.updatedAt,
            startedAt: job.startedAt || null,
            completedAt: job.completedAt || null,
        };
    }
}
