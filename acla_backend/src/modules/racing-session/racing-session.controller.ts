import { Controller, Get, UseGuards, Request, Post, Body, Query, BadRequestException, ForbiddenException, HttpException, Inject, forwardRef, Logger, Res } from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { Response } from 'express';
import { RacingSessionDetailedInfoDto, SessionBasicInfoListDto, UploadReacingSessionInitDto, AllSessionsInitResponseDto, SessionChunkDto, AllSessionsChunkRequestDto, ImitationLearningGuidanceRequestDto, ImitationLearningGuidanceResponseDto, OpportunityForecastRequestDto, OpportunityForecastResponseDto, TrackCornerKnowledgeRequestDto, TrackCornerKnowledgeResponseDto, MapBasicInfoListDto, SegmentClassificationRequestDto, SegmentClassificationResponseDto } from 'src/dto/racing-session.dto';
import { AiModelResponseDto } from 'src/dto/ai-model.dto';
import { RacingSessionService } from './racing-session.service';
import { UserSessionAiModelService } from '../user-session-ai-model/user-session-ai-model.service';
import { UserInfoService } from '../user-info/user-info.service';
import { UserACCTrackAIModel } from 'src/schemas/session-ai-model.schema';
import { AiServiceClient, ModelsConfig, TrainModelsResponse, ImitationLearningGuidanceRequest, OpportunityForecastRequest, TrackCornerKnowledgeRequest, AiLabelsResponse } from '../../shared/ai/ai-service.client';
import { model, Types } from 'mongoose';
import * as path from 'path';
import * as fs from 'fs/promises';
import * as crypto from 'crypto';

@Controller('racing-session')
export class RacingSessionController {
    private readonly logger = new Logger(RacingSessionController.name);

    private uploadStates = new Map<string, {
        metadata: UploadReacingSessionInitDto;
        fileIds: Types.ObjectId[];
        totalDataPoints: number;
        buffer: any[];
        nextChunkIndex: number;
        createdAt: Date;
    }>();

    private downloadStates = new Map<string, {
        initData: AllSessionsInitResponseDto;
        downloadedChunks: Set<string>; // Track downloaded chunks by "sessionId:chunkIndex"
        createdAt: Date;
    }>();

    constructor(
        private racingSessionService: RacingSessionService,
        @Inject(forwardRef(() => UserSessionAiModelService))
        private aiModelService: UserSessionAiModelService,
        private aiServiceClient: AiServiceClient,
        private userInfoService: UserInfoService
    ) {
        // Clean up old assembled files every hour
        setInterval(() => {
            this.cleanupOldAssembledFiles();
        }, 60 * 60 * 1000); // 1 hour

        // Clean up old streaming files every hour
        setInterval(() => {
            this.racingSessionService.cleanupOldStreamingFiles();
        }, 60 * 60 * 1000); // 1 hour

        // Clean up old download states every hour (less aggressive)
        setInterval(() => {
            this.cleanupOldDownloadStates().catch(error => {
                this.logger.error(`Error during download state cleanup: ${error.message}`);
            });
        }, 60 * 60 * 1000); // 1 hour
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('sessionbasiclist')
    retrieveAllRacingBasicSessionsInfo(@Request() req, @Body() body): Promise<SessionBasicInfoListDto | null> {
        return this.racingSessionService.retrieveAllRacingSessionsBasicInfo(body.map_name, body.user_id);
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('mapbasiclist')
    retrieveAllSessionMapBasicInfo(@Request() req, @Body() body): Promise<MapBasicInfoListDto | null> {
        return this.racingSessionService.retrieveAllSessionMapBasicInfo(body.user_id);
    }


    @UseGuards(AuthGuard('jwt'))
    @Post('detailedSessionInfo')
    retrieveSessionDetailedInfo(@Request() req, @Body() body): Promise<RacingSessionDetailedInfoDto | null> {

        return this.racingSessionService.retrieveSessionDetailedInfo(body.id);
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('download/init')
    async initializeSessionsDownload(
        @Request() req,
        @Body() body: { trackName?: string, carName?: string, sessionId?: string, chunkSize?: number }
    ): Promise<AllSessionsInitResponseDto> {
        try {
            const chunkSize = body.chunkSize || 1000; // Legacy parameter, ignored in streaming mode
            const initDataWithContext = await this.racingSessionService.initializeSessionsDownload(body.trackName, body.carName, chunkSize, body.sessionId);

            this.downloadStates.set(initDataWithContext.downloadId, {
                initData: initDataWithContext,
                downloadedChunks: new Set<string>(),
                createdAt: new Date()
            });

            this.logger.log(`Initialized download session ${initDataWithContext.downloadId} with ${initDataWithContext.totalSessions} sessions`);

            // Clean up old download states (older than 2 hours)
            this.cleanupOldDownloadStates().catch(error => {
                this.logger.error(`Error during download state cleanup: ${error.message}`);
            });

            return initDataWithContext;
        } catch (error) {
            throw new BadRequestException(`Failed to initialize download: ${error.message}`);
        }
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('download/chunk')
    async downloadSessionChunk(
        @Request() req,
        @Body() body: AllSessionsChunkRequestDto,
        @Res() res: Response
    ): Promise<void> {
        try {
            // Validate download state exists
            const downloadState = this.downloadStates.get(body.downloadId);
            if (!downloadState) {
                this.logger.warn(`Download session not found: ${body.downloadId}, available sessions: ${Array.from(this.downloadStates.keys()).join(', ')}`);
                throw new BadRequestException('Download session not found or expired');
            }

            // Update last access time to prevent premature cleanup
            downloadState.createdAt = new Date();

            // Validate session exists in the download
            const sessionExists = downloadState.initData.sessionMetadata.some(
                session => session.sessionId === body.sessionId
            );
            if (!sessionExists) {
                this.logger.error(`Session ${body.sessionId} not found in download ${body.downloadId}`);
                throw new BadRequestException('Session not found in download');
            }

            const chunkIndex = Number(body.chunkIndex || 0);
            const chunk = await this.racingSessionService.getSessionDownloadChunk(body.sessionId, chunkIndex);

            // Set response headers for streaming
            res.setHeader('Content-Type', 'application/json');
            res.setHeader('Content-Length', chunk.fileSize.toString());
            res.setHeader('Content-Disposition', `attachment; filename="session_${body.sessionId}_chunk_${chunkIndex}.json"`);
            res.setHeader('X-Download-Id', body.downloadId);
            res.setHeader('X-Session-Id', body.sessionId);
            res.setHeader('X-Chunk-Index', chunkIndex.toString());
            res.setHeader('X-Total-Chunks', chunk.totalChunks.toString());
            res.setHeader('X-Data-Points', chunk.dataPoints.toString());

            // Track downloaded session
            const chunkKey = `${body.sessionId}:${chunkIndex}`;
            downloadState.downloadedChunks.add(chunkKey);

            // Handle stream errors
            chunk.stream.on('error', (error) => {
                this.logger.error(`Error streaming session chunk ${body.sessionId}:${chunkIndex}: ${error.message}`);
                if (!res.headersSent) {
                    res.status(500).json({ error: 'Failed to stream session data' });
                }
            });

            chunk.stream.on('end', () => {
                this.logger.log(`Successfully streamed session chunk ${body.sessionId}:${chunkIndex} (${chunk.fileSize} bytes)`);
            });

            chunk.stream.pipe(res);

        } catch (error) {
            this.logger.error(`Failed to stream session chunk: ${error.message}`);
            if (!res.headersSent) {
                throw new BadRequestException(`Failed to download chunk: ${error.message}`);
            }
        }
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('analysis/user-sessions/init')
    async initializeUserSessionsAnalysis(
        @Request() req,
        @Body() body: { userId?: string }
    ) {
        const targetUserId = body.userId;
        if (!targetUserId) {
            throw new BadRequestException('userId is required');
        }
        this.assertCanAccessAnalysisTarget(req, targetUserId);

        const sessions = await this.racingSessionService.listUserSessionsForAnalysis(targetUserId);
        return {
            userId: targetUserId,
            totalSessions: sessions.length,
            sessions,
        };
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('analysis/user-sessions/chunk')
    async downloadUserSessionAnalysisChunk(
        @Request() req,
        @Body() body: { userId?: string; sessionId?: string; chunkIndex?: number },
        @Res() res: Response
    ): Promise<void> {
        try {
            const targetUserId = body.userId;
            if (!targetUserId || !body.sessionId || body.chunkIndex === undefined) {
                throw new BadRequestException('userId, sessionId, and chunkIndex are required');
            }
            this.assertCanAccessAnalysisTarget(req, targetUserId);

            const chunk = await this.racingSessionService.getUserSessionAnalysisChunk(
                targetUserId,
                body.sessionId,
                Number(body.chunkIndex),
            );

            res.setHeader('Content-Type', 'application/json');
            res.setHeader('Content-Length', chunk.fileSize.toString());
            res.setHeader('X-Session-Id', body.sessionId);
            res.setHeader('X-Chunk-Index', String(body.chunkIndex));
            res.setHeader('X-Total-Chunks', String(chunk.totalChunks));

            chunk.stream.on('error', (error) => {
                this.logger.error(`Error streaming analysis chunk ${body.sessionId}:${body.chunkIndex}: ${error.message}`);
                if (!res.headersSent) {
                    res.status(500).json({ error: 'Failed to stream analysis chunk' });
                }
            });

            chunk.stream.pipe(res);
        } catch (error) {
            this.logger.error(`Failed to stream analysis chunk: ${error.message}`);
            if (!res.headersSent) {
                if (error instanceof BadRequestException || error instanceof ForbiddenException) {
                    throw error;
                }
                throw new BadRequestException(`Failed to stream analysis chunk: ${error.message}`);
            }
        }
    }

    private assertCanAccessAnalysisTarget(req: any, targetUserId: string): void {
        const authenticatedUserId = req.user?.userId;
        const authenticatedUsername = req.user?.username;
        const isAiService = authenticatedUsername && authenticatedUsername === process.env.AI_SERVICE_USERNAME;

        if (authenticatedUserId !== targetUserId && !isAiService) {
            throw new ForbiddenException('Cannot access analysis data for another user');
        }
    }

    /**
     * Clean up download states older than 2 hours and associated streaming files
     * Increased timeout to prevent premature cleanup during active downloads
     */
    private async cleanupOldDownloadStates(): Promise<void> {
        const twoHoursAgo = new Date(Date.now() - 2 * 60 * 60 * 1000); // Increased to 2 hours
        const statestoCleanup: string[] = [];

        // First, identify states that need cleanup
        for (const [downloadId, state] of this.downloadStates.entries()) {
            if (state.createdAt < twoHoursAgo) {
                statestoCleanup.push(downloadId);
            }
        }

        // Then clean them up to avoid concurrent modification
        for (const downloadId of statestoCleanup) {
            try {
                await this.cleanupDownloadSession(downloadId);
            } catch (error) {
                this.logger.warn(`Failed to cleanup download session ${downloadId}: ${error.message}`);
            }
        }

        if (statestoCleanup.length > 0) {
            this.logger.log(`Cleaned up ${statestoCleanup.length} expired download sessions`);
        }
    }

    /**
     * Clean up a specific download session and its associated streaming files
     */
    private async cleanupDownloadSession(downloadId: string): Promise<void> {
        try {
            const state = this.downloadStates.get(downloadId);
            if (state) {
                // Check if the session was recently accessed (within last 30 minutes)
                const thirtyMinutesAgo = new Date(Date.now() - 30 * 60 * 1000);
                if (state.createdAt > thirtyMinutesAgo) {
                    this.logger.debug(`Skipping cleanup for recently accessed download session: ${downloadId}`);
                    return;
                }

                // Clean up associated streaming files
                await this.racingSessionService.cleanupStreamingFiles(downloadId);
                // Remove from memory
                this.downloadStates.delete(downloadId);
                this.logger.log(`Cleaned up download session: ${downloadId}`);
            }
        } catch (error) {
            this.logger.warn(`Failed to cleanup download session ${downloadId}: ${error.message}`);
        }
    }

    /**
     * Clean up assembled file after successful upload completion
     */
    private async cleanupAssembledFile(uploadId: string): Promise<void> {
        try {
            const assembledFilePath = path.resolve(process.cwd(), 'session_recording', 'temp', 'assembled', `${uploadId}.bin`);
            await fs.unlink(assembledFilePath);
            this.logger.log(`Cleaned up assembled file for upload ${uploadId}`);
        } catch (error) {
            // File might not exist or already deleted, which is fine
            this.logger.debug(`Could not clean up assembled file for upload ${uploadId}: ${error.message}`);
        }
    }

    /**
     * Clean up old assembled files (older than 2 hours)
     */
    private async cleanupOldAssembledFiles(): Promise<void> {
        try {
            const assembledDir = path.resolve(process.cwd(), 'session_recording', 'temp', 'assembled');
            const files = await fs.readdir(assembledDir);
            const twoHoursAgo = Date.now() - (2 * 60 * 60 * 1000); // 2 hours
            let cleanedCount = 0;

            for (const file of files) {
                try {
                    const filePath = path.join(assembledDir, file);
                    const stats = await fs.stat(filePath);

                    if (stats.mtime.getTime() < twoHoursAgo) {
                        await fs.unlink(filePath);
                        cleanedCount++;
                    }
                } catch (error) {
                    this.logger.debug(`Could not process file ${file}: ${error.message}`);
                }
            }

            if (cleanedCount > 0) {
                this.logger.log(`Cleaned up ${cleanedCount} old assembled files`);
            }
        } catch (error) {
            this.logger.debug(`Could not clean up old assembled files: ${error.message}`);
        }
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('download/status')
    async getDownloadStatus(
        @Request() req,
        @Body() body: { downloadId: string }
    ) {
        const downloadState = this.downloadStates.get(body.downloadId);
        if (!downloadState) {
            this.logger.warn(`Download status requested for non-existent session: ${body.downloadId}`);
            throw new BadRequestException('Download session not found or expired');
        }

        // Update last access time to prevent premature cleanup
        downloadState.createdAt = new Date();

        const totalPossibleChunks = downloadState.initData.totalChunks || 0;
        const downloadedChunks = downloadState.downloadedChunks.size;
        const progress = totalPossibleChunks > 0 ? (downloadedChunks / totalPossibleChunks) * 100 : 0;

        const isComplete = downloadedChunks >= totalPossibleChunks;

        // If download is complete, schedule cleanup with longer delay
        if (isComplete) {
            setTimeout(async () => {
                try {
                    await this.cleanupDownloadSession(body.downloadId);
                } catch (error) {
                    this.logger.warn(`Failed to cleanup completed download session ${body.downloadId}: ${error.message}`);
                }
            }, 10 * 60 * 1000); // Increased to 10 minutes to allow for slower connections
        }

        return {
            downloadId: body.downloadId,
            totalSessions: downloadState.initData.totalSessions,
            totalChunks: totalPossibleChunks,
            downloadedChunks,
            progress: Math.round(progress * 100) / 100, // Round to 2 decimal places
            isComplete,
            createdAt: downloadState.createdAt
        };
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('upload/init')
    async initUpload(@Body() metadata: UploadReacingSessionInitDto) {
        const uploadId = crypto.randomUUID();
        console.log('Initialized upload with ID:', uploadId, 'for user:', metadata.userId);

        this.uploadStates.set(uploadId, {
            metadata,
            fileIds: [],
            totalDataPoints: 0,
            buffer: [],
            nextChunkIndex: 0,
            createdAt: new Date()
        });
        return { uploadId: uploadId };
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('upload/chunk')
    async receiveChunk(
        @Body() body: { chunk: any[]; chunkIndex: number },
        @Query('uploadId') uploadId: string
    ) {
        const upload = this.uploadStates.get(uploadId);
        if (!upload) {
            throw new BadRequestException('Upload doesnt exist or expired');
        }

        if (body.chunkIndex !== upload.nextChunkIndex) {
            // If we receive a previous chunk, it might be a retry that succeeded but we didn't get the response?
            // But if we are strictly sequential, we expect nextChunkIndex.
            // If chunkIndex < nextChunkIndex, we can ignore it (idempotent).
            if (body.chunkIndex < upload.nextChunkIndex) {
                return { receivedChunks: upload.nextChunkIndex };
            }
            throw new BadRequestException(`Invalid chunk index. Expected ${upload.nextChunkIndex}, got ${body.chunkIndex}`);
        }

        if (body.chunk && body.chunk.length > 0) {
            upload.buffer = upload.buffer.concat(body.chunk);
            upload.totalDataPoints += body.chunk.length;
        }

        // If buffer is large enough, upload to GridFS
        const CHUNK_SIZE = 1000;
        while (upload.buffer.length >= CHUNK_SIZE) {
            const chunkToUpload = upload.buffer.splice(0, CHUNK_SIZE);
            const fileId = await this.racingSessionService.uploadSessionChunk(
                chunkToUpload,
                {
                    session_name: upload.metadata.sessionName,
                    map: upload.metadata.mapName,
                    car_name: upload.metadata.carName,
                    userId: upload.metadata.userId,
                    chunkIndex: upload.fileIds.length,
                    chunkSize: CHUNK_SIZE
                }
            );
            upload.fileIds.push(fileId as unknown as Types.ObjectId);
        }

        upload.nextChunkIndex++;
        // Update last access
        upload.createdAt = new Date();

        return { receivedChunks: upload.nextChunkIndex };
    }

    @Post('upload/complete')
    async completeUpload(
        @Body() completionData: any,
        @Query('uploadId') uploadId: string
    ) {
        const upload = this.uploadStates.get(uploadId);
        if (!upload) {
            throw new BadRequestException('Upload doesnt exist or expired');
        }

        try {
            // Upload remaining buffer
            if (upload.buffer.length > 0) {
                const fileId = await this.racingSessionService.uploadSessionChunk(
                    upload.buffer,
                    {
                        session_name: upload.metadata.sessionName,
                        map: upload.metadata.mapName,
                        car_name: upload.metadata.carName,
                        userId: upload.metadata.userId,
                        chunkIndex: upload.fileIds.length,
                        chunkSize: 1000
                    }
                );
                upload.fileIds.push(fileId as unknown as Types.ObjectId);
            }

            // Create racing session in database
            const createdSession = await this.racingSessionService.createRacingSessionFromChunks(
                upload.metadata.sessionName,
                upload.metadata.mapName,
                upload.metadata.carName,
                upload.metadata.userId,
                upload.fileIds as unknown as any[],
                upload.totalDataPoints,
                1000
            );

            return {
                message: 'Upload completed successfully',
                sessionId: createdSession._id, // Assuming createRacingSession returns the document
                aiAnalysisAvailable: true
            };

        } catch (error) {
            this.logger.error(`Error creating Racing Session: ${error.message}`);
            throw new BadRequestException(`Upload failed: ${error.message}`);
        } finally {
            // Clean up
            this.uploadStates.delete(uploadId);
        }
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('imitation-learning-guidance')
    async getImitationLearningGuidance(
        @Request() req,
        @Body() body: ImitationLearningGuidanceRequestDto
    ): Promise<ImitationLearningGuidanceResponseDto> {
        try {

            // Prepare request for AI service
            const guidanceRequest: ImitationLearningGuidanceRequest = {
                current_telemetry: body.current_telemetry,
                track_name: body.track_name,
                car_name: body.car_name,
                user_id: body.user_id || req.user?.email // Use authenticated user's email if not provided
            };

            // Call AI service for imitation learning guidance
            const response = await this.aiServiceClient.getImitationLearningGuidance(guidanceRequest);

            return {
                message: response.message,
                guidance_result: response.guidance_result,
                timestamp: response.timestamp,
                recommendations: response.recommendations,
                confidence_score: response.confidence_score,
                success: true
            };

        } catch (error) {
            console.error('Imitation learning guidance failed:', error);
            throw new BadRequestException(`Failed to get imitation learning guidance: ${error.message}`);
        }
    }

    @UseGuards(AuthGuard('jwt'))
    @Get('labels')
    async getLabels(): Promise<AiLabelsResponse> {
        try {
            return await this.aiServiceClient.getLabels();
        } catch (error) {
            console.error('AI labels retrieval failed:', error);
            throw new BadRequestException(`Failed to get AI labels: ${error.message}`);
        }
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('opportunity-forecast')
    async getOpportunityForecast(
        @Request() req,
        @Body() body: OpportunityForecastRequestDto
    ): Promise<OpportunityForecastResponseDto> {
        try {
            if (!Array.isArray(body.telemetry_data) || body.telemetry_data.length === 0) {
                throw new BadRequestException('telemetry_data is required');
            }

            const forecastRequest: OpportunityForecastRequest = {
                telemetry_data: body.telemetry_data,
                horizon_seconds: body.horizon_seconds ?? 10,
                top_k: body.top_k ?? 3
            };

            return await this.aiServiceClient.getOpportunityForecast(forecastRequest);
        } catch (error) {
            if (error instanceof BadRequestException) {
                throw error;
            }
            console.error('Opportunity forecast failed:', error);
            throw new BadRequestException(`Failed to get opportunity forecast: ${error.message}`);
        }
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('track-corner-knowledge')
    async getTrackCornerKnowledge(
        @Request() req,
        @Body() body: TrackCornerKnowledgeRequestDto
    ): Promise<TrackCornerKnowledgeResponseDto> {
        try {
            if (!body.track_name) {
                throw new BadRequestException('track_name is required');
            }
            if (!body.corner_name) {
                throw new BadRequestException('corner_name is required');
            }

            const knowledgeRequest: TrackCornerKnowledgeRequest = {
                track_name: body.track_name,
                corner_name: body.corner_name,
                normalized_position: body.normalized_position,
                trigger_position: body.trigger_position,
                current_telemetry: body.current_telemetry
            };

            return await this.aiServiceClient.getTrackCornerKnowledge(knowledgeRequest);
        } catch (error) {
            if (error instanceof BadRequestException || error instanceof HttpException) {
                throw error;
            }
            console.error('Track corner knowledge failed:', error);
            throw new BadRequestException(`Failed to get track corner knowledge: ${error.message}`);
        }
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('segment-classification')
    async classifySessionSegments(
        @Request() req,
        @Body() body: SegmentClassificationRequestDto
    ): Promise<SegmentClassificationResponseDto> {
        try {
            const sessionId = body.session_id || (body as any).sessionId;
            if (!sessionId) {
                throw new BadRequestException('session_id is required');
            }

            const userId = req.user?.userId;
            if (!userId) {
                throw new BadRequestException('Authenticated user id is required');
            }

            const sessionPayload = await this.racingSessionService.getSessionTelemetryForClassification(
                userId,
                sessionId,
            );

            return await this.aiServiceClient.classifySegments({
                session_id: sessionPayload.sessionId,
                telemetry_data: sessionPayload.telemetryData,
                track_name: sessionPayload.trackName,
                car_name: sessionPayload.carName,
            });
        } catch (error) {
            if (error instanceof BadRequestException || error instanceof ForbiddenException || error instanceof HttpException) {
                throw error;
            }
            console.error('Segment classification failed:', error);
            throw new BadRequestException(`Failed to classify session segments: ${error.message}`);
        }
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('download/cleanup')
    async cleanupDownload(
        @Request() req,
        @Body() body: { downloadId: string }
    ) {
        try {
            await this.cleanupDownloadSession(body.downloadId);
            return { message: 'Download session cleaned up successfully' };
        } catch (error) {
            throw new BadRequestException(`Failed to cleanup download: ${error.message}`);
        }
    }

    @UseGuards(AuthGuard('jwt'))
    @Get('download/debug')
    async getDownloadDebugInfo(@Request() req) {
        const activeDownloads = Array.from(this.downloadStates.entries()).map(([id, state]) => ({
            downloadId: id,
            createdAt: state.createdAt,
            sessionCount: state.initData.totalSessions,
            downloadedChunks: state.downloadedChunks.size,
            totalChunks: state.initData.totalChunks || 0
        }));

        return {
            activeDownloadSessions: activeDownloads.length,
            downloads: activeDownloads,
            currentTime: new Date()
        };
    }


}
