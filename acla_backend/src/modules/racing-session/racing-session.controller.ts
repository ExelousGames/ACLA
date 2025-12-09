import { Controller, Get, UseGuards, Request, Post, Body, Query, BadRequestException, Inject, forwardRef, Logger, Res } from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { Response } from 'express';
import { RacingSessionDetailedInfoDto, SessionBasicInfoListDto, UploadReacingSessionInitDto, AllSessionsInitResponseDto, SessionChunkDto, AllSessionsChunkRequestDto, ImitationLearningGuidanceRequestDto, ImitationLearningGuidanceResponseDto } from 'src/dto/racing-session.dto';
import { AiModelResponseDto } from 'src/dto/ai-model.dto';
import { RacingSessionService } from './racing-session.service';
import { UserSessionAiModelService } from '../user-session-ai-model/user-session-ai-model.service';
import { UserInfoService } from '../user-info/user-info.service';
import { UserACCTrackAIModel } from 'src/schemas/session-ai-model.schema';
import { AiServiceClient, ModelsConfig, TrainModelsResponse, ImitationLearningGuidanceRequest } from '../../shared/ai/ai-service.client';
import { model, Types } from 'mongoose';
import * as path from 'path';
import * as fs from 'fs/promises';
import { createReadStream } from 'fs';
import * as crypto from 'crypto';

@Controller('racing-session')
export class RacingSessionController {
    private readonly logger = new Logger(RacingSessionController.name);

    private uploadStates = new Map<string, {
        metadata: UploadReacingSessionInitDto;
        filePath: string;
        nextChunkIndex: number;
        createdAt: Date;
    }>();

    private downloadStates = new Map<string, {
        initData: AllSessionsInitResponseDto;
        downloadedChunks: Set<string>; // Track downloaded chunks by "sessionId:chunkIndex"
        streamingFiles?: Map<string, { filePath: string; fileSize: number; }>; // Track streaming file paths by sessionId
        tempDir?: string; // Directory containing temporary streaming files
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
    @Post('detailedSessionInfo')
    retrieveSessionDetailedInfo(@Request() req, @Body() body): Promise<RacingSessionDetailedInfoDto | null> {

        return this.racingSessionService.retrieveSessionDetailedInfo(body.id);
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('download/init')
    async initializeSessionsDownload(
        @Request() req,
        @Body() body: { trackName?: string, carName?: string, chunkSize?: number }
    ): Promise<AllSessionsInitResponseDto> {
        try {
            const chunkSize = body.chunkSize || 1000; // Legacy parameter, ignored in streaming mode
            const initDataWithContext = await this.racingSessionService.initializeSessionsDownload(body.trackName, body.carName, chunkSize);

            // Store download state for tracking with streaming file information
            const streamingFiles = new Map<string, { filePath: string; fileSize: number; }>();

            // Populate streaming files map from the context
            if (initDataWithContext.streamingContext?.sessionFiles) {
                for (const sessionFile of initDataWithContext.streamingContext.sessionFiles) {
                    streamingFiles.set(sessionFile.sessionId, {
                        filePath: sessionFile.filePath,
                        fileSize: sessionFile.fileSize
                    });
                }
            }

            // Remove streaming context from response (internal use only)
            const initData: AllSessionsInitResponseDto = {
                downloadId: initDataWithContext.downloadId,
                totalSessions: initDataWithContext.totalSessions,
                totalChunks: initDataWithContext.totalChunks,
                sessionMetadata: initDataWithContext.sessionMetadata
            };

            this.downloadStates.set(initData.downloadId, {
                initData,
                downloadedChunks: new Set<string>(),
                streamingFiles,
                tempDir: initDataWithContext.streamingContext?.tempDir,
                createdAt: new Date()
            });

            this.logger.log(`Initialized download session ${initData.downloadId} with ${initData.totalSessions} sessions`);

            // Clean up old download states (older than 2 hours)
            this.cleanupOldDownloadStates().catch(error => {
                this.logger.error(`Error during download state cleanup: ${error.message}`);
            });

            return initData;
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

            // Get streaming file information
            const streamingFile = downloadState.streamingFiles?.get(body.sessionId);
            if (!streamingFile) {
                this.logger.error(`Session streaming file not found for session ${body.sessionId}`);
                throw new BadRequestException('Session streaming file not found');
            }

            const { filePath, fileSize } = streamingFile;

            // Verify file exists
            try {
                await fs.access(filePath);
            } catch (error) {
                this.logger.error(`Session file not accessible: ${filePath}, error: ${error.message}`);
                throw new BadRequestException('Session file not accessible');
            }

            // Set response headers for streaming
            res.setHeader('Content-Type', 'application/json');
            res.setHeader('Content-Length', fileSize.toString());
            res.setHeader('Content-Disposition', `attachment; filename="session_${body.sessionId}.json"`);
            res.setHeader('X-Download-Id', body.downloadId);
            res.setHeader('X-Session-Id', body.sessionId);

            // Track downloaded session
            const chunkKey = `${body.sessionId}:0`; // Always index 0 for streaming mode
            downloadState.downloadedChunks.add(chunkKey);

            // Create read stream and pipe to response
            const readStream = createReadStream(filePath);

            // Handle stream errors
            readStream.on('error', (error) => {
                this.logger.error(`Error streaming file ${filePath}: ${error.message}`);
                if (!res.headersSent) {
                    res.status(500).json({ error: 'Failed to stream session data' });
                }
            });

            readStream.on('end', () => {
                this.logger.log(`Successfully streamed session ${body.sessionId} (${fileSize} bytes)`);
            });

            // Pipe the file stream to response
            readStream.pipe(res);

        } catch (error) {
            this.logger.error(`Failed to stream session chunk: ${error.message}`);
            if (!res.headersSent) {
                throw new BadRequestException(`Failed to download chunk: ${error.message}`);
            }
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

        const uploadDir = path.resolve(process.cwd(), 'session_recording', 'temp', 'uploads');
        await fs.mkdir(uploadDir, { recursive: true });
        const filePath = path.join(uploadDir, `${uploadId}.json`);

        // Initialize empty file
        await fs.writeFile(filePath, '');

        this.uploadStates.set(uploadId, {
            metadata,
            filePath,
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

        // Prepare data to write
        let dataToWrite = '';
        if (body.chunk && body.chunk.length > 0) {
            const jsonStr = JSON.stringify(body.chunk);
            // Remove [ and ]
            const content = jsonStr.substring(1, jsonStr.length - 1);

            if (upload.nextChunkIndex === 0) {
                dataToWrite = '[' + content;
            } else {
                dataToWrite = ',' + content;
            }
        } else if (upload.nextChunkIndex === 0) {
            // Handle empty first chunk (empty file)
            dataToWrite = '[';
        }

        await fs.appendFile(upload.filePath, dataToWrite);

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
            // Finalize file format
            await fs.appendFile(upload.filePath, ']');

            // Read and parse the full dataset
            // Note: For extremely large files, we should use a streaming parser, 
            // but for now we load into memory to pass to createRacingSession which expects an array.
            const fileContent = await fs.readFile(upload.filePath, 'utf8');
            let fullDataset: any[];
            try {
                fullDataset = JSON.parse(fileContent);
            } catch (e) {
                this.logger.error(`Failed to parse uploaded file: ${e.message}`);
                throw new BadRequestException('Invalid JSON data in upload');
            }

            // Create racing session in database
            const createdSession = await this.racingSessionService.createRacingSession(
                upload.metadata.sessionName,
                upload.metadata.mapName,
                upload.metadata.carName,
                upload.metadata.userId,
                fullDataset
            );

            // AI training logic (commented out in original, keeping it commented out or preserving if it was active)
            // ... (Preserving original commented out code structure if needed, but for brevity I'll omit the commented block unless requested)

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
            try {
                await fs.unlink(upload.filePath);
            } catch (e) {
                this.logger.warn(`Failed to delete temp file ${upload.filePath}: ${e.message}`);
            }
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
            hasStreamingFiles: !!state.streamingFiles && state.streamingFiles.size > 0,
            tempDir: state.tempDir
        }));

        return {
            activeDownloadSessions: activeDownloads.length,
            downloads: activeDownloads,
            currentTime: new Date()
        };
    }


}
