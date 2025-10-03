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

@Controller('racing-session')
export class RacingSessionController {
    private readonly logger = new Logger(RacingSessionController.name);

    private uploadStates = new Map<string, {
        metadata: UploadReacingSessionInitDto;
        session_data_chunks: string[][];
        received: number;
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
        this.uploadStates.set(uploadId, {
            metadata,
            session_data_chunks: [],
            received: 0
        });
        return { uploadId: uploadId };
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('upload/chunk')
    async receiveChunk(
        @Body() body: { chunk: string[]; chunkIndex: number },
        @Query('uploadId') uploadId: string
    ) {
        const upload = this.uploadStates.get(uploadId);
        if (!upload) {
            throw new BadRequestException('Upload doesnt exist');
        }
        upload.session_data_chunks[body.chunkIndex] = body.chunk;
        upload.received++;


        return { receivedChunks: upload.session_data_chunks.length };
    }

    @Post('upload/complete')
    async completeUpload(
        @Body() completionData: any,
        @Query('uploadId') uploadId: string //Extracts values from the URL query string (the part after ? in a URL)
    ) {

        //get the data about the session upload
        const upload = this.uploadStates.get(uploadId);
        if (!upload) {
            throw new BadRequestException('Upload doesnt exist');
        }
        const fullDataset = upload.session_data_chunks.flat();

        // Create racing session in database
        try {
            const createdSession = await this.racingSessionService.createRacingSession(
                upload.metadata.sessionName,
                upload.metadata.mapName,
                upload.metadata.carName,
                upload.metadata.userId,
                fullDataset
            );

            // // ai training
            // try {
            //     // First, find the user by user id  to get their ObjectId
            //     const userInfo = await this.userInfoService.findOneById(upload.metadata.userId);

            //     if (!userInfo) {
            //         console.log('User not found for id:', upload.metadata.userId);
            //     } else {

            //         const userId = upload.metadata.userId;
            //         // Check for active AI model first, before processing the session

            //         //list of ai models will be trained
            //         const modelsConfig: ModelsConfig[] = [
            //             { config_id: "lap_prediction", target_variable: "Graphics_current_time", model_type: "lap_time_prediction" }
            //         ];

            //         let activeModel: UserACCTrackAIModel & { _id: Types.ObjectId; } | null = null;

            //         // Check for existing active models for each config
            //         for (const modelConfig of modelsConfig) {

            //             // Check if user has an active model for this track
            //             activeModel = await this.aiModelService.findActiveUserSessionAIModel(
            //                 userId,
            //                 upload.metadata.mapName,
            //                 upload.metadata.carName,
            //                 modelConfig.model_type,
            //                 modelConfig.target_variable
            //             );

            //             // add active model if any
            //             modelConfig.existing_model_data = activeModel ? activeModel : null;
            //         }

            //         // Train the models using the AI service client
            //         const trainedModelsResponse: TrainModelsResponse = await this.aiServiceClient.trainModels({
            //             session_id: createdSession.id,
            //             telemetry_data: fullDataset,
            //             models_config: modelsConfig,
            //             user_id: userId,
            //             parallel_training: false
            //         });

            //         //save the training result to database
            //         if (trainedModelsResponse) {
            //             // Process and save the training result as needed
            //             for (const [configId, response] of Object.entries(trainedModelsResponse.training_results)) {

            //                 //find the matching config
            //                 const modelConfig = modelsConfig.find(config => config.config_id === configId);
            //                 console.log("Training result for response:", response);
            //                 if (response.success) {
            //                     //if there is an active model, update the model in database
            //                     if (modelConfig && modelConfig.existing_model_data) {

            //                         await this.aiModelService.updateModel(modelConfig.existing_model_data._id.toString(), {
            //                             modelData: response.model_data,
            //                             modelType: response.model_type,
            //                             algorithmUsed: response.algorithm_used,
            //                             algorithmType: response.algorithm_type,
            //                             targetVariable: response.target_variable,
            //                             trainingMetrics: response.training_metrics,
            //                             featureNames: response.feature_names,
            //                             featureCount: response.features_count,
            //                             samplesProcessed: response.samples_processed,
            //                             modelVersion: response.model_version, // Version number for incremental training
            //                             algorithmStrengths: response.algorithm_strengths, // Summary of telemetry data used
            //                             recommendations: response.recommendations, // Training recommendations
            //                             algorithmDescription: response.algorithm_description, // Description of the algorithm used
            //                             training_time: response.training_time, // Training time
            //                             dataQualityScore: response.data_quality_score, // Feature importance scores
            //                             timestamp: response.timestamp, // When the model was trained
            //                             isActive: true // Whether this model version is active
            //                         });
            //                     } else {
            //                         //else create a new model
            //                         await this.aiModelService.createModel({
            //                             userId: userId,
            //                             trackName: upload.metadata.mapName,
            //                             carName: upload.metadata.carName,
            //                             modelData: response.model_data,
            //                             modelType: response.model_type,
            //                             algorithmUsed: response.algorithm_used,
            //                             algorithmType: response.algorithm_type,
            //                             targetVariable: response.target_variable,
            //                             trainingMetrics: response.training_metrics,
            //                             featureNames: response.feature_names,
            //                             featureCount: response.features_count,
            //                             samplesProcessed: response.samples_processed,
            //                             modelVersion: response.model_version, // Version number for incremental training
            //                             recommendations: response.recommendations, // Training recommendations
            //                             algorithmDescription: response.algorithm_description, // Description of the algorithm used
            //                             algorithmStrengths: response.algorithm_strengths,
            //                             training_time: response.training_time,
            //                             dataQualityScore: response.data_quality_score, // Alternative algorithms for this model type
            //                             timestamp: response.timestamp, // When the model was trained
            //                             isActive: true // Whether this model version is active
            //                         });
            //                     }
            //                 }
            //             }
            //         }

            //     }
            // } catch (modelError) {
            //     console.error('Model setup failed:', modelError);
            // }

        } catch (error) {
            console.error('Error creating Racing Session:', error);
        } finally {
            // Clean up upload state from memory to prevent memory leaks
            this.uploadStates.delete(uploadId);

            // Clean up any assembled files for this upload
            await this.cleanupAssembledFile(uploadId);
        }

        return {
            message: 'Upload completed successfully',
            sessionId: uploadId,
            aiAnalysisAvailable: true
        };
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
