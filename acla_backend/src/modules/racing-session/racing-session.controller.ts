import { Controller, Get, UseGuards, Request, Post, Body, Query, BadRequestException, Inject, forwardRef } from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { RacingSessionDetailedInfoDto, SessionBasicInfoListDto, UploadReacingSessionInitDto, AllSessionsInitResponseDto, SessionChunkDto, AllSessionsChunkRequestDto } from 'src/dto/racing-session.dto';
import { AiModelResponseDto } from 'src/dto/ai-model.dto';
import { RacingSessionService } from './racing-session.service';
import { UserSessionAiModelService } from '../user-session-ai-model/user-session-ai-model.service';
import { UserInfoService } from '../user-info/user-info.service';
import { UserACCTrackAIModel } from 'src/schemas/session-ai-model.schema';
import { AiServiceClient, ModelsConfig, TrainModelsResponse } from '../../shared/ai/ai-service.client';
import { model, Types } from 'mongoose';

@Controller('racing-session')
export class RacingSessionController {
    private uploadStates = new Map<string, {
        metadata: UploadReacingSessionInitDto;
        session_data_chunks: string[][];
        received: number;
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
    ) { }

    @UseGuards(AuthGuard('jwt'))
    @Post('sessionbasiclist')
    retrieveAllRacingBasicSessionsInfo(@Request() req, @Body() body): Promise<SessionBasicInfoListDto | null> {
        return this.racingSessionService.retrieveAllRacingSessionsBasicInfo(body.map_name, body.username);
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
        @Body() body: { trackName: string, carName: string, chunkSize?: number }
    ): Promise<AllSessionsInitResponseDto> {
        try {
            const chunkSize = body.chunkSize || 1000; // Default chunk size
            const initData = await this.racingSessionService.initializeSessionsDownload(body.trackName, body.carName, chunkSize);

            // Store download state for tracking
            this.downloadStates.set(initData.downloadId, {
                initData,
                downloadedChunks: new Set<string>(),
                createdAt: new Date()
            });

            // Clean up old download states (older than 1 hour)
            this.cleanupOldDownloadStates();

            return initData;
        } catch (error) {
            throw new BadRequestException(`Failed to initialize download: ${error.message}`);
        }
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('download/chunk')
    async downloadSessionChunk(
        @Request() req,
        @Body() body: AllSessionsChunkRequestDto
    ): Promise<SessionChunkDto> {
        try {
            // Validate download state exists
            const downloadState = this.downloadStates.get(body.downloadId);
            if (!downloadState) {
                throw new BadRequestException('Download session not found or expired');
            }

            // Validate session exists in the download
            const sessionExists = downloadState.initData.sessionMetadata.some(
                session => session.sessionId === body.sessionId
            );
            if (!sessionExists) {
                throw new BadRequestException('Session not found in download');
            }

            const chunkSize = 1000; // Use consistent chunk size
            const chunk = await this.racingSessionService.getSessionChunk(
                body.sessionId,
                body.chunkIndex,
                chunkSize
            );

            // Set the download ID from the request
            chunk.downloadId = body.downloadId;

            // Track downloaded chunk
            const chunkKey = `${body.sessionId}:${body.chunkIndex}`;
            downloadState.downloadedChunks.add(chunkKey);

            return chunk;
        } catch (error) {
            throw new BadRequestException(`Failed to download chunk: ${error.message}`);
        }
    }

    /**
     * Clean up download states older than 1 hour
     */
    private cleanupOldDownloadStates(): void {
        const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000);

        for (const [downloadId, state] of this.downloadStates.entries()) {
            if (state.createdAt < oneHourAgo) {
                this.downloadStates.delete(downloadId);
            }
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
            throw new BadRequestException('Download session not found or expired');
        }

        const totalPossibleChunks = downloadState.initData.totalChunks;
        const downloadedChunks = downloadState.downloadedChunks.size;
        const progress = totalPossibleChunks > 0 ? (downloadedChunks / totalPossibleChunks) * 100 : 0;

        return {
            downloadId: body.downloadId,
            totalSessions: downloadState.initData.totalSessions,
            totalChunks: totalPossibleChunks,
            downloadedChunks,
            progress: Math.round(progress * 100) / 100, // Round to 2 decimal places
            isComplete: downloadedChunks >= totalPossibleChunks,
            createdAt: downloadState.createdAt
        };
    }

    @UseGuards(AuthGuard('jwt'))
    @Post('upload/init')
    async initUpload(@Body() metadata: UploadReacingSessionInitDto) {
        const uploadId = crypto.randomUUID();

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

            // ai training
            try {
                // First, find the user by email to get their ObjectId
                const userInfo = await this.userInfoService.findOneWithEmail(upload.metadata.userId);

                if (!userInfo) {
                    console.log('User not found for email:', upload.metadata.userId);
                } else {

                    const userId = upload.metadata.userId;
                    // Check for active AI model first, before processing the session

                    //list of ai models will be trained
                    const modelsConfig: ModelsConfig[] = [
                        { config_id: "lap_prediction", target_variable: "Graphics_current_time", model_type: "lap_time_prediction" }
                    ];

                    let activeModel: UserACCTrackAIModel & { _id: Types.ObjectId; } | null = null;

                    // Check for existing active models for each config
                    for (const modelConfig of modelsConfig) {

                        // Check if user has an active model for this track
                        activeModel = await this.aiModelService.findActiveUserSessionAIModel(
                            userId,
                            upload.metadata.mapName,
                            upload.metadata.carName,
                            modelConfig.model_type,
                            modelConfig.target_variable
                        );

                        // add active model if any
                        modelConfig.existing_model_data = activeModel ? activeModel : null;
                    }

                    // Train the models using the AI service client
                    const trainedModelsResponse: TrainModelsResponse = await this.aiServiceClient.trainModels({
                        session_id: createdSession.id,
                        telemetry_data: fullDataset,
                        models_config: modelsConfig,
                        user_id: userId,
                        parallel_training: false
                    });

                    //save the training result to database
                    if (trainedModelsResponse) {
                        // Process and save the training result as needed
                        for (const [configId, response] of Object.entries(trainedModelsResponse.training_results)) {

                            //find the matching config
                            const modelConfig = modelsConfig.find(config => config.config_id === configId);
                            console.log("Training result for response:", response);
                            if (response.success) {
                                //if there is an active model, update the model in database
                                if (modelConfig && modelConfig.existing_model_data) {

                                    await this.aiModelService.updateModel(modelConfig.existing_model_data._id.toString(), {
                                        modelData: response.model_data,
                                        modelType: response.model_type,
                                        algorithmUsed: response.algorithm_used,
                                        algorithmType: response.algorithm_type,
                                        targetVariable: response.target_variable,
                                        trainingMetrics: response.training_metrics,
                                        featureNames: response.feature_names,
                                        featureCount: response.features_count,
                                        samplesProcessed: response.samples_processed,
                                        modelVersion: response.model_version, // Version number for incremental training
                                        algorithmStrengths: response.algorithm_strengths, // Summary of telemetry data used
                                        recommendations: response.recommendations, // Training recommendations
                                        algorithmDescription: response.algorithm_description, // Description of the algorithm used
                                        training_time: response.training_time, // Training time
                                        dataQualityScore: response.data_quality_score, // Feature importance scores
                                        timestamp: response.timestamp, // When the model was trained
                                        isActive: true // Whether this model version is active
                                    });
                                } else {
                                    //else create a new model
                                    await this.aiModelService.createModel({
                                        userId: userId,
                                        trackName: upload.metadata.mapName,
                                        carName: upload.metadata.carName,
                                        modelData: response.model_data,
                                        modelType: response.model_type,
                                        algorithmUsed: response.algorithm_used,
                                        algorithmType: response.algorithm_type,
                                        targetVariable: response.target_variable,
                                        trainingMetrics: response.training_metrics,
                                        featureNames: response.feature_names,
                                        featureCount: response.features_count,
                                        samplesProcessed: response.samples_processed,
                                        modelVersion: response.model_version, // Version number for incremental training
                                        recommendations: response.recommendations, // Training recommendations
                                        algorithmDescription: response.algorithm_description, // Description of the algorithm used
                                        algorithmStrengths: response.algorithm_strengths,
                                        training_time: response.training_time,
                                        dataQualityScore: response.data_quality_score, // Alternative algorithms for this model type
                                        timestamp: response.timestamp, // When the model was trained
                                        isActive: true // Whether this model version is active
                                    });
                                }
                            }
                        }
                    }

                }
            } catch (modelError) {
                console.error('Model setup failed:', modelError);
            }

        } catch (error) {
            console.error('Error creating Racing Session:', error);
        }
        return {
            message: 'Upload completed successfully',
            sessionId: uploadId,
            aiAnalysisAvailable: true
        };
    }
}
