import { Controller, Get, UseGuards, Request, Post, Body, Query, BadRequestException, Inject, forwardRef } from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { RacingSessionDetailedInfoDto, SessionBasicInfoListDto, UploadReacingSessionInitDto } from 'src/dto/racing-session.dto';
import { AiModelResponseDto } from 'src/dto/ai-model.dto';
import { RacingSessionService } from './racing-session.service';
import { AiModelService } from '../ai-model/ai-model.service';
import { UserInfoService } from '../user-info/user-info.service';
import { UserTrackAIModel } from 'src/schemas/session-ai-model.schema';
import { AiServiceClient, ModelsConfig, TrainModelsResponse } from '../ai-model/ai-service.client';
import { model, Types } from 'mongoose';

@Controller('racing-session')
export class RacingSessionController {
    private uploadStates = new Map<string, {
        metadata: UploadReacingSessionInitDto;
        session_data_chunks: string[][];
        received: number;
    }>();

    constructor(
        private racingSessionService: RacingSessionService,
        @Inject(forwardRef(() => AiModelService))
        private aiModelService: AiModelService,
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
                const userInfo = await this.userInfoService.findOne(upload.metadata.userId);

                if (!userInfo) {
                    console.log('User not found for email:', upload.metadata.userId);
                } else {

                    const userId = upload.metadata.userId;
                    // Check for active AI model first, before processing the session

                    //list of ai models will be trained
                    const modelsConfig: ModelsConfig[] = [
                        { config_id: "lap_prediction", target_variable: "Graphics_current_time", model_type: "lap_time_prediction" }
                    ];

                    let activeModel: UserTrackAIModel & { _id: Types.ObjectId; } | null = null;

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
