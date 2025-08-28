import { Controller, Get, UseGuards, Request, Post, Body, Query, BadRequestException, Inject, forwardRef } from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { RacingSessionDetailedInfoDto, SessionBasicInfoListDto, UploadReacingSessionInitDto } from 'src/dto/racing-session.dto';
import { AiModelResponseDto } from 'src/dto/ai-model.dto';
import { RacingSessionService } from './racing-session.service';
import { AiService } from '../ai-service/ai-service.service';
import { AiModelService } from '../ai-model/ai-model.service';
import { UserInfoService } from '../user-info/user-info.service';

@Controller('racing-session')
export class RacingSessionController {
    private uploadStates = new Map<string, {
        metadata: UploadReacingSessionInitDto;
        session_data_chunks: string[][];
        received: number;
    }>();

    constructor(
        private racingSessionService: RacingSessionService,
        private aiService: AiService,
        @Inject(forwardRef(() => AiModelService))
        private aiModelService: AiModelService,
        private userInfoService: UserInfoService
    ) { }

    @UseGuards(AuthGuard('jwt'))
    @Post('sessionbasiclist')
    retrieveAllRacingBasicSessionsInfo(@Request() req, @Body() body): Promise<SessionBasicInfoListDto | null> {
        return this.racingSessionService.retrieveAllRacingSessionsInfo(body.map_name, body.username);
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
        const upload = this.uploadStates.get(uploadId);
        if (!upload) {
            throw new BadRequestException('Upload doesnt exist');
        }

        const fullDataset = upload.session_data_chunks.flat();

        console.log(`Upload complete for ID: ${uploadId}, total chunks: ${upload.session_data_chunks.length}, total records: ${fullDataset.length}`);

        // Create racing session in database
        const createdSession = await this.racingSessionService.createRacingSession(
            upload.metadata.sessionName,
            uploadId,
            upload.metadata.mapName,
            upload.metadata.userEmail,
            fullDataset
        );

        // Check for active AI model first, before processing the session
        let activeModel: AiModelResponseDto | null = null;
        let modelReady = false;

        try {
            // First, find the user by email to get their ObjectId
            const userInfo = await this.userInfoService.findOne(upload.metadata.userEmail);
            if (!userInfo) {
                console.log('User not found for email:', upload.metadata.userEmail);
            } else {
                const userId = (userInfo as any).id.toString();

                // Check if user has an active model for this track
                activeModel = await this.aiModelService.findActiveModel(
                    userId,
                    upload.metadata.mapName,
                    'lap_time_prediction'
                );

                if (activeModel) {
                    console.log('Found active model for user:', userId, 'track:', upload.metadata.mapName);
                    modelReady = true;
                } else {
                    console.log('No active model found, training new model from scratch...');

                    // Train a new model from scratch
                    const trainingResult = await this.aiModelService.trainFromScratch({
                        userId: userId,
                        trackName: upload.metadata.mapName,
                        modelType: 'lap_time_prediction',
                        sessionIds: [uploadId],
                        modelName: `${upload.metadata.mapName}_lap_prediction_${new Date().toISOString().split('T')[0]}`,
                        description: `Lap time prediction model for ${upload.metadata.mapName}`,
                        hyperparameters: {
                            // Add default hyperparameters for training
                            learning_rate: 0.001,
                            batch_size: 32,
                            epochs: 100,
                            validation_split: 0.2
                        }
                    });

                    if (trainingResult && trainingResult.id) {
                        // Get the newly trained model
                        activeModel = trainingResult;
                        modelReady = true;
                        console.log('New model trained successfully:', trainingResult.id);
                    } else {
                        console.error('Failed to train new model');
                        modelReady = false;
                    }
                }
            }
        } catch (modelError) {
            console.error('Model setup failed:', modelError);
            modelReady = false;
        }

        // Process the session with AI - either train/update model or do general analysis
        try {
            if (activeModel && modelReady) {
                // If we have an active model, update it with incremental training
                console.log('Updating existing model with new session data:', uploadId);
                try {
                    const incrementalResult = await this.aiModelService.incrementalTraining({
                        modelId: activeModel.id,
                        newSessionIds: [uploadId],
                        validateModel: true
                    });

                    console.log('Model updated with new session data:', incrementalResult.modelVersion);
                } catch (incrementalError) {
                    console.error('Incremental training failed:', incrementalError);
                    // If incremental training fails, create a new model from scratch
                    console.log('Creating new model due to incremental training failure...');
                    try {
                        const userInfo = await this.userInfoService.findOne(upload.metadata.userEmail);
                        if (userInfo) {
                            const userId = (userInfo as any).id.toString();

                            const newTrainingResult = await this.aiModelService.trainFromScratch({
                                userId: userId,
                                trackName: upload.metadata.mapName,
                                modelType: 'lap_time_prediction',
                                sessionIds: [uploadId],
                                modelName: `${upload.metadata.mapName}_lap_prediction_${new Date().toISOString().split('T')[0]}_recovery`,
                                description: `Recovery model for ${upload.metadata.mapName} after incremental training failure`,
                                hyperparameters: {
                                    learning_rate: 0.001,
                                    batch_size: 32,
                                    epochs: 100,
                                    validation_split: 0.2
                                }
                            });

                            if (newTrainingResult && newTrainingResult.id) {
                                console.log('Recovery model created successfully:', newTrainingResult.id);
                                activeModel = newTrainingResult;
                                modelReady = true;
                            } else {
                                console.error('Failed to create recovery model');
                            }
                        }
                    } catch (recoveryError) {
                        console.error('Recovery model creation failed:', recoveryError);
                        console.error('All AI model operations failed for session:', uploadId);
                    }
                }
            } else {
                // No model available, train a new model from scratch
                console.log('No active model found, training new model from scratch for session:', uploadId);
                try {
                    const userInfo = await this.userInfoService.findOne(upload.metadata.userEmail);
                    if (userInfo) {
                        const userId = (userInfo as any).id.toString();

                        const newTrainingResult = await this.aiModelService.trainFromScratch({
                            userId: userId,
                            trackName: upload.metadata.mapName,
                            modelType: 'lap_time_prediction',
                            sessionIds: [uploadId],
                            modelName: `${upload.metadata.mapName}_lap_prediction_${new Date().toISOString().split('T')[0]}`,
                            description: `Lap time prediction model for ${upload.metadata.mapName}`,
                            hyperparameters: {
                                learning_rate: 0.001,
                                batch_size: 32,
                                epochs: 100,
                                validation_split: 0.2
                            }
                        });

                        if (newTrainingResult && newTrainingResult.id) {
                            console.log('New model trained successfully:', newTrainingResult.id);
                        } else {
                            console.error('Failed to train new model');
                        }
                    } else {
                        console.error('User not found for email:', upload.metadata.userEmail);
                    }
                } catch (trainingError) {
                    console.error('Training new model failed:', trainingError);
                }
            }

        } catch (error) {
            console.error('AI processing failed for session:', uploadId, error);
            // Don't fail the upload if AI processing fails
        }

        this.uploadStates.delete(uploadId);

        return {
            message: 'Upload completed successfully',
            sessionId: uploadId,
            aiAnalysisAvailable: true
        };
    }
}
