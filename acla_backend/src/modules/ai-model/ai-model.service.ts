import { Injectable, NotFoundException, BadRequestException, Inject, forwardRef } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model, Types } from 'mongoose';
import { AiModel, AiModelDocument } from 'src/schemas/ai-model.schema';
import { UserInfo } from 'src/schemas/user-info.schema';
import { CreateAiModelDto, UpdateAiModelDto, GetAiModelDto, IncrementalTrainingDto, ModelPredictionDto, AiModelResponseDto } from 'src/dto/ai-model.dto';
import { AiServiceClient, QueryRequest, AnalysisRequest, DatasetUploadRequest } from './ai-service.client';
import { RacingSessionService } from '../racing-session/racing-session.service';

@Injectable()
export class AiModelService {
    constructor(
        @InjectModel(AiModel.name) private aiModelModel: Model<AiModelDocument>,
        @InjectModel(UserInfo.name) private userInfoModel: Model<UserInfo>,
        private aiServiceClient: AiServiceClient,
        @Inject(forwardRef(() => RacingSessionService))
        private racingSessionService: RacingSessionService,
    ) { }

    /**
     * Helper method to convert a user UUID to MongoDB ObjectId
     */
    private async getUserObjectId(userUuid: string): Promise<Types.ObjectId | null> {
        const user = await this.userInfoModel.findOne({ id: userUuid }).exec();
        return user ? (user as any)._id : null;
    }

    async createModel(createAiModelDto: CreateAiModelDto): Promise<AiModelResponseDto> {
        // If setting as active, deactivate other models for the same user/track/type
        if (createAiModelDto.isActive) {
            await this.deactivateModels(
                createAiModelDto.userId,
                createAiModelDto.trackName,
                createAiModelDto.modelMetadata.modelType
            );
        }

        const newModel = new this.aiModelModel(createAiModelDto);
        const savedModel = await newModel.save();

        return this.mapToResponseDto(savedModel);
    }

    async findModelsByUser(getUserDto: GetAiModelDto): Promise<AiModelResponseDto[]> {
        const query: any = {
            userId: getUserDto.userId,
            trackName: getUserDto.trackName,
        };

        if (getUserDto.modelType) {
            query['modelMetadata.modelType'] = getUserDto.modelType;
        }

        if (getUserDto.activeOnly) {
            query.isActive = true;
        }

        const models = await this.aiModelModel.find(query).sort({ createdAt: -1 });
        return models.map(model => this.mapToResponseDto(model));
    }

    async findActiveModel(userId: string, trackName: string, modelType: string): Promise<AiModelResponseDto | null> {
        const userObjectId = await this.getUserObjectId(userId);
        if (!userObjectId) {
            console.log('User not found with id:', userId);
            return null;
        }

        const model = await this.aiModelModel.findOne({
            userId: userObjectId,
            trackName,
            'modelMetadata.modelType': modelType,
            isActive: true,
        });

        return model ? this.mapToResponseDto(model) : null;
    }

    async updateModel(modelId: string, updateAiModelDto: UpdateAiModelDto): Promise<AiModelResponseDto> {
        const model = await this.aiModelModel.findById(modelId);
        if (!model) {
            throw new NotFoundException('AI Model not found');
        }

        // If setting as active, deactivate other models
        if (updateAiModelDto.isActive) {
            await this.deactivateModels(
                model.userId,
                model.trackName,
                model.modelMetadata.modelType
            );
        }

        Object.assign(model, updateAiModelDto);
        const updatedModel = await model.save();

        return this.mapToResponseDto(updatedModel);
    }

    async deleteModel(modelId: string): Promise<void> {
        const result = await this.aiModelModel.findByIdAndDelete(modelId);
        if (!result) {
            throw new NotFoundException('AI Model not found');
        }
    }

    async incrementalTraining(incrementalTrainingDto: IncrementalTrainingDto): Promise<AiModelResponseDto> {
        const model = await this.aiModelModel.findById(incrementalTrainingDto.modelId);
        if (!model) {
            throw new NotFoundException('AI Model not found');
        }

        // Get new racing session data
        const newSessionsData: any[] = [];
        for (const sessionId of incrementalTrainingDto.newSessionIds) {
            const sessionData = await this.racingSessionService.retrieveSessionDetailedInfo(sessionId);
            if (sessionData) {
                newSessionsData.push(sessionData.data);
            }
        }

        if (newSessionsData.length === 0) {
            throw new BadRequestException('No valid racing sessions found for training');
        }

        try {
            // Send incremental training request to AI service
            const trainingRequest = {
                modelId: (model._id as any).toString(),
                existingModelData: model.modelData,
                newTrainingData: newSessionsData,
                modelMetadata: model.modelMetadata,
                trainingParameters: incrementalTrainingDto.trainingParameters || {},
            };

            const trainingResult = await this.aiServiceClient.incrementalTraining(trainingRequest);

            // Update model with new data
            const updatedModelData = {
                modelData: trainingResult.updatedModelData,
                modelMetadata: {
                    ...model.modelMetadata,
                    trainingSessionsCount: model.modelMetadata.trainingSessionsCount + newSessionsData.length,
                    lastTrainingDate: new Date(),
                    performanceMetrics: trainingResult.performanceMetrics,
                    accuracy: trainingResult.accuracy,
                    mse: trainingResult.mse,
                },
                trainingSessionIds: [...model.trainingSessionIds, ...incrementalTrainingDto.newSessionIds],
                validationResults: trainingResult.validationResults,
                featureImportance: trainingResult.featureImportance,
                trainingDuration: trainingResult.trainingDuration,
            };

            // Create new version of the model
            const newVersion = this.generateNewVersion(model.modelVersion);
            const newModel = new this.aiModelModel({
                ...model.toObject(),
                _id: undefined,
                modelVersion: newVersion,
                ...updatedModelData,
                isActive: true, // New version becomes active
            });

            // Deactivate old version
            model.isActive = false;
            await model.save();

            // Save new version
            const savedNewModel = await newModel.save();

            return this.mapToResponseDto(savedNewModel);

        } catch (error) {
            throw new BadRequestException(`Incremental training failed: ${error.message}`);
        }
    }

    async trainFromScratch(trainingDto: {
        userId: string;
        trackName: string;
        modelType: string;
        sessionIds: string[];
        modelName: string;
        description?: string;
        hyperparameters?: any;
    }): Promise<AiModelResponseDto> {
        try {
            // Get racing session data for training
            const sessionData: any[] = [];
            for (const sessionId of trainingDto.sessionIds) {
                const session = await this.racingSessionService.retrieveSessionDetailedInfo(sessionId);
                if (session) {
                    sessionData.push(session.data);
                }
            }

            if (sessionData.length === 0) {
                throw new BadRequestException('No valid racing sessions found for training');
            }

            // Deactivate existing active models for this user/track/type
            const userObjectId = await this.getUserObjectId(trainingDto.userId);
            if (!userObjectId) {
                throw new BadRequestException('User not found');
            }
            await this.deactivateModels(userObjectId, trainingDto.trackName, trainingDto.modelType);

            // Send training request to AI service
            const trainingRequest = {
                session_data: sessionData,
                model_type: trainingDto.modelType,
                training_parameters: trainingDto.hyperparameters || {}
            };

            const trainingResult = await this.aiServiceClient.trainModelFromScratch(trainingRequest);

            // Create model in database
            const createModelDto: CreateAiModelDto = {
                userId: userObjectId,
                trackName: trainingDto.trackName,
                modelName: trainingDto.modelName,
                modelVersion: '1.0.0',
                modelData: trainingResult.model_data,
                modelMetadata: {
                    trainingSessionsCount: sessionData.length,
                    lastTrainingDate: new Date(),
                    performanceMetrics: trainingResult.performance_metrics,
                    modelType: trainingDto.modelType,
                    accuracy: trainingResult.model_metadata?.accuracy,
                    mse: trainingResult.model_metadata?.mse,
                    features: trainingResult.model_metadata?.features || [],
                    hyperparameters: trainingDto.hyperparameters
                },
                trainingSessionIds: trainingDto.sessionIds,
                isActive: true,
                description: trainingDto.description || `Initial ${trainingDto.modelType} model for ${trainingDto.trackName}`,
                validationResults: trainingResult.validation_results,
                featureImportance: trainingResult.feature_importance,
                trainingDuration: trainingResult.training_duration
            };

            const newModel = new this.aiModelModel(createModelDto);
            const savedModel = await newModel.save();

            return this.mapToResponseDto(savedModel);

        } catch (error) {
            throw new BadRequestException(`Model training from scratch failed: ${error.message}`);
        }
    }

    async deactivateModels(userId: Types.ObjectId, trackName: string, modelType: string): Promise<void> {
        await this.aiModelModel.updateMany(
            {
                userId,
                trackName,
                'modelMetadata.modelType': modelType,
                isActive: true,
            },
            { isActive: false }
        );
    }

    private generateNewVersion(currentVersion: string): string {
        const versionParts = currentVersion.split('.');
        const majorVersion = parseInt(versionParts[0]) || 1;
        const minorVersion = parseInt(versionParts[1]) || 0;
        const patchVersion = parseInt(versionParts[2]) || 0;

        return `${majorVersion}.${minorVersion}.${patchVersion + 1}`;
    }

    private mapToResponseDto(model: AiModelDocument): AiModelResponseDto {
        return {
            id: (model._id as any).toString(),
            userId: model.userId,
            trackName: model.trackName,
            modelName: model.modelName,
            modelVersion: model.modelVersion,
            modelMetadata: model.modelMetadata,
            trainingSessionIds: model.trainingSessionIds,
            isActive: model.isActive,
            description: model.description,
            validationResults: model.validationResults,
            featureImportance: model.featureImportance,
            modelSize: model.modelSize,
            trainingDuration: model.trainingDuration,
            createdAt: (model as any).createdAt,
            updatedAt: (model as any).updatedAt,
        };
    }

    async processAIQuery(queryRequest: any): Promise<any> {
        try {
            // Forward the AI query to the AI service
            const result = await this.aiServiceClient.processQuery(queryRequest);
            return {
                success: true,
                query: queryRequest.question,
                answer: result.answer,
                function_calls: result.function_calls || [],
                context: result.context,
                ai_processing: true
            };
        } catch (error) {
            throw new BadRequestException(`AI query processing failed: ${error.message}`);
        }
    }

    async analyzeRacingSession(sessionData: any) {
        return await this.aiServiceClient.analyzeRacingSession(sessionData);
    }

    async healthCheck() {
        return await this.aiServiceClient.checkHealth();
    }

