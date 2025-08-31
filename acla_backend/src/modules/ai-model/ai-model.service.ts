import { Injectable, NotFoundException, BadRequestException, Inject, forwardRef } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model, Types } from 'mongoose';
import { SessionAIModel, SessionAIModelSchema } from 'src/schemas/session-ai-model.schema';
import { UserInfo } from 'src/schemas/user-info.schema';
import { CreateAiModelDto, UpdateAiModelDto, GetAiModelDto, IncrementalTrainingDto, ModelPredictionDto, AiModelResponseDto } from 'src/dto/ai-model.dto';
import { AiServiceClient, QueryRequest } from './ai-service.client';
import { RacingSessionService } from '../racing-session/racing-session.service';

@Injectable()
export class AiModelService {
    constructor(
        @InjectModel(SessionAIModel.name) private sessionAIModelModel: Model<SessionAIModel>,
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

    async createModel(createAiModelDto: CreateAiModelDto): Promise<SessionAIModel> {
        // If setting as active, deactivate other models for the same user/track/type
        if (createAiModelDto.isActive) {
            await this.deactivateModels(
                createAiModelDto.userId,
                createAiModelDto.trackName,
                createAiModelDto.modelType
            );
        }

        const newModel = new this.sessionAIModelModel(createAiModelDto);
        const savedModel = await newModel.save();

        return savedModel;
    }

    async findModelsByUser(getUserDto: GetAiModelDto): Promise<SessionAIModel[]> {
        const query: any = {
            userId: getUserDto.userId,
            trackName: getUserDto.trackName,
            modelType: getUserDto.modelType
        };

        if (getUserDto.activeOnly) {
            query.isActive = true;
        }

        return await this.sessionAIModelModel.find(query).sort({ createdAt: -1 });
    }

    async findActiveModel(userId: string, trackName: string, modelType: string): Promise<SessionAIModel | null> {
        const userObjectId = await this.getUserObjectId(userId);
        if (!userObjectId) {
            console.log('User not found with id:', userId);
            return null;
        }

        const model = await this.sessionAIModelModel.findOne({
            userId: userObjectId,
            trackName,
            modelType: modelType,
            isActive: true,
        });

        return model ? model : null;
    }

    async updateModel(modelId: string, updateAiModelDto: UpdateAiModelDto): Promise<SessionAIModel> {
        const model = await this.sessionAIModelModel.findById(modelId);
        if (!model) {
            throw new NotFoundException('AI Model not found');
        }

        // If updating to active, deactivate other models for the same user/track/type
        if (updateAiModelDto.isActive === true) {
            await this.deactivateModels(
                model.userId,
                model.trackName,
                model.modelType
            );
        }

        // Validate feature consistency if updating feature-related fields
        if (updateAiModelDto.featureNames && updateAiModelDto.featureCount) {
            if (updateAiModelDto.featureNames.length !== updateAiModelDto.featureCount) {
                throw new BadRequestException('Feature count does not match the number of feature names');
            }
        } else if (updateAiModelDto.featureNames) {
            updateAiModelDto.featureCount = updateAiModelDto.featureNames.length;
        } else if (updateAiModelDto.featureCount && model.featureNames) {
            if (updateAiModelDto.featureCount !== model.featureNames.length) {
                throw new BadRequestException('Feature count does not match existing feature names');
            }
        }

        Object.assign(model, updateAiModelDto);
        return await model.save();
    }

    async getModelPerformanceMetrics(modelId: string): Promise<any> {
        const model = await this.sessionAIModelModel.findById(modelId);
        if (!model) {
            throw new NotFoundException('AI Model not found');
        }

        return {
            modelId: modelId,
            modelType: model.modelType,
            algorithmUsed: model.algorithmUsed,
            algorithmType: model.algorithmType,
            targetVariable: model.targetVariable,
            trainingMetrics: model.trainingMetrics,
            featureImportance: model.featureImportance,
            featureCount: model.featureCount,
            trainingSamples: model.trainingSamples,
            modelVersion: model.modelVersion,
            isActive: model.isActive,
            trainedAt: model.trainedAt,
            supportsIncremental: model.supportsIncremental,
            recommendations: model.recommendations,
            alternativeAlgorithms: model.alternativeAlgorithms
        };
    }

    async deleteModel(modelId: string): Promise<void> {
        const result = await this.sessionAIModelModel.findByIdAndDelete(modelId);
        if (!result) {
            throw new NotFoundException('AI Model not found');
        }
    }

    async deactivateModels(userId: Types.ObjectId, trackName: string, modelType: string): Promise<void> {
        await this.sessionAIModelModel.updateMany(
            {
                userId,
                trackName,
                modelType,
                isActive: true,
            },
            { isActive: false }
        );
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
    async healthCheck() {
        return await this.aiServiceClient.checkHealth();
    }
}
