import { Injectable, NotFoundException, BadRequestException, Inject, forwardRef } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model, Types } from 'mongoose';
import { UserACCTrackAIModel, SessionAIModelSchema } from 'src/schemas/session-ai-model.schema';
import { UserInfo } from 'src/schemas/user-info.schema';
import { CreateSeesionAIModelDto as CreateACCSeesionAIModelDto, UpdateAiModelDto as UpdateACCAiModelDto, GetAiModelDto as GetACCAiModelDto, IncrementalTrainingDto, ModelPredictionDto, AiModelResponseDto } from 'src/dto/ai-model.dto';
import { AiServiceClient, QueryRequest } from '../../shared/ai/ai-service.client';
import { RacingSessionService } from '../racing-session/racing-session.service';

@Injectable()
export class UserSessionAiModelService {
    constructor(
        @InjectModel(UserACCTrackAIModel.name) private userACCTrackAIModel: Model<UserACCTrackAIModel>,
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

    async createModel(createAiModelDto: CreateACCSeesionAIModelDto): Promise<any> {
        // If setting as active, deactivate other models for the same user/track/type

        if (createAiModelDto.isActive) {
            try {
                await this.deactivateModels(
                    createAiModelDto.userId,
                    createAiModelDto.trackName,
                    createAiModelDto.carName,
                    createAiModelDto.modelType
                );
            } catch (error) {
                console.error('Error deactivating models:', error);
            }
        }

        const newModel = new this.userACCTrackAIModel(createAiModelDto);
        const savedModel = await newModel.save();

        return savedModel;
    }

    async findModelsByUser(getUserDto: GetACCAiModelDto): Promise<UserACCTrackAIModel[]> {
        const query: any = {
            userId: getUserDto.userId,
            trackName: getUserDto.trackName,
            modelType: getUserDto.modelType
        };

        if (getUserDto.activeOnly) {
            query.isActive = true;
        }

        return await this.userACCTrackAIModel.find(query).sort({ trainedAt: -1 });
    }

    /**
     * 
     * @param userId 
     * @param trackName 
     * @param carName 
     * @param modelType 
     * @param target_variable 
     * @returns 
     */
    async findActiveUserSessionAIModel(userId: string, trackName: string, carName: string, modelType: string, target_variable: string): Promise<any | null> {
        const userObjectId = await this.getUserObjectId(userId);
        if (!userObjectId) {
            console.log('User not found with id:', userId);
            return null;
        }

        const data = await this.userACCTrackAIModel.findOne({
            userId: userId,
            carName: carName,
            trackName: trackName,
            modelType: modelType,
            targetVariable: target_variable,
            isActive: true,
        });

        return data;
    }

    async updateModel(modelId: string, updateAiModelDto: UpdateACCAiModelDto): Promise<UserACCTrackAIModel> {
        const model = await this.userACCTrackAIModel.findById(modelId);
        if (!model) {
            throw new NotFoundException('AI Model not found');
        }

        try {
            // Update the model with new values
            Object.assign(model, updateAiModelDto);

            return await model.save();
        } catch (error) {
            throw new BadRequestException('Error updating AI Model: ' + error.message);
        }
    }

    async deleteModel(modelId: string): Promise<void> {
        const result = await this.userACCTrackAIModel.findByIdAndDelete(modelId);
        if (!result) {
            throw new NotFoundException('AI Model not found');
        }
    }

    async deactivateModels(userId: string, trackName: string, carName: string, modelType: string): Promise<void> {
        await this.userACCTrackAIModel.updateMany(
            {
                userId,
                trackName,
                carName,
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
