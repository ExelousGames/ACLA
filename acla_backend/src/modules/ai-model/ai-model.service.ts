
import { Injectable, InternalServerErrorException, NotFoundException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { AIModel } from '../../schemas/ai-model.schema';
import { CreateAiModelDto, UpdateAiModelDto } from './dto/ai-model.dto';

@Injectable()
export class AiModelService {
    constructor(
        @InjectModel(AIModel.name) private aiModelModel: Model<AIModel>,
    ) { }

    async create(createAiModelDto: CreateAiModelDto): Promise<AIModel> {
        const createdModel = new this.aiModelModel(createAiModelDto);
        return createdModel.save();
    }

    async findAll(filters: {
        trackName?: string;
        carName?: string;
        modelType?: string;
        isActive?: boolean;
    } = {}): Promise<AIModel[]> {
        const query: any = {};

        if (filters.trackName) query.trackName = filters.trackName;
        if (filters.carName) query.carName = filters.carName;
        if (filters.modelType) query.modelType = filters.modelType;
        if (filters.isActive !== undefined) query.isActive = filters.isActive;

        return this.aiModelModel.find(query).exec();
    }

    async findOne(id: string): Promise<AIModel> {
        const model = await this.aiModelModel.findById(id).exec();
        if (!model) {
            throw new NotFoundException(`AI Model with ID ${id} not found`);
        }
        return model;
    }

    async update(id: string, updateAiModelDto: UpdateAiModelDto): Promise<AIModel> {
        const updatedModel = await this.aiModelModel
            .findByIdAndUpdate(id, updateAiModelDto, { new: true })
            .exec();

        if (!updatedModel) {
            throw new NotFoundException(`AI Model with ID ${id} not found`);
        }
        return updatedModel;
    }

    async remove(id: string): Promise<void> {
        const result = await this.aiModelModel.findByIdAndDelete(id).exec();
        if (!result) {
            throw new NotFoundException(`AI Model with ID ${id} not found`);
        }
    }

    async getActiveModel(trackName: string, carName: string, modelType: string): Promise<AIModel> {
        const model = await this.aiModelModel
            .findOne({
                trackName,
                carName,
                modelType,
                isActive: true
            })
            .exec();

        if (!model) {
            throw new NotFoundException(
                `Active AI Model not found for track: ${trackName}, car: ${carName}, type: ${modelType}`
            );
        }
        return model;
    }

    async activateModel(id: string): Promise<AIModel> {
        const model = await this.findOne(id);

        // Deactivate all other models with same track, car, and type
        await this.aiModelModel
            .updateMany(
                {
                    trackName: model.trackName,
                    carName: model.carName,
                    modelType: model.modelType,
                    _id: { $ne: id }
                },
                { isActive: false }
            )
            .exec();

        // Activate the target model
        const activatedModel = await this.aiModelModel
            .findByIdAndUpdate(id, { isActive: true }, { new: true })
            .exec();

        return activatedModel!;
    }


    async save_imitation_learning_results(updateAiModelDto: UpdateAiModelDto): Promise<any> {

        const { trackName, carName } = updateAiModelDto;
        if (!trackName || !carName) {
            throw new InternalServerErrorException("Missing trackName or carName in results");
        }
        // Save the results to the database or process as needed
        const existingEntries = await this.aiModelModel.findOne({ carName: updateAiModelDto.carName, trackName: updateAiModelDto.trackName, modelType: updateAiModelDto.modelType, isActive: true });

        if (existingEntries) {
            this.aiModelModel.updateOne({ _id: existingEntries._id }, { $set: { modelData: updateAiModelDto.modelData, metadata: updateAiModelDto.metadata } });
        }
        else {

            const { modelData, metadata } = updateAiModelDto;
            if (!modelData || !metadata) {
                throw new InternalServerErrorException("Incomplete imitation learning results");
            }
            try {
                const createdEntry = new this.aiModelModel({ trackName, carName, modelType: updateAiModelDto.modelType, modelData, metadata, isActive: true });
                return createdEntry.save();

            } catch (error) {
                console.error("Error saving imitation learning results:", error);
                throw new InternalServerErrorException("Failed to save imitation learning results");
            }
        }
        return null;
    }

}