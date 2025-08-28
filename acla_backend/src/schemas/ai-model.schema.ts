import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document, Types } from 'mongoose';

export type AiModelDocument = AiModel & Document & {
    _id: Types.ObjectId;
    createdAt: Date;
    updatedAt: Date;
};

@Schema({ timestamps: true })
export class AiModel {
    @Prop({ type: Types.ObjectId, ref: 'UserInfo', required: true })
    userId: Types.ObjectId;

    @Prop({ required: true })
    trackName: string;

    @Prop({ required: true })
    modelName: string;

    @Prop({ required: true })
    modelVersion: string;

    @Prop({ required: true, type: Object })
    modelData: any; // Serialized model data (weights, parameters, etc.)

    @Prop({ required: true, type: Object })
    modelMetadata: {
        trainingSessionsCount: number;
        lastTrainingDate: Date;
        performanceMetrics: any;
        modelType: string; // 'lap_time_prediction', 'sector_analysis', 'setup_optimization', etc.
        accuracy?: number;
        mse?: number;
        features: string[];
        hyperparameters?: any;
    };

    @Prop({ required: true, type: [String] })
    trainingSessionIds: string[]; // IDs of racing sessions used for training

    @Prop({ required: true })
    isActive: boolean; // Whether this is the active model for this user/track combination

    @Prop()
    description?: string;

    @Prop({ type: Object })
    validationResults?: any; // Cross-validation or test set results

    @Prop({ type: Object })
    featureImportance?: any; // Feature importance scores

    @Prop()
    modelSize?: number; // Size in bytes

    @Prop()
    trainingDuration?: number; // Training time in milliseconds
}

export const AiModelSchema = SchemaFactory.createForClass(AiModel);

// Create compound indexes for efficient queries
AiModelSchema.index({ userId: 1, trackName: 1, 'modelMetadata.modelType': 1, isActive: 1 });
AiModelSchema.index({ userId: 1, trackName: 1, modelVersion: -1 });
AiModelSchema.index({ userId: 1, isActive: 1 });
AiModelSchema.index({ trackName: 1, 'modelMetadata.modelType': 1 });
