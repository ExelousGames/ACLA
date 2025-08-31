import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Types } from 'mongoose';


@Schema({ timestamps: true })
export class UserTrackAIModel {

    @Prop({ type: String, ref: 'UserInfo', required: true })
    userId: string;

    @Prop({ required: true })
    trackName: string;

    @Prop({ required: true })
    carName: string;

    @Prop({ required: true })
    modelType: string; // lap_time_prediction, sector_time_optimization, etc.

    @Prop({ required: true })
    targetVariable: string; // lap_time, sector_time, etc.

    @Prop({ required: true })
    modelData: string; // Base64 encoded serialized model

    @Prop({ required: true })
    algorithmUsed: string; // random_forest, gradient_boosting, neural_network, etc.

    @Prop({ required: true })
    algorithmType: string; // regression, classification

    @Prop({ type: Object, required: true })
    trainingMetrics: Record<string, any>; // Model performance metrics

    @Prop({ type: [String], required: true })
    featureNames: string[]; // List of feature names used

    @Prop({ required: true })
    featureCount: number; // Number of features

    @Prop({ required: true })
    trainingSamples: number; // Number of training samples

    @Prop({ required: true, default: "1" })
    modelVersion: string; // Version number for incremental training

    @Prop({ type: Object, default: {} })
    telemetrySummary: Record<string, any>; // Summary of telemetry data used

    @Prop({ type: [String], default: [] })
    recommendations: string[]; // Training recommendations

    @Prop({ default: '' })
    algorithmDescription: string; // Description of the algorithm used

    @Prop({ required: true, default: true })
    supportsIncremental: boolean; // Whether model supports incremental learning

    @Prop({ type: Object, default: {} })
    featureImportance: Record<string, any>; // Feature importance scores

    @Prop({ type: [String], default: [] })
    alternativeAlgorithms: string[]; // Alternative algorithms for this model type

    @Prop({ required: true })
    trainedAt: string; // When the model was trained

    @Prop({ required: true, default: true })
    isActive: boolean; // Whether this model version is active

}

export const SessionAIModelSchema = SchemaFactory.createForClass(UserTrackAIModel);

// Create compound indexes for efficient queries
SessionAIModelSchema.index({ userId: 1, trackName: 1, modelType: 1, isActive: 1 });
SessionAIModelSchema.index({ userId: 1, trackName: 1, modelVersion: -1 });
SessionAIModelSchema.index({ userId: 1, isActive: 1 });
SessionAIModelSchema.index({ trackName: 1, modelType: 1 });
SessionAIModelSchema.index({ algorithmUsed: 1, modelType: 1 });
SessionAIModelSchema.index({ trainedAt: -1 });
SessionAIModelSchema.index({ isActive: 1, trainedAt: -1 });
