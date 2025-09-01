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

    @Prop()
    algorithmType: string; // regression, classification

    @Prop({ type: Object })
    trainingMetrics: Record<string, any>; // Model performance metrics

    @Prop()
    samplesProcessed: number;

    @Prop({ type: [String], required: true })
    featureNames: string[]; // List of feature names used

    @Prop()
    featureCount: number; // Number of features

    @Prop({ required: true, default: "1" })
    modelVersion: string; // Version number for incremental training

    @Prop({ type: [String], default: [] })
    recommendations: string[]; // Training recommendations

    @Prop({ default: '' })
    algorithmDescription: string; // Description of the algorithm used

    @Prop({ type: [String], default: [] })
    algorithmStrengths: string[];

    @Prop({ default: '' })
    training_time: string;

    @Prop({ default: '' })
    dataQualityScore: number;

    @Prop({ default: '' })
    timestamp: string;

    @Prop({ required: true, default: true })
    isActive: boolean; // Whether this model version is active

}

export const SessionAIModelSchema = SchemaFactory.createForClass(UserTrackAIModel);

// Create compound indexes for efficient queries
SessionAIModelSchema.index({ userId: 1, trackName: 1, modelType: 1, targetVariable: 1, isActive: 1 });

