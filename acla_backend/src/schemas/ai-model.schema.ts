import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Schema as MongooseSchema } from 'mongoose';
import { ObjectId } from 'mongodb';


@Schema({ timestamps: true })
export class AIModel {

    @Prop({ required: true })
    trackName: string;

    @Prop({ required: true })
    carName: string;

    @Prop({ required: true })
    modelType: string; // lap_time_prediction, sector_time_optimization, etc.

    @Prop({ type: MongooseSchema.Types.ObjectId })
    modelDataFileId: ObjectId; // GridFS file ID for model data

    @Prop({ type: MongooseSchema.Types.Mixed })
    metadata: any;

    @Prop({ required: true, default: true })
    isActive: boolean; // Whether this model version is active

}

export const AIModelSchema = SchemaFactory.createForClass(AIModel);

// Create compound indexes for efficient queries
AIModelSchema.index({ trackName: 1, carName: 1, modelType: 1, isActive: 1 });

