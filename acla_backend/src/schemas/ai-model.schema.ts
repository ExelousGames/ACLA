import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Schema as MongooseSchema } from 'mongoose';
import { ObjectId } from 'mongodb';


@Schema({ timestamps: true })
export class AIModel {

    @Prop()
    trackName: string;

    @Prop()
    carName: string;

    @Prop({ required: true })
    modelType: string; // lap_time_prediction, sector_time_optimization, etc.

    @Prop({ type: MongooseSchema.Types.ObjectId, required: true })
    modelDataFileId: ObjectId; // GridFS file ID for model data

    @Prop({ type: MongooseSchema.Types.Mixed })
    metadata: any;

    @Prop({ required: true, default: true })
    isActive: boolean; // Whether this model version is active

}

export const AIModelSchema = SchemaFactory.createForClass(AIModel);

// Create compound indexes for efficient queries
AIModelSchema.index({ modelType: 1, isActive: 1 });

