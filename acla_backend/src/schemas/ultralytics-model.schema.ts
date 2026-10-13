import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { ObjectId } from 'mongodb';
import { Schema as MongooseSchema } from 'mongoose';

export const ULTRALYTICS_TASKS = [
  'detect',
  'segment',
  'classify',
  'pose',
  'obb',
] as const;
export type UltralyticsTask = (typeof ULTRALYTICS_TASKS)[number];

@Schema({ timestamps: true, collection: 'ultralytics_models' })
export class UltralyticsModel {
  @Prop({ required: true, trim: true })
  name: string;

  @Prop({ required: true, enum: ['ultralytics'], default: 'ultralytics' })
  framework: string;

  @Prop({ required: true, enum: ['labelme'], default: 'labelme' })
  annotationFormat: string;

  @Prop({ type: String, required: true, enum: ULTRALYTICS_TASKS })
  task: UltralyticsTask;

  // Array position is the class ID used by the trained weights.
  @Prop({ type: [String], required: true })
  classNames: string[];

  @Prop({ type: MongooseSchema.Types.Mixed, default: {} })
  metadata: Record<string, unknown>;

  @Prop({ type: MongooseSchema.Types.ObjectId, required: true })
  modelFileId: ObjectId;

  @Prop({ required: true })
  filename: string;

  @Prop({ required: true, min: 1 })
  sizeBytes: number;

  @Prop({ required: true })
  sha256: string;
}

export const UltralyticsModelSchema =
  SchemaFactory.createForClass(UltralyticsModel);
UltralyticsModelSchema.index({ name: 1, createdAt: -1 });
