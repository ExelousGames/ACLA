import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { HydratedDocument } from 'mongoose';

export type UserSummaryAnalysisJobDocument = HydratedDocument<UserSummaryAnalysisJob>;

export type UserSummaryAnalysisJobStatus = 'queued' | 'running' | 'completed' | 'failed';

@Schema()
export class UserSummaryAnalysisJob {
    @Prop({ required: true, index: true })
    userId: string;

    @Prop({ required: true, enum: ['queued', 'running', 'completed', 'failed'], default: 'queued', index: true })
    status: UserSummaryAnalysisJobStatus;

    @Prop({ type: Object, default: {} })
    progress: Record<string, any>;

    @Prop({ type: Number, default: 10 })
    sessionLimit: number;

    @Prop({ type: Object, default: null })
    result: Record<string, any> | null;

    @Prop({ type: String, default: null })
    error: string | null;

    @Prop({ default: Date.now, index: true })
    createdAt: Date;

    @Prop({ default: Date.now })
    updatedAt: Date;

    @Prop({ type: Date, default: null })
    startedAt: Date | null;

    @Prop({ type: Date, default: null })
    completedAt: Date | null;
}

export const UserSummaryAnalysisJobSchema = SchemaFactory.createForClass(UserSummaryAnalysisJob);

UserSummaryAnalysisJobSchema.index(
    { userId: 1, status: 1 },
    {
        unique: true,
        partialFilterExpression: { status: { $in: ['queued', 'running'] } },
        name: 'one_active_user_summary_analysis_per_user',
    },
);
