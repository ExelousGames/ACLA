import { Prop, Schema, SchemaFactory } from "@nestjs/mongoose";
import { Types } from "mongoose";

@Schema()
export class Role {
    @Prop({ required: true })
    id: string;

    @Prop({ required: true, unique: true })
    name: string;

    @Prop({ required: true })
    description: string;

    @Prop({ type: [{ type: Types.ObjectId, ref: 'Permission' }] })
    permissions: Types.ObjectId[];

    @Prop({ default: Date.now })
    createdAt: Date;

    @Prop({ default: true })
    isActive: boolean;
}

export const RoleSchema = SchemaFactory.createForClass(Role);
