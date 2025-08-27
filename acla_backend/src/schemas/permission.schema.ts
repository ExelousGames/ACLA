import { Prop, Schema, SchemaFactory } from "@nestjs/mongoose";

export enum PermissionAction {
    CREATE = 'create',
    READ = 'read',
    UPDATE = 'update',
    DELETE = 'delete',
    MANAGE = 'manage'
}

// Define resources in the systems
export enum PermissionResource {
    USER = 'user',
    MENU = 'menu',
    RACING_SESSION = 'racing_session',
    RACING_MAP = 'racing_map',
    ALL = 'all'
}
//
@Schema()
export class Permission {
    @Prop({ required: true })
    id: string;

    @Prop({ required: true })
    name: string;

    @Prop({ required: true })
    description: string;

    @Prop({ required: true, enum: PermissionAction })
    action: PermissionAction;

    @Prop({ required: true, enum: PermissionResource })
    resource: PermissionResource;

    @Prop({ default: Date.now })
    createdAt: Date;
}

export const PermissionSchema = SchemaFactory.createForClass(Permission);
