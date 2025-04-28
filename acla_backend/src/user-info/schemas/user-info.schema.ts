import { Prop, Schema, SchemaFactory } from "@nestjs/mongoose";

@Schema()
export class UserInfo {

    @Prop()
    id: string;

    @Prop()
    username: string;

}

export const UserInfoSchema = SchemaFactory.createForClass(UserInfo);