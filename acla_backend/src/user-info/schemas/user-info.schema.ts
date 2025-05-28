import { Prop, Schema, SchemaFactory } from "@nestjs/mongoose";

//Each schema maps to a MongoDB collection and defines the shape of the documents within that collection
//The @Schema() decorator marks a class as a schema definition. It maps our Cat class to a MongoDB collection of the same name, but with an additional “s” at the end - so the final mongo collection name will be cats
@Schema()
export class UserInfo {

    @Prop()
    id: string;

    @Prop({ required: true })
    username: string;

    @Prop({ required: true })
    password: string;
}

export const UserInfoSchema = SchemaFactory.createForClass(UserInfo);