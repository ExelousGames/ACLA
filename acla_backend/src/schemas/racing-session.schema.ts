import { Prop, Schema, SchemaFactory } from "@nestjs/mongoose";
import { ObjectId, Types } from "mongoose";

//Each schema maps to a MongoDB collection and defines the shape of the documents within that collection
//The @Schema() decorator marks a class as a schema definition. It maps our Cat class to a MongoDB collection of the same name, but with an additional “s” at the end - so the final mongo collection name will be cats
@Schema()
export class RacingSession {

    @Prop({ required: true })
    session_name: string;

    @Prop({ required: true })
    map: string;

    @Prop({ required: true })
    car_name: string;

    @Prop({ type: String, ref: 'UserInfo', required: true })
    user_id: string;

    @Prop()
    points: [{
        id: number,
        position_x: number,
        position_y: number,
        description: string,
        info: string,
        variables: [{ key: string, value: string }] //any word match {key} in description or info will be replaced with the value
    }]

    @Prop({ type: Array })
    data: any[];
}

export const RacingSessionSchema = SchemaFactory.createForClass(RacingSession);
// Creating a compound index
RacingSessionSchema.index({ session_name: 1, map: 1, car_name: 1, user_id: 1 }, { unique: true });