import { Prop, Schema, SchemaFactory } from "@nestjs/mongoose";

//Each schema maps to a MongoDB collection and defines the shape of the documents within that collection
//The @Schema() decorator marks a class as a schema definition. It maps our Cat class to a MongoDB collection of the same name, but with an additional “s” at the end - so the final mongo collection name will be cats
@Schema()
export class RacingMap {

    @Prop({ required: true, unique: true })
    name: string;

    @Prop({ required: true })
    mimetype: string;

    @Prop({ required: true, type: Buffer })
    data: Buffer; // Binary image data

    @Prop()
    points: [{
        position: number[],
        type: number,
        index: number, //type and index are used together. some points are index sensitive
        description?: string,
        info?: string,
        variables?: [{ key: string, value: string }] //any word match {key} in description or info will be replaced with the value
    }]
}

export const RacingMapSchema = SchemaFactory.createForClass(RacingMap);