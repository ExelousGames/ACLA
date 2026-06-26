import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';

export type CircuitMapGame = 'acc' | 'other';
export type CircuitMapCaptureMode = 'left_boundary' | 'right_boundary' | 'pit_lane';

export class CircuitMapBinSample {
    @Prop({ required: true })
    bin: number;

    @Prop({ required: true })
    normalized_position: number;

    @Prop({ required: true })
    x: number;

    @Prop({ required: true })
    y: number;

    @Prop({ required: true })
    z: number;

    @Prop({ required: true, default: 1 })
    sample_count: number;

    @Prop({ required: true })
    updated_at: string;

    @Prop({ required: false })
    locked?: boolean;
}

export class CircuitMapSamplesByMode {
    @Prop({ type: [Object], default: [] })
    left_boundary: CircuitMapBinSample[];

    @Prop({ type: [Object], default: [] })
    right_boundary: CircuitMapBinSample[];

    @Prop({ type: [Object], default: [] })
    pit_lane: CircuitMapBinSample[];
}

@Schema()
export class CircuitMap {
    @Prop({ type: String, required: true, enum: ['acc', 'other'], default: 'acc' })
    game: CircuitMapGame;

    @Prop({ required: true })
    circuit_name: string;

    @Prop({ type: String, required: false, default: null })
    source_track_key?: string | null;

    @Prop({ required: true, default: 1000 })
    resolution: number;

    @Prop({ type: Object, default: {} })
    samples: CircuitMapSamplesByMode;

    @Prop({ required: true, default: 0 })
    sample_count: number;

    @Prop({ required: true, default: () => new Date().toISOString() })
    updated_at: string;

    @Prop({ required: true, default: () => new Date().toISOString() })
    created_at: string;
}

export const CircuitMapSchema = SchemaFactory.createForClass(CircuitMap);
CircuitMapSchema.index({ game: 1, circuit_name: 1 }, { unique: false });
CircuitMapSchema.index({ game: 1, source_track_key: 1 }, { unique: false });
