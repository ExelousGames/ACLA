import { BadRequestException, Injectable, NotFoundException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model, Types } from 'mongoose';
import {
    CircuitMap,
    CircuitMapBinSample,
    CircuitMapCaptureMode,
    CircuitMapGame,
    CircuitMapSamplesByMode,
} from 'src/schemas/circuit-map.schema';

type CircuitMapPayload = {
    game?: CircuitMapGame;
    circuit_name?: string;
    source_track_key?: string | null;
    resolution?: number;
    samples?: Partial<Record<CircuitMapCaptureMode, CircuitMapBinSample[]>>;
};

const CAPTURE_MODES: CircuitMapCaptureMode[] = ['left_boundary', 'right_boundary', 'pit_lane'];

@Injectable()
export class CircuitMapService {
    constructor(
        @InjectModel(CircuitMap.name)
        private readonly circuitMapModel: Model<CircuitMap>,
    ) { }

    async list(game?: CircuitMapGame) {
        const query = game ? { game } : {};
        const maps = await this.circuitMapModel
            .find(query)
            .sort({ updated_at: -1, circuit_name: 1 })
            .lean()
            .exec();

        return {
            list: maps.map((map: any) => this.toSummaryDto(map)),
        };
    }

    async get(id: string) {
        this.assertObjectId(id);
        const map = await this.circuitMapModel.findById(id).lean().exec();
        if (!map) {
            throw new NotFoundException('Circuit map not found');
        }
        return this.toDto(map);
    }

    async create(payload: CircuitMapPayload) {
        const data = this.normalizePayload(payload, true);
        const created = await this.circuitMapModel.create({
            ...data,
            created_at: new Date().toISOString(),
        });
        return this.toDto(created.toObject());
    }

    async update(id: string, payload: CircuitMapPayload) {
        this.assertObjectId(id);
        const data = this.normalizePayload(payload, false);
        const updated = await this.circuitMapModel
            .findByIdAndUpdate(id, data, { new: true })
            .lean()
            .exec();

        if (!updated) {
            throw new NotFoundException('Circuit map not found');
        }
        return this.toDto(updated);
    }

    private normalizePayload(payload: CircuitMapPayload, requireName: boolean) {
        const circuitName = payload.circuit_name?.trim();
        if (requireName && !circuitName) {
            throw new BadRequestException('circuit_name is required');
        }

        const game = payload.game === 'other' ? 'other' : 'acc';
        const samples = this.normalizeSamples(payload.samples);
        const sampleCount = this.countSamples(samples);

        return {
            game,
            ...(circuitName ? { circuit_name: circuitName } : {}),
            source_track_key: payload.source_track_key || null,
            resolution: Number.isFinite(Number(payload.resolution)) ? Number(payload.resolution) : 1000,
            samples,
            sample_count: sampleCount,
            updated_at: new Date().toISOString(),
        };
    }

    private normalizeSamples(samples?: CircuitMapPayload['samples']): CircuitMapSamplesByMode {
        return CAPTURE_MODES.reduce((normalized, mode) => ({
            ...normalized,
            [mode]: Array.isArray(samples?.[mode])
                ? samples[mode]!.map((sample) => ({
                    bin: Number(sample.bin),
                    normalized_position: Number(sample.normalized_position),
                    x: Number(sample.x),
                    y: Number(sample.y),
                    z: Number(sample.z),
                    sample_count: Number(sample.sample_count || 1),
                    updated_at: sample.updated_at || new Date().toISOString(),
                    locked: sample.locked,
                }))
                : [],
        }), {
            left_boundary: [],
            right_boundary: [],
            pit_lane: [],
        } as CircuitMapSamplesByMode);
    }

    private countSamples(samples: CircuitMapSamplesByMode): number {
        return CAPTURE_MODES.reduce((sum, mode) => sum + (samples[mode]?.length || 0), 0);
    }

    private toSummaryDto(map: any) {
        return {
            id: String(map._id),
            game: map.game,
            circuit_name: map.circuit_name,
            source_track_key: map.source_track_key ?? null,
            updated_at: map.updated_at ?? null,
            sample_count: Number(map.sample_count ?? this.countSamples(map.samples || {})),
        };
    }

    private toDto(map: any) {
        return {
            ...this.toSummaryDto(map),
            resolution: Number(map.resolution ?? 1000),
            samples: this.normalizeSamples(map.samples),
        };
    }

    private assertObjectId(id: string) {
        if (!Types.ObjectId.isValid(id)) {
            throw new BadRequestException('Invalid circuit map id');
        }
    }
}
