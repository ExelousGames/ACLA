import { BadRequestException, Injectable, NotFoundException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model, Types } from 'mongoose';
import {
    CircuitMap,
    CircuitMapBinSample,
    CircuitMapCenterlineTag,
    CircuitMapCenterlineSegment,
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
    centerline_tags?: CircuitMapCenterlineTag[];
    centerline_segments?: CircuitMapCenterlineSegment[];
};

const CAPTURE_MODES: CircuitMapCaptureMode[] = ['left_boundary', 'middle_line', 'right_boundary', 'pit_lane'];

@Injectable()
export class CircuitMapService {
    constructor(
        @InjectModel(CircuitMap.name)
        private readonly circuitMapModel: Model<CircuitMap>,
    ) { }

    listCenterlineTags() {
        return { tags: ['corner', 'slow', 'fast', 'long straight', 'consecutive corners'] };
    }

    async list(game?: CircuitMapGame) {
        if (game !== undefined && game !== 'acc' && game !== 'iracing') {
            throw new BadRequestException('game must be acc or iracing');
        }
        const query = game !== undefined ? { game } : {};
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

    async remove(id: string): Promise<void> {
        this.assertObjectId(id);
        const result = await this.circuitMapModel.deleteOne({ _id: id }).exec();
        if (result.deletedCount === 0) {
            throw new NotFoundException('Circuit map not found');
        }
    }

    private normalizePayload(payload: CircuitMapPayload, isCreate: boolean) {
        const circuitName = payload.circuit_name?.trim();
        if (isCreate && !circuitName) {
            throw new BadRequestException('circuit_name is required');
        }

        const game = payload.game;
        if ((isCreate || game !== undefined) && game !== 'acc' && game !== 'iracing') {
            throw new BadRequestException('game must be acc or iracing');
        }
        const samples = this.normalizeSamples(payload.samples);
        const sampleCount = this.countSamples(samples);

        return {
            ...(game !== undefined ? { game } : {}),
            ...(circuitName ? { circuit_name: circuitName } : {}),
            source_track_key: payload.source_track_key || null,
            resolution: Number.isFinite(Number(payload.resolution)) ? Number(payload.resolution) : 1000,
            samples,
            ...(isCreate || payload.centerline_segments !== undefined || payload.centerline_tags !== undefined
                ? { centerline_segments: payload.centerline_segments !== undefined
                    ? this.normalizeCenterlineSegments(payload.centerline_segments)
                    : this.migrateCenterlineTags(payload.centerline_tags === undefined ? [] : payload.centerline_tags) }
                : {}),
            sample_count: sampleCount,
            updated_at: new Date().toISOString(),
        };
    }

    private normalizeCenterlineSegments(segments: CircuitMapCenterlineSegment[]): CircuitMapCenterlineSegment[] {
        if (!Array.isArray(segments)) {
            throw new BadRequestException('centerline_segments must be an array');
        }
        const ids = new Set<string>();
        return segments.map((tag) => {
            if (!tag || typeof tag.id !== 'string' || !tag.id.trim() || tag.id.length > 120
                || !Array.isArray(tag.tags) || tag.tags.length === 0
                || tag.tags.some((label) => typeof label !== 'string' || !label.trim() || label.trim().length > 120)
                || typeof tag.start_position !== 'number' || !Number.isFinite(tag.start_position)
                || typeof tag.end_position !== 'number' || !Number.isFinite(tag.end_position)
                || tag.start_position < 0 || tag.start_position > 1
                || tag.end_position < 0 || tag.end_position > 1
                || tag.start_position === tag.end_position) {
                throw new BadRequestException('Each centerline segment requires an id, one or more tags of 1–120 characters, and distinct start/end positions from 0 to 1');
            }
            const id = tag.id.trim();
            if (ids.has(id)) {
                throw new BadRequestException('Centerline segment ids must be unique');
            }
            ids.add(id);
            return { id, tags: Array.from(new Set(tag.tags.map((label) => label.trim()))), start_position: tag.start_position, end_position: tag.end_position };
        });
    }

    private migrateCenterlineTags(tags: CircuitMapCenterlineTag[]): CircuitMapCenterlineSegment[] {
        if (!Array.isArray(tags)) throw new BadRequestException('centerline_tags must be an array');
        const converted = this.normalizeCenterlineSegments(tags.map((tag) => ({
            id: tag?.id, tags: [tag?.label], start_position: tag?.start_position, end_position: tag?.end_position,
        })));
        const segments: CircuitMapCenterlineSegment[] = [];
        const isCorner = (label: string) => ['corner', 'slow corner', 'fast corner'].includes(label.toLowerCase());
        for (const tag of converted) {
            const existing = segments.find((segment) => segment.start_position === tag.start_position
                && segment.end_position === tag.end_position);
            if (existing) {
                // Keep the corner identity used by track guides and live coaching.
                if (tag.tags.some(isCorner) && !existing.tags.some(isCorner)) existing.id = tag.id;
                existing.tags = Array.from(new Set([...existing.tags, ...tag.tags]));
            } else {
                segments.push(tag);
            }
        }
        return segments;
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
            middle_line: [],
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
            centerline_segments: map.centerline_segments ?? this.migrateCenterlineTags(map.centerline_tags ?? []),
        };
    }

    private assertObjectId(id: string) {
        if (!Types.ObjectId.isValid(id)) {
            throw new BadRequestException('Invalid circuit map id');
        }
    }
}
