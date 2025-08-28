import { Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { AllMapsBasicInfoListDto, MapBasicInfo } from 'src/dto/map.dto';
import { RacingMap } from 'src/schemas/map.schema';
import { v4 as uuid } from 'uuid';

@Injectable()
export class RacingMapService {

    constructor(@InjectModel(RacingMap.name) private racingMap: Model<RacingMap>) {
    }

    async createNewMap(name: string): Promise<{ success: boolean; message: string; map?: RacingMap }> {
        try {
            // Check if map with this name already exists
            const existingMap = await this.racingMap.findOne({ name: name }).exec();

            if (existingMap) {
                return { success: false, message: 'Map with this name already exists' };
            }

            // Create a default empty image (1x1 transparent PNG)
            const defaultImageBuffer = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==', 'base64');

            // Create new map
            const newMap = new this.racingMap({
                name: name,
                ImageData: defaultImageBuffer,
                mimetype: 'image/png',
                points: []
            });

            const savedMap = await newMap.save();

            return {
                success: true,
                message: 'Map created successfully',
                map: savedMap
            };
        } catch (error) {
            return {
                success: false,
                message: `Failed to create map: ${error.message}`
            };
        }
    }

    async getRacingMap(name: string): Promise<RacingMap | null> {
        return this.racingMap.findOne({ name: name }).exec();
    }

    async retrieveAllMapBasicInfos(): Promise<AllMapsBasicInfoListDto | null> {

        try {
            let racingMap = new AllMapsBasicInfoListDto();
            const rawData = await this.racingMap.find().exec();
            racingMap.list = rawData.map((element) => {

                return { name: element.name };

            });

            return racingMap;
        }
        catch (error) {
            // Handle errors appropriately
            throw new Error(`Failed to process data: ${error.message}`);
        }

    }

    async uploadMapImage(mapName: string, file: any): Promise<{ success: boolean; message: string }> {
        try {

            // Validate the uploaded file
            if (!file) {
                return { success: false, message: 'No file provided' };
            }

            // Find the existing map by name
            const existingMap = await this.racingMap.findOne({ name: mapName }).exec();

            if (!existingMap) {
                return { success: false, message: 'Map not found' };
            }

            // Update the map with the new image data
            existingMap.ImageData = file.buffer;
            existingMap.mimetype = file.mimetype;
            await existingMap.save();

            return { success: true, message: 'Image uploaded successfully' };
        } catch (error) {
            return { success: false, message: `Failed to upload image: ${error.message}` };
        }
    }

    /**
     * Get the image data and mimetype for a specific map
     * @param mapName The name of the map
     * @returns The image data and mimetype, or null if not found
     */
    async getMapImage(mapName: string): Promise<{ imageData: string; mimetype: string } | null> {
        try {

            // Find the map by name
            const map = await this.racingMap.findOne({ name: mapName }).exec();

            // Check if map exists and has image data
            if (!map || !map.ImageData) {
                return null;
            }

            // Convert buffer to base64
            const base64 = map.ImageData.toString('base64');

            return {
                imageData: base64,
                mimetype: map.mimetype
            };
        } catch (error) {
            throw new Error(`Failed to retrieve image: ${error.message}`);
        }
    }

    /**
     * Update the racing map points data
     * @param mapName The name of the map
     * @param points The array of points to save
     * @returns Success response
     */
    async updateMapPoints(mapName: string, points: any[]): Promise<{ success: boolean; message: string }> {
        try {
            // Find the existing map by name
            const existingMap = await this.racingMap.findOne({ name: mapName }).exec();

            if (!existingMap) {
                return { success: false, message: 'Map not found' };
            }

            // Update the map with the new points data
            existingMap.points = points as any;
            await existingMap.save();

            return { success: true, message: 'Map data saved successfully' };
        } catch (error) {
            return { success: false, message: `Failed to save map data: ${error.message}` };
        }
    }
}
