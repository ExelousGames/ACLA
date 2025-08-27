import { Controller, Get, Request, Body, Post, UseInterceptors, UploadedFile } from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { AllMapsBasicInfoListDto } from 'src/dto/map.dto';
import { RacingMapService } from './racing-map.service';
import { RacingMap } from 'src/schemas/map.schema';
import { Auth } from '../../common/decorators/auth.decorator';
import { PermissionAction, PermissionResource } from '../../schemas/permission.schema';

@Controller('racingmap')
export class RacingMapController {

    constructor(private racingMapService: RacingMapService) { }

    @Auth({ permissions: [{ action: PermissionAction.READ, resource: PermissionResource.RACING_MAP }] })
    @Get('map/infolists')
    async getAllMapsBasicInfos(@Body() body: any): Promise<AllMapsBasicInfoListDto | null> {
        return this.racingMapService.retrieveAllMapBasicInfos();
    }

    @Auth({ permissions: [{ action: PermissionAction.CREATE, resource: PermissionResource.RACING_MAP }] })
    @Post('map/create')
    async createNewMap(@Body() body: { name: string }): Promise<{ success: boolean; message: string; map?: RacingMap }> {
        return this.racingMapService.createNewMap(body.name);
    }

    @Auth({ permissions: [{ action: PermissionAction.READ, resource: PermissionResource.RACING_MAP }] })
    @Post('map/infolists')
    async getMap(@Body() body: any): Promise<RacingMap | null> {
        return this.racingMapService.getRacingMap(body.name);
    }

    @Auth({ permissions: [{ action: PermissionAction.UPDATE, resource: PermissionResource.RACING_MAP }] })
    @Post('map/upload-image')
    @UseInterceptors(FileInterceptor('image'))
    async uploadMapImage(
        @UploadedFile() file: any,
        @Body('mapName') mapName: string
    ): Promise<{ success: boolean; message: string }> {
        return this.racingMapService.uploadMapImage(mapName, file);
    }

    @Auth({ permissions: [{ action: PermissionAction.READ, resource: PermissionResource.RACING_MAP }] })
    @Post('map/image')
    async getMapImage(@Body() body: { name: string }): Promise<{ imageData: string; mimetype: string } | null> {
        return this.racingMapService.getMapImage(body.name);
    }
}
