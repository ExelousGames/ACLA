import { Module } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { GridFSService } from './gridfs.service';

@Module({
    imports: [MongooseModule.forFeature([])], // We don't need specific schemas for GridFS
    providers: [GridFSService],
    exports: [GridFSService],
})
export class GridFSModule { }
