import { Module } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { CircuitMap, CircuitMapSchema } from 'src/schemas/circuit-map.schema';
import { CircuitMapController } from './circuit-map.controller';
import { CircuitMapService } from './circuit-map.service';

@Module({
    imports: [
        MongooseModule.forFeature([{ name: CircuitMap.name, schema: CircuitMapSchema }]),
    ],
    controllers: [CircuitMapController],
    providers: [CircuitMapService],
})
export class CircuitMapModule { }
