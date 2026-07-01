import { Module } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { UserInfo, UserInfoSchema } from '../../schemas/user-info.schema';
import { Role, RoleSchema } from '../../schemas/role.schema';
import { Permission, PermissionSchema } from '../../schemas/permission.schema';
import { PasswordService } from '../utils/password.service';
import { SeedService } from './seed.service';

@Module({
    imports: [
        MongooseModule.forFeature([
            { name: UserInfo.name, schema: UserInfoSchema },
            { name: Role.name, schema: RoleSchema },
            { name: Permission.name, schema: PermissionSchema },
        ]),
    ],
    providers: [SeedService, PasswordService],
    exports: [SeedService],
})
export class SeedModule {}
