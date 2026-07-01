import { Injectable, Logger, OnModuleInit } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { v4 as uuid } from 'uuid';
import { UserInfo } from '../../schemas/user-info.schema';
import { Role } from '../../schemas/role.schema';
import { Permission } from '../../schemas/permission.schema';
import { PasswordService } from '../utils/password.service';

@Injectable()
export class SeedService implements OnModuleInit {
    private readonly logger = new Logger(SeedService.name);

    constructor(
        @InjectModel(UserInfo.name) private userInfoModel: Model<UserInfo>,
        @InjectModel(Role.name) private roleModel: Model<Role>,
        @InjectModel(Permission.name) private permissionModel: Model<Permission>,
        private passwordService: PasswordService,
    ) {}

    async onModuleInit() {
        await this.seedServiceReaderRole();
        await this.seedAdminAccount();
        await this.seedAiServiceAccount();
    }

    private async seedServiceReaderRole() {
        const existingRole = await this.roleModel.findOne({ name: 'Service Reader' }).exec();
        if (existingRole) {
            this.logger.log('Service Reader role already exists, skipping');
            return;
        }

        // Look up all read permissions
        const readPermissions = await this.permissionModel.find({
            action: 'read',
        }).exec();

        if (readPermissions.length === 0) {
            this.logger.warn('No read permissions found in database. Ensure init-mongo.js has run first.');
            return;
        }

        const permissionIds = readPermissions.map(p => p._id);

        const role = new this.roleModel({
            id: 'role_service_reader',
            name: 'Service Reader',
            description: 'Read-only access to all resources (for AI service)',
            permissions: permissionIds,
            createdAt: new Date(),
            isActive: true,
        });

        await role.save();
        this.logger.log(`Service Reader role created with ${readPermissions.length} read permissions`);
    }

    private async seedAdminAccount() {
        const email = process.env.ADMIN_USERNAME;
        const password = process.env.ADMIN_PASSWORD;

        if (!email || !password) {
            this.logger.warn('ADMIN_USERNAME or ADMIN_PASSWORD not set, skipping admin account seed');
            return;
        }

        const existing = await this.userInfoModel.findOne({ email }).exec();
        if (existing) {
            this.logger.log(`Admin account (${email}) already exists, skipping`);
            return;
        }

        const superAdminRole = await this.roleModel.findOne({ name: 'Super Admin' }).exec();
        if (!superAdminRole) {
            this.logger.warn('Super Admin role not found. Ensure init-mongo.js has run first.');
            return;
        }

        const hashedPassword = await this.passwordService.hashPassword(password);

        const adminUser = new this.userInfoModel({
            id: uuid(),
            email,
            password: hashedPassword,
            roles: [superAdminRole._id],
            permissions: [],
            isActive: true,
            createdAt: new Date(),
            lastLogin: new Date(),
        });

        await adminUser.save();
        this.logger.log(`Admin account (${email}) created with Super Admin role`);
    }

    private async seedAiServiceAccount() {
        const email = process.env.AI_SERVICE_USERNAME;
        const password = process.env.AI_SERVICE_PASSWORD;

        if (!email || !password) {
            this.logger.warn('AI_SERVICE_USERNAME or AI_SERVICE_PASSWORD not set, skipping AI service account seed');
            return;
        }

        const existing = await this.userInfoModel.findOne({ email }).exec();
        if (existing) {
            this.logger.log(`AI service account (${email}) already exists, skipping`);
            return;
        }

        const serviceReaderRole = await this.roleModel.findOne({ name: 'Service Reader' }).exec();
        if (!serviceReaderRole) {
            this.logger.warn('Service Reader role not found. This should not happen — seed may have failed.');
            return;
        }

        const hashedPassword = await this.passwordService.hashPassword(password);

        const serviceUser = new this.userInfoModel({
            id: uuid(),
            email,
            password: hashedPassword,
            roles: [serviceReaderRole._id],
            permissions: [],
            isActive: true,
            createdAt: new Date(),
            lastLogin: new Date(),
        });

        await serviceUser.save();
        this.logger.log(`AI service account (${email}) created with Service Reader role`);
    }
}
