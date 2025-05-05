"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __metadata = (this && this.__metadata) || function (k, v) {
    if (typeof Reflect === "object" && typeof Reflect.metadata === "function") return Reflect.metadata(k, v);
};
var __param = (this && this.__param) || function (paramIndex, decorator) {
    return function (target, key) { decorator(target, key, paramIndex); }
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.UserInfoService = void 0;
const common_1 = require("@nestjs/common");
const user_info_model_1 = require("./dto/user-info.model");
const uuid_1 = require("uuid");
const user_info_schema_1 = require("./schemas/user-info.schema");
const mongoose_1 = require("@nestjs/mongoose");
const mongoose_2 = require("mongoose");
let UserInfoService = class UserInfoService {
    userInfoModel;
    constructor(userInfoModel) {
        this.userInfoModel = userInfoModel;
    }
    getUser(id) {
        let info = new user_info_model_1.userInfoDto;
        return info;
    }
    async createUser(createUserInfoDto) {
        const newUserInfo = {
            id: (0, uuid_1.v4)(),
            username: createUserInfoDto.name
        };
        const createdInfo = new this.userInfoModel(newUserInfo);
        return createdInfo.save();
    }
    deleteTask(id) {
    }
};
exports.UserInfoService = UserInfoService;
exports.UserInfoService = UserInfoService = __decorate([
    (0, common_1.Injectable)(),
    __param(0, (0, mongoose_1.InjectModel)(user_info_schema_1.UserInfo.name)),
    __metadata("design:paramtypes", [mongoose_2.Model])
], UserInfoService);
//# sourceMappingURL=user-info.service.js.map