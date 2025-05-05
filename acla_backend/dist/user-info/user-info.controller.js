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
exports.UserInfoController = void 0;
const common_1 = require("@nestjs/common");
const user_info_service_1 = require("./user-info.service");
const user_info_model_1 = require("./dto/user-info.model");
const create_user_dto_1 = require("./dto/create-user.dto");
const user_info_schema_1 = require("./schemas/user-info.schema");
let UserInfoController = class UserInfoController {
    userinfoService;
    constructor(userinfoService) {
        this.userinfoService = userinfoService;
    }
    getUser(id) {
        return this.userinfoService.getUser(id);
    }
    createUser(createUserInfoDto) {
        this.userinfoService.createUser(createUserInfoDto).then((dto) => {
            return dto;
        }).catch((error) => {
            console.log(error);
        });
        return new user_info_schema_1.UserInfo;
    }
    deleteUser(id) {
        this.userinfoService.deleteTask(id);
    }
};
exports.UserInfoController = UserInfoController;
__decorate([
    (0, common_1.Get)(':id'),
    __param(0, (0, common_1.Param)('id')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [Object]),
    __metadata("design:returntype", user_info_model_1.userInfoDto)
], UserInfoController.prototype, "getUser", null);
__decorate([
    (0, common_1.Post)(),
    __param(0, (0, common_1.Body)('infoDto')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [create_user_dto_1.CreateUserInfoDto]),
    __metadata("design:returntype", user_info_schema_1.UserInfo)
], UserInfoController.prototype, "createUser", null);
__decorate([
    (0, common_1.Delete)(':id'),
    __param(0, (0, common_1.Param)('id')),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [String]),
    __metadata("design:returntype", void 0)
], UserInfoController.prototype, "deleteUser", null);
exports.UserInfoController = UserInfoController = __decorate([
    (0, common_1.Controller)('userinfo'),
    __metadata("design:paramtypes", [user_info_service_1.UserInfoService])
], UserInfoController);
//# sourceMappingURL=user-info.controller.js.map