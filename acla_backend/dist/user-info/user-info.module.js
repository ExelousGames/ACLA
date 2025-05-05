"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.UserInfoModule = void 0;
const common_1 = require("@nestjs/common");
const user_info_controller_1 = require("./user-info.controller");
const user_info_service_1 = require("./user-info.service");
const mongoose_1 = require("@nestjs/mongoose");
const user_info_schema_1 = require("./schemas/user-info.schema");
let UserInfoModule = class UserInfoModule {
};
exports.UserInfoModule = UserInfoModule;
exports.UserInfoModule = UserInfoModule = __decorate([
    (0, common_1.Module)({
        imports: [mongoose_1.MongooseModule.forFeature([{ name: user_info_schema_1.UserInfo.name, schema: user_info_schema_1.UserInfoSchema }])],
        controllers: [user_info_controller_1.UserInfoController],
        providers: [user_info_service_1.UserInfoService]
    })
], UserInfoModule);
//# sourceMappingURL=user-info.module.js.map