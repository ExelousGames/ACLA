import type { DrivingAction } from './driving-observation';
import type { PhysicalController } from './driving-control';

export interface ControllerOutput {
    deviceId?: number;
}
export type ControllerAction = DrivingAction;
export interface VirtualController {
    connect(options?: ControllerOutput): Promise<void>;
    apply(action: ControllerAction): Promise<void>;
    disconnect(): Promise<void>;
}
export interface AccControlPreset {
    name: string;
    path: string;
    deviceId: number;
}
export interface VirtualControllerApi extends VirtualController {
    readPhysicalControls?(): Promise<PhysicalController[]>;
    installAccPreset(deviceId: number): Promise<AccControlPreset>;
}
declare global {
    interface Window { virtualController?: VirtualControllerApi }
}
