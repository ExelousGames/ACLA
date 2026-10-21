import type { DrivingAction } from './driving-observation';

export interface ControllerOutput {
    backend: 'vjoy' | 'xbox';
    deviceId?: number;
}
export interface VirtualController {
    connect(options?: ControllerOutput): Promise<void>;
    apply(action: DrivingAction): Promise<void>;
    disconnect(): Promise<void>;
}
declare global {
    interface Window { virtualController?: VirtualController }
}
