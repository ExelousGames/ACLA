export interface TaskStartFunction {
    (signal: AbortSignal): void | Promise<void>;
}
