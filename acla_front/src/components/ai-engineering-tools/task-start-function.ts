export interface TaskStartFunction {
    (signal: AbortSignal): unknown | Promise<unknown>;
}
