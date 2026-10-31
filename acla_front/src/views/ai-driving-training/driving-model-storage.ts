import { DrivingTrainingDocument, readTrainingDocument } from './driving-policy';

export interface DrivingModelFiles {
    current(): Promise<string | null>;
    choose(mode: 'new' | 'open'): Promise<string | null>;
    select(filePath: string): Promise<void>;
    read(filePath: string): Promise<string>;
    write(filePath: string, json: string): Promise<void>;
}

declare global {
    interface Window { drivingModelFiles?: DrivingModelFiles }
}

const pendingSaves = new Map<string, Promise<boolean>>();

export function trackDrivingModelSave(filePath: string, task: Promise<boolean>): void {
    pendingSaves.set(filePath, task);
    const finished = () => { if (pendingSaves.get(filePath) === task) pendingSaves.delete(filePath); };
    void task.then(finished, finished);
}

export function drivingModelFiles(): DrivingModelFiles {
    if (!window.drivingModelFiles) throw new Error('Model files require the updated desktop app. Restart the app after updating.');
    return window.drivingModelFiles;
}

export function discardBrowserModels(): void {
    // Browser checkpoints are retired. Delete them without loading or migrating weights.
    try {
        const obsolete = Array.from({ length: localStorage.length }, (_, i) => localStorage.key(i)!)
            .filter((key) => /^acla\.self-driving\.policy\.v\d+$/.test(key)
                || key === 'acla.self-driving.model' || key === 'acla.self-driving.aggression');
        obsolete.forEach((key) => localStorage.removeItem(key));
    } catch { /* Model files work even if browser storage is unavailable. */ }
}

export async function loadDrivingModel(filePath: string): Promise<DrivingTrainingDocument> {
    // A page remount must wait for the previous page's GPU checkpoint and disk write.
    if (await pendingSaves.get(filePath) === false) throw new Error('The last model save failed.');
    return readTrainingDocument(JSON.parse(await drivingModelFiles().read(filePath)));
}

export async function saveDrivingModel(filePath: string, document: DrivingTrainingDocument): Promise<void> {
    await drivingModelFiles().write(filePath, JSON.stringify(document));
}
