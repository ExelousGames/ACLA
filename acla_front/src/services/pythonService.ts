import { PythonShell } from 'python-shell';
import path from 'path';
import { IpcRendererEvent } from 'electron';
import type { DesktopGame } from 'contexts/DesktopGameContext';
import type {
    RecordedFileReadEvent,
    RecordingStartResult,
    RecordingStopResult,
    RecordingViewUpdate,
} from 'views/live-session/live-session-types';

// Function with additional property
export interface CallbackFunction {
    (event: IpcRendererEvent, returnedShellId: number, message: any): void;
    __listener?: any
};



declare global {

    //The interface Window extension you're seeing in the React component is a TypeScript feature 
    // that allows you to safely extend the global window object with custom properties.
    //preload.js runs in runtime. the renderer process has no Node.js or Electron module access. 
    interface Window {
        electronAPI: {
            detectDesktopGame: () => Promise<{
                supported: boolean;
                detectedGame: DesktopGame | null;
            }>;
            startRecordingSession: (config: { game: DesktopGame }) => Promise<RecordingStartResult>;
            stopRecordingSession: () => Promise<RecordingStopResult>;
            onRecordingViewUpdate: (callback: (update: RecordingViewUpdate) => void) => () => void;
            onRecordingSessionEnded: (callback: (result: RecordingStopResult) => void) => () => void;
            startRecordedFileRead: (request: {
                filePath: string;
                game: DesktopGame;
                purpose: 'validate' | 'consume';
            }) => Promise<{ readId: string }>;
            cancelRecordedFileRead: (readId: string) => Promise<void>;
            onRecordedFileReadEvent: (callback: (event: RecordedFileReadEvent) => void | Promise<void>) => () => void;

            /**
             * Run python script in main process
             * @param script 
             * @param options 
             * @returns 
             */
            runPythonScript: (script: string, options: PythonShellOptions) => Promise<{ shellId: number }>;
            stopPythonScript: (shellId: number) => Promise<{ success: boolean; error?: string }>;
            writeTempFile: (options: { content: string; directory?: string; prefix?: string; extension?: string }) => Promise<{ success: boolean; path?: string; error?: string; skipped?: boolean }>;
            deleteTempFile: (filePath: string) => Promise<{ success: boolean; error?: string; skipped?: boolean }>;
            validateTelemetryFile: (filePath: string) => Promise<{
                exists: boolean;
                readable: boolean;
                hasData: boolean;
                size: number;
                error?: string;
            }>;

            /**
             * 
             * @param callback 
             * @returns the function of removing this listener
             */
            //onPythonMessage: (callback: ExtendedCallbackFunction) => void;
            onPythonMessage: (callback: (shellId: number, message: string) => void) => () => {};
            OnPythonMessageOnce: (callback: (shellId: number, message: string) => void) => void;

            /**
             * called when a script running in main process is terminated
             * @param callback function used for handling termination of a script 
             * @returns function to remove listener
             */
            onPythonEnd: (
                listenerIdOrCallback: string | ((shellId: number, listenerId?: string) => void),
                callback?: (shellId: number, listenerId?: string) => void
            ) => () => void;

            /**
             * Send message to a script running in main process 
             * @param shellId 
             * @param message 
             * @returns 
             */
            sendMessageToPython: (shellId: number, message: string) => Promise<{ success: boolean; error?: string }>;

        };
    }
}

export interface PythonResult {
    success: boolean;
    result?: number;
    error?: string;
}

export interface PythonShellOptions {
    mode?: 'text' | 'json' | 'binary';
    /**
     * The path where to locate the "python" executable. Default: "python3" ("python" for Windows)
     */
    pythonPath?: string;
    pythonOptions?: string[];
    /**
     * The default path where to look for scripts. Default is the current working directory.
     */
    scriptPath?: string;
    /**
     *  Array of arguments to pass to the script
     */
    args?: any[];
}
