import { IpcRendererEvent } from 'electron';

// Function with additional property
export interface CallbackFunction {
    (event: IpcRendererEvent, returnedShellId: number, message: any): void;
    __listener?: any
};

export interface PythonStartDetails {
    script: string;
    args?: any[];
    keepAlive: boolean;
    pythonPath?: string;
    startedAt: number;
}

export type PythonEndReason = 'close' | 'error' | 'terminated' | 'unknown';

export interface PythonEndDetails extends PythonStartDetails {
    reason: PythonEndReason;
    exitCode?: number | null;
    signal?: string | null;
    error?: string;
    stopRequestedBy?: string | null;
    finishedAt: number;
    durationMs: number;
    lastMessageAt?: number | null;
    messageCount?: number;
}

declare global {

    //The interface Window extension you're seeing in the React component is a TypeScript feature 
    // that allows you to safely extend the global window object with custom properties.
    //preload.js runs in runtime. the renderer process has no Node.js or Electron module access. 
    interface Window {
        electronAPI: {
            /**
             * Run python script in main process
             * @param script 
             * @param options 
             * @returns 
             */
            runPythonScript: (script: string, options: PythonShellOptions) => Promise<{ shellId: number; metadata?: PythonStartDetails }>;
            stopPythonScript: (shellId: number, initiator?: string) => Promise<{ success: boolean; error?: string }>;
            writeTempFile: (options: { content: string; directory?: string; prefix?: string; extension?: string }) => Promise<{ success: boolean; path?: string; error?: string; skipped?: boolean }>;
            deleteTempFile: (filePath: string) => Promise<{ success: boolean; error?: string; skipped?: boolean }>;

            /**
             * 
             * @param callback 
             * @returns the function of removing this listener
             */
            //onPythonMessage: (callback: ExtendedCallbackFunction) => void;
            onPythonMessage: (callback: (shellId: number, message: string) => void) => () => {};
            OnPythonMessageOnce: (callback: (shellId: number, message: string) => void) => void;

            /**
             * Notify when a python process transitions into running state
             */
            onPythonStart: (callback: (shellId: number, details: PythonStartDetails) => void) => () => void;

            /**
             * called when a script running in main process is terminated
             * @param callback function used for handling termination of a script 
             * @returns function to remove listener
             */
            onPythonEnd: (callback: (shellId: number, details?: PythonEndDetails) => void) => () => void;

            /**
             * Send message to a script running in main process 
             * @param shellId 
             * @param message 
             * @returns 
             */
            sendMessageToPython: (shellId: number, message: string) => Promise<{ success: boolean; error?: string }>;

            /**
             * Start local speech recognition (offline)
             * @returns Promise<{success: boolean, recordingId?: string}>
             */
            startSpeechRecognition: () => Promise<{ success: boolean, recordingId?: string, error?: string }>;

            /**
             * Stop local speech recognition
             * @returns Promise<{success: boolean, transcript?: string}>
             */
            stopSpeechRecognition: () => Promise<{ success: boolean, transcript?: string, error?: string }>;

            /**
             * Check if speech recognition is available
             * @returns Promise<boolean>
             */
            isSpeechRecognitionAvailable: () => Promise<boolean>;

            /**
             * Listen for speech recognition status updates
             * @param callback Function to handle status updates
             * @returns Function to remove listener
             */
            onSpeechRecognitionStatus: (callback: (status: any) => void) => () => void;

            /**
             * Listen for speech recognition completion
             * @param callback Function to handle completion
             * @returns Function to remove listener  
             */
            onSpeechRecognitionComplete: (callback: (result: { success: boolean, transcript?: string, error?: string }) => void) => () => void;
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