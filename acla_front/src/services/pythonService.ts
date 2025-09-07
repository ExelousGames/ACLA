import { PythonShell } from 'python-shell';
import path from 'path';
import { IpcRendererEvent } from 'electron';

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
            /**
             * Run python script in main process
             * @param script 
             * @param options 
             * @returns 
             */
            runPythonScript: (script: string, options: PythonShellOptions) => { shellId: number };

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
             * @returns 
             */
            onPythonEnd: (callback: (shellId: number) => void) => void;

            /**
             * Send message to a script running in main process 
             * @param shellId 
             * @param message 
             * @returns 
             */
            sendMessageToPython: (shellId: number, message: string) => void;

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