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