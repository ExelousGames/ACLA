import { PythonShell } from 'python-shell';
import path from 'path';

declare global {

    //The interface Window extension you're seeing in the React component is a TypeScript feature 
    // that allows you to safely extend the global window object with custom properties. 
    interface Window {
        electronAPI: {
            //preload.js runs in runtime. the renderer process has no Node.js or Electron module access. 
            runPythonScript: (scriptPath: string, options: PythonShellOptions) => Promise<string[]>;
            onPythonMessage: (callback: (message: string) => void) => void;
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