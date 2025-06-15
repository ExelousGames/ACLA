import { PythonShell } from 'python-shell';
import path from 'path';

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
export async function PythonRunner(scriptName: string, options: PythonShellOptions, b: number): Promise<PythonResult> {
    return new Promise((resolve) => {
        PythonShell.run(scriptName, options);
    });
}