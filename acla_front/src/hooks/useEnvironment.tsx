import { Environment } from '../utils/environment';
import { useEnvironment } from '../contexts/EnvironmentContext';

export const useEnvironmentDetector = (): {
    isElectron: boolean;
    isWeb: boolean;
    environment: Environment;
} => {
    const environment = useEnvironment();
    return {
        isElectron: environment === 'electron',
        isWeb: environment === 'web',
        environment,
    };
};