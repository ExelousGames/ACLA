import {
    AiToolComponentErrorConstructor,
    VisualizationComponentError,
} from 'contexts/AiToolComponentError';

export const runVisualizationBooleanCallback = (
    componentName: string,
    ErrorType: AiToolComponentErrorConstructor<VisualizationComponentError>,
    fallbackMessage: string,
    callback: (() => boolean) | undefined,
): true => {
    if (!callback) {
        throw new ErrorType(componentName, fallbackMessage);
    }

    let succeeded: boolean;
    try {
        succeeded = callback();
    } catch (error) {
        if (
            error instanceof ErrorType
            && error.componentName === componentName
        ) {
            throw error;
        }
        throw new ErrorType(
            componentName,
            error instanceof Error && error.message ? error.message : fallbackMessage,
            { cause: error },
        );
    }

    if (!succeeded) {
        throw new ErrorType(componentName, fallbackMessage);
    }
    return true;
};
