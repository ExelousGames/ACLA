export interface ScreenCaptureSource { id: string; name: string }

declare global {
    interface Window {
        screenCapture?: {
            listSources(): Promise<ScreenCaptureSource[]>;
            selectSource(id: string): Promise<void>;
        };
    }
}

export async function captureGameScreen(sourceId: string): Promise<MediaStream> {
    const capture = window.screenCapture;
    if (!capture) throw new Error('Track Vision is available only in the Electron desktop app.');
    if (!navigator.mediaDevices?.getDisplayMedia) {
        throw new Error('Screen capture is unavailable in the desktop app.');
    }
    if (!sourceId) throw new Error('Choose a game window or screen first.');
    await capture.selectSource(sourceId);
    return navigator.mediaDevices.getDisplayMedia({
        audio: false,
        video: { frameRate: { ideal: 15, max: 15 }, width: { ideal: 1280 }, height: { ideal: 720 } },
    });
}
