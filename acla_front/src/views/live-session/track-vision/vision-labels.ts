const TRACK_LABELS = ['track', 'road', 'asphalt', 'tarmac'];

export function isTrackLabel(label: string | undefined) {
    return TRACK_LABELS.includes(label?.trim().toLowerCase() ?? '');
}
