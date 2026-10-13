const TRACK_LABELS = ['track', 'road', 'asphalt', 'tarmac'];

export function isTrackLabel(label: string | undefined) {
    return TRACK_LABELS.includes(label?.trim().toLowerCase() ?? '');
}

export function isCarInteriorLabel(label: string | undefined) {
    return label?.trim().toLowerCase().replace(/\s+/g, ' ') === 'car interior';
}
