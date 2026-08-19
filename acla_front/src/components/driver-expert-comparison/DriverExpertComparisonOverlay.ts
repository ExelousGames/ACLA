import type { DesktopGame } from 'contexts/DesktopGameContext';
import type { DriverExpertComparisonData } from './DriverExpertComparisonGraph';

export interface DriverExpertComparisonSnapshot {
    title: string;
    comparison: DriverExpertComparisonData;
    game?: DesktopGame | null;
}
