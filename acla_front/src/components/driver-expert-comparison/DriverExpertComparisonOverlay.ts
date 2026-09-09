import type { DesktopGame } from 'contexts/DesktopGameContext';
import type {
    DriverExpertComparisonData,
    DriverExpertComparisonLabelGroup,
} from './DriverExpertComparisonGraph';

export interface DriverExpertComparisonSnapshot {
    title: string;
    comparison: DriverExpertComparisonData;
    labelGroups?: readonly DriverExpertComparisonLabelGroup[];
    game?: DesktopGame | null;
}
