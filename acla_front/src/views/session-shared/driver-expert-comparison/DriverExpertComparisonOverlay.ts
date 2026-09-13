import type { DesktopGame } from 'contexts/DesktopGameContext';
import type { TtsPack } from 'components/tts';
import type { DriverExpertComparisonLabelRange } from './DriverExpertComparisonRoadSigns';
import type {
    DriverExpertComparisonData,
    DriverExpertComparisonLabelGroup,
} from './DriverExpertComparisonGraph';

export interface DriverExpertComparisonSnapshot {
    title: string;
    comparison: DriverExpertComparisonData;
    labelGroups?: readonly DriverExpertComparisonLabelGroup[];
    labelRanges?: readonly DriverExpertComparisonLabelRange[];
    game?: DesktopGame | null;
    voice?: TtsPack;
}
