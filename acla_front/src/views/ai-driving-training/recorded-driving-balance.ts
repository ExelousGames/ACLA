import { emptyRecordedSamplingWeights, RECORDED_SAMPLING_CATEGORIES, UNLABELED_RECORDED_SAMPLING_CATEGORY,
    validateRecordedBatchLabelCount, type DrivingModelConfig, type RecordedSamplingCategory, type RecordedSamplingWeights } from './driving-model-config';
import type { DrivingObservation } from './driving-observation';
import { isConsecutiveRecordedObservation } from './recorded-driving-replay';

const normalize = (label: string | undefined) => label?.trim().toLowerCase() ?? '';
const categoriesByLabel = new Map(RECORDED_SAMPLING_CATEGORIES.map((category) => [normalize(category), category] as const));
export interface RecordedDrivingCoverage { eligible: RecordedSamplingWeights }

/** Count aligned origins before decoding video and verify those same origins during replay. */
export class RecordedDrivingBalance {
    private samples: { label: string; category: RecordedSamplingCategory }[] = [];
    private eligible = emptyRecordedSamplingWeights();
    private previous?: { observation: DrivingObservation; label: string };
    private replayIndex = 0;
    weights = emptyRecordedSamplingWeights();

    constructor(private config: DrivingModelConfig) {}

    count(observation: DrivingObservation | undefined, clipLabel?: string, startsRange?: boolean) {
        const previous = this.previous;
        this.previous = observation && { observation, label: normalize(clipLabel) };
        if (startsRange || !previous || !observation
            || !isConsecutiveRecordedObservation(previous.observation, observation, this.config)) return;

        const { label } = previous;
        const category = categoriesByLabel.get(label) ?? UNLABELED_RECORDED_SAMPLING_CATEGORY;
        this.eligible[category]++;
        this.samples.push({ label, category });
    }

    prepare(noSamplesError?: string): RecordedDrivingCoverage {
        this.previous = undefined;
        const configured = this.config.recordedSamplingWeights;
        this.weights = { ...configured };
        for (const category of RECORDED_SAMPLING_CATEGORIES) {
            if (!this.eligible[category]) this.weights[category] = 0;
        }
        if (!RECORDED_SAMPLING_CATEGORIES.some((category) => this.eligible[category] > 0)) {
            throw new Error(noSamplesError || 'No consecutive aligned samples were available for replay.');
        }
        const eligibleLabelCount = RECORDED_SAMPLING_CATEGORIES.filter((category) => this.weights[category] > 0).length;
        if (!eligibleLabelCount) {
            throw new Error('The recording has no examples with a positive sampling weight. Adjust the sampling weights or select different annotations.');
        }
        validateRecordedBatchLabelCount(this.config.recordedBatchSize, eligibleLabelCount);
        return { eligible: { ...this.eligible } };
    }

    select(clipLabel?: string): RecordedSamplingCategory {
        const sample = this.samples[this.replayIndex];
        if (!sample || sample.label !== normalize(clipLabel)) {
            throw new Error('The aligned samples changed after counting the recording.');
        }
        this.replayIndex++;
        return sample.category;
    }

    get complete() { return this.replayIndex === this.samples.length; }
}
