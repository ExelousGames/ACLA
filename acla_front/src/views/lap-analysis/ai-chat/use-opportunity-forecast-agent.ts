import { useEffect, useRef } from 'react';
import {
    getOpportunityForecast,
    OpportunityForecastOpportunity,
    OpportunityForecastResponse,
} from 'services/opportunityForecastService';

const MAX_BUFFER_ROWS = 80;
const FORECAST_INTERVAL_MS = 5000;
const DEFAULT_HORIZON_SECONDS = 10;
const DEFAULT_TOP_K = 3;

export interface OpportunityForecastAgentOptions {
    enabled: boolean;
    liveData?: Record<string, any> | null;
    sendObservation: (data: Record<string, unknown>) => boolean;
    onForecast: (summary: string, response: OpportunityForecastResponse) => void;
}

const formatOpportunity = (item: OpportunityForecastOpportunity): string => {
    const percent = Math.round(item.probability * 100);
    const section = item.circuit_section_name ? ` at ${item.circuit_section_name}` : '';
    return `${item.label_name} ${percent}%${section}`;
};

const buildSignature = (items: OpportunityForecastOpportunity[]): string =>
    items
        .map((item) => [
            item.label_id,
            Math.round(item.probability * 20),
            item.circuit_section_id || item.circuit_section_name || '',
        ].join(':'))
        .join('|');

export function useOpportunityForecastAgent({
    enabled,
    liveData,
    sendObservation,
    onForecast,
}: OpportunityForecastAgentOptions) {
    const bufferRef = useRef<Record<string, any>[]>([]);
    const requestInFlightRef = useRef(false);
    const lastSignatureRef = useRef('');

    useEffect(() => {
        if (!liveData || Object.keys(liveData).length === 0) {
            return;
        }
        bufferRef.current = [...bufferRef.current, liveData].slice(-MAX_BUFFER_ROWS);
    }, [liveData]);

    useEffect(() => {
        if (!enabled) {
            lastSignatureRef.current = '';
            return;
        }

        let cancelled = false;

        const requestForecast = async () => {
            if (requestInFlightRef.current || bufferRef.current.length === 0) {
                return;
            }

            requestInFlightRef.current = true;
            try {
                const response = await getOpportunityForecast({
                    telemetry_data: bufferRef.current,
                    horizon_seconds: DEFAULT_HORIZON_SECONDS,
                    top_k: DEFAULT_TOP_K,
                });
                if (cancelled) {
                    return;
                }
                const opportunities = response.opportunities ?? [];
                if (response.model_status === 'not_trained' || opportunities.length === 0) {
                    return;
                }

                const signature = buildSignature(opportunities);
                if (!signature || signature === lastSignatureRef.current) {
                    return;
                }

                const sent = sendObservation({
                    event: 'opportunity_forecast',
                    horizon_seconds: response.horizon_seconds,
                    opportunities: opportunities.map((item) => ({
                        label_id: item.label_id,
                        label_name: item.label_name,
                        parent_label: item.parent_label,
                        probability: item.probability,
                        circuit_section_id: item.circuit_section_id,
                        circuit_section_name: item.circuit_section_name,
                    })),
                });
                if (!sent) {
                    return;
                }

                lastSignatureRef.current = signature;
                onForecast(
                    `Opportunity forecast, next ${response.horizon_seconds}s: ${opportunities.map(formatOpportunity).join(' | ')}`,
                    response,
                );
            } catch (err) {
                console.warn('[opportunity-forecast-agent] forecast failed:', err);
            } finally {
                requestInFlightRef.current = false;
            }
        };

        void requestForecast();
        const timer = window.setInterval(() => {
            void requestForecast();
        }, FORECAST_INTERVAL_MS);

        return () => {
            cancelled = true;
            window.clearInterval(timer);
        };
    }, [enabled, onForecast, sendObservation]);
}
