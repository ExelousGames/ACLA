"""Future overtake/defense opportunity forecasting."""

from app.ml.opportunity_forecaster.service import (
    OpportunityForecasterService,
    opportunity_forecaster,
)

__all__ = ["OpportunityForecasterService", "opportunity_forecaster"]
