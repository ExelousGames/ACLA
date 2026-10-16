"""Optional local training API, separate from the frontend-facing service."""

from fastapi import FastAPI
from training.api.annotation import router
from training.api.ultralytics import router as ultralytics_router

app = FastAPI(title="ACLA Local Training")
app.include_router(router)
app.include_router(ultralytics_router)
