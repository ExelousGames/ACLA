"""Optional local annotation API, separate from the frontend-facing service."""

from fastapi import FastAPI
from training.api.annotation import router

app = FastAPI(title="ACLA Local Annotation")
app.include_router(router)
