"""
Dataset management endpoints
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import pandas as pd
import io
from datetime import datetime
from app.models import DatasetInfo

router = APIRouter(prefix="/datasets", tags=["datasets"])

# In-memory storage for datasets
datasets_cache = {}

@router.post("/upload", response_model=DatasetInfo)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload and process a dataset"""
    try:
        # Read the file content
        content = await file.read()
        
        # Parse based on file extension
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith('.json'):
            df = pd.read_json(io.StringIO(content.decode('utf-8')))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Generate dataset ID
        dataset_id = f"{file.filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store dataset
        datasets_cache[dataset_id] = {
            "dataframe": df,
            "metadata": {
                "filename": file.filename,
                "upload_time": datetime.utcnow().isoformat(),
                "size": len(df)
            }
        }
        
        return DatasetInfo(
            id=dataset_id,
            name=file.filename,
            size=len(df),
            columns=list(df.columns),
            upload_time=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.get("/", response_model=List[DatasetInfo])
async def list_datasets():
    """List all available datasets"""
    result = []
    for dataset_id, data in datasets_cache.items():
        metadata = data["metadata"]
        df = data["dataframe"]
        
        result.append(DatasetInfo(
            id=dataset_id,
            name=metadata["filename"],
            size=len(df),
            columns=list(df.columns),
            upload_time=metadata["upload_time"]
        ))
    
    return result
