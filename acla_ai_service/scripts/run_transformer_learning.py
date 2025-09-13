#!/usr/bin/env python3
import asyncio
from pathlib import Path
import sys

# Add the parent directory to the Python path to import modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from app.services.full_dataset_ml_service import Full_dataset_TelemetryMLService

async def main():
    try:
        ml_service = Full_dataset_TelemetryMLService()
        results = await ml_service.transformerLearning('brands_hatch')
        print(results)
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
