"""
Streamlined Training-Optimized Cache Service for Large Datasets

Clean, efficient cache service designed specifically for ML model training:
- Simple JSON storage format for maximum compatibility
- Direct streaming processing for training pipelines
- Minimal memory footprint with direct dictionary operations
- Fast access optimized for feature extraction
- Chunk-agnostic: treats each chunk as opaque part of larger dataset
- No assumptions about chunk contents or structure
"""

import json
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator
from datetime import datetime, timedelta


class TrainingOptimizedCache:
    """
    Large Dataset Cache for ML Training Pipelines
    
    Designed for processing massive datasets (GB to TB scale):
    - Multi-part Parquet storage for memory efficiency
    - Streaming processing to avoid memory overflow
    - Aggressive chunking for optimal resource usage
    - Manifest-based file management for large datasets
    """
    
    def __init__(self, cache_directory: str = "telemetry_data_cache"):
        """Initialize training-optimized cache"""
        self.cache_dir = Path(cache_directory)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Parquet-only storage for consistency
        self.parquet_dir = self.cache_dir / "parquet_storage"
        self.parquet_dir.mkdir(exist_ok=True)
        
        # Simple metadata tracking
        self.metadata_db = self.cache_dir / "cache_metadata.db"
        self._init_metadata_db()
        
        # Clean cache on initialization to ensure fresh start
        print(f"[INFO] Cleaning cache directory on initialization...")
        self.clear_cache()
        
        # Large dataset optimization settings
        self.compression = 'snappy'  # Fast compression for large files
        self.row_group_size = 50000   # Conservative row groups for memory efficiency
        
        print(f"[INFO] Training-optimized cache initialized: {self.cache_dir}")
        print(f"[INFO] Parquet-only storage with snappy compression")
        print(f"[INFO] Optimized for ML training pipelines")
    
    def _init_metadata_db(self):
        """Initialize simple metadata database"""
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cached_datasets (
                    cache_key TEXT PRIMARY KEY,
                    track_name TEXT,
                    car_name TEXT,
                    cached_at TIMESTAMP NOT NULL,
                    file_path TEXT NOT NULL,
                    data_size_mb REAL,
                    chunk_count INTEGER,
                    record_count INTEGER,
                    last_accessed TIMESTAMP
                )
            """)
    
    async def cache_chunks_streaming(self, cache_key: str, chunks_iterator: Iterator[Dict[str, Any]]) -> bool:
        """Cache chunk data to Parquet format (training optimized) - Cache each chunk as a separate Parquet file"""
        try:
            chunk_count = 0
            total_records = 0
            chunk_files = []  # Keep track of chunk files
            
            print(f"[INFO] Starting chunk-based parquet caching for {cache_key}")
            
            # Process each chunk - completely agnostic to contents
            async for chunk_data in chunks_iterator:
                if not chunk_data:
                    print(f"[WARNING] Skipping empty chunk {chunk_count}")
                    continue
                
                # Create chunk file - no assumptions about content
                chunk_file = self.parquet_dir / f"{cache_key}_chunk_{chunk_count}.parquet"
                
                # Store chunk data directly as DataFrame - agnostic approach
                try:
                    chunk_df = pd.DataFrame([chunk_data])  # Treat entire chunk as single record
                    chunk_df.to_parquet(chunk_file, compression=self.compression, index=False)
                    chunk_files.append(chunk_file)
                    
                    chunk_count += 1
                    total_records += 1  # Each chunk is one record
                except Exception as e:
                    print(f"[WARNING] Failed to save chunk {chunk_count}: {e}")
                    continue

            if not chunk_files:
                print(f"[WARNING] No chunks to cache for {cache_key}")
                return False
            
            # Create or append to manifest file to track all chunk files
            manifest_file = self.parquet_dir / f"{cache_key}_manifest.txt"
            
            # Check if manifest already exists (for accumulating multiple cache operations)
            existing_chunks = []
            if manifest_file.exists():
                with open(manifest_file, 'r') as f:
                    existing_chunks = [line.strip() for line in f.readlines() if line.strip()]
            
            # Append new chunks to existing ones
            all_chunks = existing_chunks + [chunk_file.name for chunk_file in chunk_files]
            
            # Write complete manifest
            with open(manifest_file, 'w') as f:
                for chunk_name in all_chunks:
                    f.write(f"{chunk_name}\n")
            
            # Calculate total size for all chunks (existing + new)
            all_chunk_paths = [self.parquet_dir / chunk_name for chunk_name in all_chunks]
            total_size_mb = sum(f.stat().st_size for f in all_chunk_paths if f.exists()) / (1024 * 1024)
            
            # Calculate total counts for all chunks
            total_chunk_count = len(all_chunks)
            
            # Calculate total records for new chunks only (existing records already counted)
            existing_records = 0
            if existing_chunks:
                # Get existing record count from database
                existing_info = self._get_cache_metadata(cache_key, max_age_hours=24)
                if existing_info:
                    existing_records = existing_info.get('record_count', 0)
            
            total_all_records = existing_records + total_records
            
            print(f"[INFO] Chunk-based storage: {total_size_mb:.1f}MB across {total_chunk_count} total chunk files ({len(chunk_files)} new)")
            
            # Update metadata - use the manifest file as the primary file path
            self._update_cache_metadata(
                cache_key, str(manifest_file), total_size_mb, total_chunk_count, total_all_records
            )
            
            print(f"[INFO] Cached {chunk_count} new chunks, {total_records} new records. Total: {total_chunk_count} chunks, {total_all_records} records ({total_size_mb:.1f}MB)")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to cache chunks for {cache_key}: {e}")
            # Clean up failed files
            try:
                manifest_file = self.parquet_dir / f"{cache_key}_manifest.txt"
                if manifest_file.exists():
                    manifest_file.unlink()
                
                # Clean up any chunk files that were created
                chunk_pattern = f"{cache_key}_chunk_*.parquet"
                for file_path in self.parquet_dir.glob(chunk_pattern):
                    file_path.unlink()
            except Exception:
                pass
            return False

    def _convert_df_to_chunks(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert DataFrame back to chunk format - content agnostic"""
        # Simply convert entire DataFrame to single chunk - no assumptions about structure
        return [{
            "chunkId": "chunk_0",
            "data": df.to_dict('records')
        }]
    
    def get_cached_data_chunks(self, cache_key: str) -> Iterator[pd.DataFrame]:
        """
        Direct iterator that yields DataFrame chunks from cached data - no fallbacks or validation.
        Pure streaming approach that assumes cache exists and is valid.
        
        Args:
            cache_key: Cache key to get data for
            
        Yields:
            pd.DataFrame: Individual chunks of cached data (loaded only when accessed)
        """
        # Direct manifest file access - no metadata checks
        manifest_file = self.parquet_dir / f"{cache_key}_manifest.txt"
        
        # Read chunk file list directly from manifest
        try:
            if not manifest_file.exists():
                print(f"[ERROR] Manifest file not found for cache key {cache_key}")
                return
        except Exception as e:
            raise RuntimeError(f"Could not access manifest file for {cache_key}: {e}")
        
        with open(manifest_file, 'r') as f:
            chunk_files = [line.strip() for line in f.readlines() if line.strip()]
        
        # Pure generator - direct chunk iteration with no validation
        for chunk_filename in chunk_files:
            chunk_path = self.parquet_dir / chunk_filename
            
            # Load and yield chunk data directly
            chunk_df = pd.read_parquet(chunk_path)
            yield chunk_df
            
            # Immediate memory cleanup
            del chunk_df

    
    def _get_cache_metadata(self, cache_key: str, max_age_hours: int) -> Optional[Dict[str, Any]]:
        """Get cache metadata from database"""
        with sqlite3.connect(self.metadata_db) as conn:
            cursor = conn.execute("""
                SELECT * FROM cached_datasets 
                WHERE cache_key = ? AND cached_at > ?
            """, (cache_key, datetime.now() - timedelta(hours=max_age_hours)))
            
            row = cursor.fetchone()
            if row:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, row))
        
        return None
    
    def _update_cache_metadata(self, cache_key: str, file_path: str, data_size_mb: float, 
                             chunk_count: int, record_count: int):
        """Update cache metadata in database"""
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cached_datasets 
                (cache_key, track_name, car_name, cached_at, file_path,
                 data_size_mb, chunk_count, record_count, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cache_key, None, None, datetime.now(), file_path,
                data_size_mb, chunk_count, record_count, datetime.now()
            ))
    
    def _remove_cache_entry(self, cache_key: str):
        """Remove cache entry and all associated chunk files"""
        with sqlite3.connect(self.metadata_db) as conn:
            cursor = conn.execute(
                "SELECT file_path FROM cached_datasets WHERE cache_key = ?", 
                (cache_key,)
            )
            row = cursor.fetchone()
            
            if row and row[0]:
                try:
                    manifest_file = Path(row[0])
                    
                    # If manifest exists, remove all chunk files it references
                    if manifest_file.exists():
                        with open(manifest_file, 'r') as f:
                            chunk_files = [line.strip() for line in f.readlines() if line.strip()]
                        
                        # Remove each chunk file
                        for chunk_filename in chunk_files:
                            chunk_path = manifest_file.parent / chunk_filename
                            chunk_path.unlink(missing_ok=True)
                        
                        # Remove manifest file
                        manifest_file.unlink(missing_ok=True)
                    
                    # Also clean up any orphaned chunk files for this cache_key
                    chunk_pattern = f"{cache_key}_chunk_*.parquet"
                    for file_path in self.parquet_dir.glob(chunk_pattern):
                        file_path.unlink(missing_ok=True)
                        
                except Exception as e:
                    print(f"[WARNING] Error cleaning up cache files: {e}")
            
            conn.execute("DELETE FROM cached_datasets WHERE cache_key = ?", (cache_key,))
    

    
    def clear_cache(self, cache_key: Optional[str] = None):
        """Clear cache for specific cache key or all cache entries"""
        if cache_key:
            self._remove_cache_entry(cache_key)
        else:
            with sqlite3.connect(self.metadata_db) as conn:
                conn.execute("DELETE FROM cached_datasets")
            # Clear parquet directory with better error handling
            import shutil
            import os
            if self.parquet_dir.exists():
                try:
                    shutil.rmtree(self.parquet_dir)
                    self.parquet_dir.mkdir(exist_ok=True)
                except OSError as e:
                    print(f"[WARNING] Could not remove directory {self.parquet_dir}: {e}")
                    print("[INFO] Attempting to clear individual files...")
                    try:
                        # Try to remove individual files first
                        for file_path in self.parquet_dir.rglob("*"):
                            if file_path.is_file():
                                try:
                                    file_path.unlink()
                                except OSError as file_error:
                                    print(f"[WARNING] Could not remove file {file_path}: {file_error}")
                        
                        # Try to remove empty directories
                        for dir_path in sorted(self.parquet_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                            if dir_path.is_dir() and dir_path != self.parquet_dir:
                                try:
                                    dir_path.rmdir()
                                except OSError:
                                    pass  # Directory not empty, skip
                        
                        print("[INFO] Individual file cleanup completed")
                    except Exception as cleanup_error:
                        print(f"[ERROR] File cleanup failed: {cleanup_error}")
                        print("[INFO] Cache directory may contain residual files")
                    
                    # Ensure the main directory exists
                    if not self.parquet_dir.exists():
                        self.parquet_dir.mkdir(exist_ok=True)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        with sqlite3.connect(self.metadata_db) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) as entries, 
                       COALESCE(SUM(data_size_mb), 0) as total_size
                FROM cached_datasets
            """)
            row = cursor.fetchone()
            entries, total_size = row if row else (0, 0)
        
        return {
            "cache_entries": entries,
            "total_size_mb": total_size,
            "storage_format": "Parquet with snappy compression",
            "cache_directory": str(self.cache_dir)
        }


# Create shared service instance - clean and simple
training_cache = TrainingOptimizedCache()

def get_shared_data_cache():
    """Get the shared training-optimized cache instance"""
    return training_cache
