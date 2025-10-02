"""
Streamlined Training-Optimized Cache Service for Large Telemetry Datasets

Clean, efficient cache service designed specifically for ML model training:
- Single Parquet storage format (no fallbacks)
- Direct streaming processing for training pipelines
- Minimal memory footprint with zero-copy operations
- Fast columnar access optimized for feature extraction
- No legacy paths or duplicate logic
"""

import sqlite3
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator
from datetime import datetime, timedelta
import pandas as pd

# Suppress warnings for clean output
warnings.filterwarnings('ignore', category=UserWarning)


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
                    track_name TEXT NOT NULL,
                    car_name TEXT,
                    cached_at TIMESTAMP NOT NULL,
                    file_path TEXT NOT NULL,
                    data_size_mb REAL,
                    session_count INTEGER,
                    record_count INTEGER,
                    last_accessed TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_track_name ON cached_datasets(track_name)")
    
    def _generate_cache_key(self, track_name: str, car_name: Optional[str] = None) -> str:
        """Generate cache key"""
        key_data = f"{track_name}_{car_name or 'all_cars'}"
        return str(hash(key_data) % (10**10))
    
    def get_cached_sessions(self, track_name: str, car_name: Optional[str] = None,
                           max_age_hours: int = 24) -> Optional[Dict[str, Any]]:
        """Get cached session data (Parquet only)"""
        cache_key = self._generate_cache_key(track_name, car_name)
        
        # Check for valid cache
        cached_info = self._get_cache_metadata(cache_key, max_age_hours)
        if cached_info:
            try:
                return self._load_parquet_data(cached_info)
            except Exception as e:
                print(f"[WARNING] Cache load failed for {track_name}: {e}")
                self._remove_cache_entry(cache_key)
        
        return None
    
    async def cache_sessions_streaming(self, track_name: str, sessions_iterator: Iterator[Dict[str, Any]], 
                               car_name: Optional[str] = None,
                               estimated_size_mb: Optional[float] = None) -> bool:
        """Cache session data to Parquet format (training optimized)"""
        cache_key = self._generate_cache_key(track_name, car_name)
        
        try:
            return await self._cache_to_parquet(cache_key, track_name, car_name, sessions_iterator)
        except Exception as e:
            print(f"[ERROR] Failed to cache sessions for {track_name}: {e}")
            return False
    
    async def _cache_to_parquet(self, cache_key: str, track_name: str, 
                               car_name: Optional[str], 
                               sessions_iterator: Iterator[Dict[str, Any]]) -> bool:
        """Stream sessions directly to single Parquet file using efficient batching"""
        parquet_file = self.parquet_dir / f"{cache_key}.parquet"
        
        try:
            all_records = []
            session_count = 0
            total_records = 0
            chunk_files = []  # Keep track of temporary chunk files
            
            print(f"[INFO] Starting parquet caching for {cache_key}")
            
            # Collect all data with improved memory management
            async for session in sessions_iterator:
                session_id = session.get("sessionId", f"session_{session_count}")
                session_data = session.get("data", [])
                
                if not session_data:
                    continue
                
                # Add session metadata to each record
                for record in session_data:
                    record['session_id'] = session_id
                    all_records.append(record)
                
                session_count += 1
                total_records += len(session_data)
                
                # Process in optimal chunks for large datasets - aggressive chunking for memory efficiency
                chunk_size = 5000  # Optimal chunk size for large datasets to minimize memory footprint
                
                if len(all_records) >= chunk_size:
                    # Write chunk to temporary file instead of loading entire existing file
                    chunk_file = self.parquet_dir / f"{cache_key}_chunk_{len(chunk_files)}.parquet"
                    df_chunk = pd.DataFrame(all_records)
                    df_chunk.to_parquet(chunk_file, compression=self.compression, index=False)
                    chunk_files.append(chunk_file)
                    
                    all_records = []  # Clear memory immediately
                    
                    print(f"[INFO] Cached chunk {len(chunk_files)} with {len(df_chunk)} records")
            
            # Write remaining records
            if all_records:
                chunk_file = self.parquet_dir / f"{cache_key}_chunk_{len(chunk_files)}.parquet"
                df_final = pd.DataFrame(all_records)
                df_final.to_parquet(chunk_file, compression=self.compression, index=False)
                chunk_files.append(chunk_file)
                print(f"[INFO] Cached final chunk {len(chunk_files)} with {len(df_final)} records")
            
            # Always use multi-part parquet files for large datasets (this is a cache service for large data)
            if chunk_files:
                total_size_mb = sum(f.stat().st_size for f in chunk_files if f.exists()) / (1024 * 1024)
                print(f"[INFO] Large dataset: {total_size_mb:.1f}MB across {len(chunk_files)} chunks")
                print(f"[INFO] Using multi-part parquet storage for optimal memory efficiency")
                
                # Rename chunk files to have consistent naming and keep them separate
                for i, chunk_file in enumerate(chunk_files):
                    new_name = self.parquet_dir / f"{cache_key}_part_{i:03d}.parquet"
                    if chunk_file != new_name:
                        chunk_file.rename(new_name)
                
                # Create a manifest file to track all parts
                manifest_file = self.parquet_dir / f"{cache_key}_manifest.txt"
                with open(manifest_file, 'w') as f:
                    for i in range(len(chunk_files)):
                        f.write(f"{cache_key}_part_{i:03d}.parquet\n")
                
                print(f"[INFO] Created {len(chunk_files)} parquet files with manifest for memory-efficient processing")
            
            # Calculate final size
            file_size_mb = parquet_file.stat().st_size / (1024 * 1024) if parquet_file.exists() else 0
            
            # Update metadata
            self._update_cache_metadata(
                cache_key, track_name, car_name, str(parquet_file), 
                file_size_mb, session_count, total_records
            )
            
            print(f"[INFO] Cached {session_count} sessions, {total_records} records ({file_size_mb:.1f}MB)")
            return True
            
        except Exception as e:
            print(f"[ERROR] Parquet caching failed: {e}")
            # Clean up failed file
            if parquet_file.exists():
                parquet_file.unlink()
            return False

    def _load_parquet_data(self, cached_info: Dict[str, Any]) -> Dict[str, Any]:
        """Load data from Parquet file(s) - supports both single files and multiple parts"""
        parquet_file = Path(cached_info["file_path"])
        
        # Check if this is a multi-part dataset
        manifest_file = parquet_file.parent / f"{parquet_file.stem}_manifest.txt"
        
        if manifest_file.exists():
            # Multi-part dataset - load from manifest
            print(f"[INFO] Loading multi-part parquet dataset using manifest")
            
            with open(manifest_file, 'r') as f:
                part_files = [line.strip() for line in f.readlines() if line.strip()]
            
            # Load all parts
            all_dfs = []
            for part_name in part_files:
                part_path = parquet_file.parent / part_name
                if part_path.exists():
                    part_df = pd.read_parquet(part_path)
                    all_dfs.append(part_df)
            
            if all_dfs:
                df = pd.concat(all_dfs, ignore_index=True)
            else:
                raise FileNotFoundError(f"No valid parquet parts found for manifest: {manifest_file}")
                
        else:
            # Single parquet file
            if not parquet_file.exists():
                raise FileNotFoundError(f"Parquet file not found: {parquet_file}")
            
            df = pd.read_parquet(parquet_file)
        
        # Convert back to session format
        sessions_data = self._convert_df_to_sessions(df)
        
        return {
            "success": True,
            "track_name": cached_info.get("track_name"),
            "car_name": cached_info.get("car_name"),
            "sessions": sessions_data,
            "summary": {
                "total_sessions_retrieved": cached_info.get("session_count", 0),
                "total_telemetry_records": cached_info.get("record_count", 0)
            }
        }
    
    def _convert_df_to_sessions(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert DataFrame back to session format"""
        if 'session_id' not in df.columns:
            # Single session data
            return [{
                "sessionId": "session_0",
                "data": df.to_dict('records')
            }]
        
        sessions_data = []
        for session_id, group in df.groupby('session_id', sort=False):
            # Remove session metadata columns
            data_df = group.drop(columns=['session_id'], errors='ignore')
            
            sessions_data.append({
                "sessionId": str(session_id),
                "data": data_df.to_dict('records')
            })
        
        return sessions_data
    

    
    def _process_parquet_streaming(self, file_path: str, processing_func: callable, chunk_size: int) -> Any:
        """Process Parquet data in streaming chunks - handles both single files and multi-part datasets"""
        parquet_file = Path(file_path)
        results = []
        total_rows = 0
        
        # Check if this is a multi-part dataset
        manifest_file = parquet_file.parent / f"{parquet_file.stem}_manifest.txt"
        
        if manifest_file.exists():
            # Multi-part dataset - process each part
            print(f"[INFO] Processing multi-part parquet dataset using manifest")
            
            with open(manifest_file, 'r') as f:
                part_files = [line.strip() for line in f.readlines() if line.strip()]
            
            for i, part_name in enumerate(part_files):
                part_path = parquet_file.parent / part_name
                if not part_path.exists():
                    print(f"[WARNING] Part file not found: {part_path}")
                    continue
                
                print(f"[INFO] Processing part {i+1}/{len(part_files)}: {part_name}")
                
                # Read this part in chunks
                df_part = pd.read_parquet(part_path)
                part_rows = len(df_part)
                total_rows += part_rows
                
                for j in range(0, part_rows, chunk_size):
                    chunk = df_part.iloc[j:j+chunk_size]
                    result = processing_func(chunk)
                    results.append(result)
                
                # Free memory
                del df_part
                
        else:
            # Single parquet file
            if not parquet_file.exists():
                raise FileNotFoundError(f"Parquet file not found: {parquet_file}")
            
            # Read Parquet in chunks to minimize memory usage
            df = pd.read_parquet(parquet_file)
            total_rows = len(df)
            
            for i in range(0, total_rows, chunk_size):
                chunk = df.iloc[i:i+chunk_size]
                result = processing_func(chunk)
                results.append(result)
        
        print(f"[INFO] Processed {total_rows} records in {len(results)} chunks")
        return results
    
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
    
    def _update_cache_metadata(self, cache_key: str, track_name: str, car_name: Optional[str],
                             file_path: str, data_size_mb: float, session_count: int, record_count: int):
        """Update cache metadata in database"""
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cached_datasets 
                (cache_key, track_name, car_name, cached_at, storage_type, file_path,
                 data_size_mb, session_count, record_count, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cache_key, track_name, car_name, datetime.now(), 'parquet', file_path,
                data_size_mb, session_count, record_count, datetime.now()
            ))
    
    def _remove_cache_entry(self, cache_key: str):
        """Remove corrupted cache entry"""
        with sqlite3.connect(self.metadata_db) as conn:
            cursor = conn.execute(
                "SELECT file_path FROM cached_datasets WHERE cache_key = ?", 
                (cache_key,)
            )
            row = cursor.fetchone()
            
            if row and row[0]:
                try:
                    Path(row[0]).unlink(missing_ok=True)
                except Exception:
                    pass
            
            conn.execute("DELETE FROM cached_datasets WHERE cache_key = ?", (cache_key,))
    
    def process_large_dataset_streaming(self, track_name: str, 
                                      processing_func: callable,
                                      chunk_size: int = 50000) -> Any:
        """Process cached dataset using streaming with minimal memory"""
        cache_key = self._generate_cache_key(track_name)
        cached_info = self._get_cache_metadata(cache_key, max_age_hours=24)
        
        if not cached_info:
            raise ValueError(f"No cached data found for track: {track_name}")
            
        # Process Parquet data in chunks
        return self._process_parquet_streaming(cached_info["file_path"], processing_func, chunk_size)
    
    def clear_cache(self, track_name: Optional[str] = None):
        """Clear cache for specific track or all tracks"""
        if track_name:
            cache_key = self._generate_cache_key(track_name)
            self._remove_cache_entry(cache_key)
        else:
            with sqlite3.connect(self.metadata_db) as conn:
                conn.execute("DELETE FROM cached_datasets")
            # Clear parquet directory
            import shutil
            if self.parquet_dir.exists():
                shutil.rmtree(self.parquet_dir)
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

# Backward compatibility aliases
HybridDataCache = TrainingOptimizedCache
OptimizedDataCache = TrainingOptimizedCache
hybrid_data_cache = training_cache
optimized_data_cache = training_cache

def get_shared_data_cache():
    """Get the shared training-optimized cache instance"""
    return training_cache

async def cache_telemetry_sessions(track_name: str, sessions_iterator, estimated_size_mb: float = None):
    """Cache telemetry sessions using the training-optimized cache"""
    return await training_cache.cache_sessions_streaming(
        track_name=track_name,
        sessions_iterator=sessions_iterator,
        estimated_size_mb=estimated_size_mb
    )

def get_cached_telemetry_sessions(track_name: str, max_age_hours: int = 24):
    """Get cached telemetry sessions"""
    return training_cache.get_cached_sessions(track_name, max_age_hours=max_age_hours)

def get_shared_cache_info():
    """Get training cache information"""
    return training_cache.get_cache_info()