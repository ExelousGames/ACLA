"""
Hybrid Data Cache Service for Large Telemetry Datasets

This service provides efficient handling of large telemetry datasets using:
- Dask for out-of-core processing
- HDF5 for efficient disk storage with compression
- Memory cache for frequently accessed data
- Streaming processing to avoid memory overflow
"""

import os
import json
import gzip
import hashlib
import sqlite3
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Tuple, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Dask imports for large dataset processing - REQUIRED
try:
    import dask
    # Check pandas version compatibility first
    import pandas as pd_version_check
    pandas_version = pd_version_check.__version__
    
    import dask.dataframe as dd
    from dask.distributed import LocalCluster, Client
    DASK_AVAILABLE = True
    print(f"[INFO] Dask successfully imported with pandas {pandas_version}")
except ImportError as e:
    raise ImportError(f"[CRITICAL ERROR] Dask is required for telemetry processing but not available. "
                     f"Install with: pip install dask[distributed]. Error: {str(e)}")
except (TypeError, AttributeError) as e:
    try:
        import pandas as pd_err_check
        pandas_ver = pd_err_check.__version__
        import dask as dask_err_check  
        dask_ver = dask_err_check.__version__ if hasattr(dask_err_check, '__version__') else 'unknown'
    except:
        pandas_ver = 'unknown'
        dask_ver = 'unknown'
    
    raise ImportError(f"[CRITICAL ERROR] Dask version compatibility issue detected. "
                     f"Current versions - Pandas: {pandas_ver}, Dask: {dask_ver}. "
                     f"Update requirements.txt with compatible versions: "
                     f"dask[dataframe]==2024.2.1 and pandas==2.1.4. "
                     f"Then rebuild Docker container. Error: {str(e)}")

# h5py import - REQUIRED for HDF5 storage
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError as e:
    raise ImportError(f"[CRITICAL ERROR] h5py is required for HDF5 storage but not available. "
                     f"Install with: pip install h5py. Error: {str(e)}")

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)


class HybridDataCache:
    """
    Hybrid cache for large telemetry datasets with Dask integration
    
    Features:
    - Memory cache for frequently used small datasets
    - HDF5 disk storage with compression for large datasets
    - Dask processing for out-of-core operations
    - Streaming data processing to avoid memory overflow
    """
    
    def __init__(self, 
                 cache_directory: str = "telemetry_data_cache",
                 max_memory_datasets: int = 3,
                 enable_dask: bool = True,
                 dask_memory_limit: str = "2GB"):
        """
        Initialize hybrid data cache
        
        Args:
            cache_directory: Directory for cache storage
            max_memory_datasets: Max datasets to keep in memory
            enable_dask: Enable Dask for large dataset processing
            dask_memory_limit: Memory limit for Dask workers
        """
        self.cache_dir = Path(cache_directory)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Memory cache for small/frequent datasets
        self.memory_cache = {}
        self.max_memory_datasets = max_memory_datasets
        self.access_order = []  # LRU tracking
        
        # HDF5 storage paths
        self.hdf5_dir = self.cache_dir / "hdf5_storage"
        self.hdf5_dir.mkdir(exist_ok=True)
        
        # SQLite metadata database
        self.metadata_db = self.cache_dir / "cache_metadata.db"
        self._init_metadata_db()
        
        # Dask setup - REQUIRED
        if not DASK_AVAILABLE:
            raise RuntimeError("[CRITICAL ERROR] Dask is required but not available during initialization")
        if not HDF5_AVAILABLE:
            raise RuntimeError("[CRITICAL ERROR] h5py is required but not available during initialization")
            
        self.enable_dask = enable_dask
        self.dask_client = None
        self.dask_memory_limit = dask_memory_limit
        
        if self.enable_dask:
            self._setup_dask()
        
        print(f"[INFO] Hybrid data cache initialized at {self.cache_dir}")
        print(f"[INFO] Dask enabled: {self.enable_dask}")
        print(f"[INFO] Required dependencies verified: Dask={DASK_AVAILABLE}, h5py={HDF5_AVAILABLE}")
    
    def _init_metadata_db(self):
        """Initialize SQLite metadata database"""
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cached_datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT UNIQUE NOT NULL,
                    track_name TEXT NOT NULL,
                    car_name TEXT,
                    cached_at TIMESTAMP NOT NULL,
                    storage_type TEXT NOT NULL,  -- 'memory', 'hdf5', 'parquet'
                    file_path TEXT,
                    data_size_mb REAL,
                    session_count INTEGER,
                    record_count INTEGER,
                    compression_ratio REAL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP,
                    metadata_json TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_track_name ON cached_datasets(track_name)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_key ON cached_datasets(cache_key)
            """)
    
    def _setup_dask(self):
        """Setup Dask client for distributed processing - REQUIRED"""
        try:
            # Use existing client if available, otherwise create new one
            try:
                self.dask_client = Client.current()
                print(f"[INFO] Using existing Dask client: {self.dask_client}")
            except ValueError:
                # Create local cluster with memory limit
                cluster = LocalCluster(
                    n_workers=2,
                    threads_per_worker=2,
                    memory_limit=self.dask_memory_limit,
                    processes=False,  # Use threads for better memory sharing
                    dashboard_address=None  # Disable dashboard for simplicity
                )
                self.dask_client = Client(cluster)
                print(f"[INFO] Created new Dask client: {self.dask_client}")
        except Exception as e:
            raise RuntimeError(f"[CRITICAL ERROR] Failed to setup required Dask client: {e}")
    
    def _generate_cache_key(self, track_name: str, car_name: Optional[str] = None, 
                           filters: Optional[Dict[str, Any]] = None) -> str:
        """Generate unique cache key"""
        key_data = f"{track_name}_{car_name or 'all_cars'}_{filters or {}}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached_sessions(self, track_name: str, car_name: Optional[str] = None,
                           max_age_hours: int = 24) -> Optional[Dict[str, Any]]:
        """
        Get cached session data with intelligent loading
        
        Returns:
            Dictionary with session data or None if not cached/expired
        """
        cache_key = self._generate_cache_key(track_name, car_name)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            self._update_access(cache_key)
            print(f"[INFO] Memory cache hit for {track_name}")
            return self.memory_cache[cache_key]
        
        # Check disk cache
        cached_info = self._get_cache_metadata(cache_key, max_age_hours)
        if cached_info:
            try:
                data = self._load_from_disk(cached_info)
                
                # Load into memory if small enough
                if cached_info["data_size_mb"] < 500:  # 500MB threshold
                    self._add_to_memory_cache(cache_key, data)
                
                self._update_access(cache_key)
                print(f"[INFO] Disk cache hit for {track_name} ({cached_info['storage_type']})")
                return data
                
            except Exception as e:
                print(f"[WARNING] Failed to load cached data: {e}")
                # Clean up corrupted cache entry
                self._remove_cache_entry(cache_key)
        
        return None
    
    async def cache_sessions_streaming(self, track_name: str, sessions_iterator: Iterator[Dict[str, Any]], 
                               car_name: Optional[str] = None,
                               estimated_size_mb: Optional[float] = None) -> bool:
        """
        Cache session data using streaming processing for large datasets
        
        Args:
            track_name: Track name
            sessions_iterator: Iterator yielding session dictionaries
            car_name: Optional car name
            estimated_size_mb: Estimated size for storage decision
            
        Returns:
            True if successful, False otherwise
        """
        cache_key = self._generate_cache_key(track_name, car_name)
        
        try:
            # Both Dask and HDF5 are required - no fallback options
            if not HDF5_AVAILABLE:
                raise RuntimeError("[CRITICAL ERROR] h5py is required for caching but not available")
            if not DASK_AVAILABLE:
                raise RuntimeError("[CRITICAL ERROR] Dask is required for caching but not available")
            
            # Choose storage method based on estimated size
            use_hdf5 = (estimated_size_mb and estimated_size_mb > 100) or self.enable_dask
            
            if use_hdf5:
                success = await self._cache_to_hdf5_streaming(
                    cache_key, track_name, car_name, sessions_iterator
                )
            else:
                success = await self._cache_to_memory_streaming(
                    cache_key, track_name, car_name, sessions_iterator
                )
            
            if success:
                print(f"[INFO] Successfully cached {track_name} using streaming")
                return True
            else:
                raise RuntimeError(f"[CRITICAL ERROR] Failed to cache {track_name} - no fallback available")
                
        except Exception as e:
            print(f"[ERROR] Failed to cache sessions for {track_name}: {e}")
            return False
    
    async def _cache_to_hdf5_streaming(self, cache_key: str, track_name: str, 
                                car_name: Optional[str], 
                                sessions_iterator: Iterator[Dict[str, Any]]) -> bool:
        """Stream sessions to HDF5 storage - REQUIRED"""
        if not HDF5_AVAILABLE:
            raise RuntimeError("[CRITICAL ERROR] HDF5 storage is required but h5py not available")
        if not DASK_AVAILABLE:
            raise RuntimeError("[CRITICAL ERROR] Dask is required but not available")
            
        hdf5_path = self.hdf5_dir / f"{cache_key}.h5"
        temp_hdf5_path = self.hdf5_dir / f"{cache_key}.tmp.h5"
        
        try:
            # Clean up any existing files first
            for path in [hdf5_path, temp_hdf5_path]:
                if path.exists():
                    try:
                        path.unlink()
                        print(f"[INFO] Cleaned up existing file: {path}")
                    except Exception as cleanup_error:
                        print(f"[WARNING] Could not clean up existing file {path}: {cleanup_error}")
            
            session_count = 0
            total_records = 0
            
            # Write to temporary file first, then rename atomically
            with h5py.File(temp_hdf5_path, 'w') as hdf_file:
                # Create groups for organized storage
                sessions_group = hdf_file.create_group("sessions")
                metadata_group = hdf_file.create_group("metadata")
                
                # Process sessions in streaming fashion
                session_idx = 0
                async for session in sessions_iterator:
                    session_id = session.get("sessionId", f"session_{session_idx}")
                    session_data = session.get("data", [])
                    
                    if not session_data:
                        continue
                    
                    # Convert to DataFrame for efficient storage
                    try:
                        df = pd.DataFrame(session_data)
                        
                        # Store as compressed dataset
                        session_dataset = sessions_group.create_group(f"session_{session_idx}")
                        
                        # Store each column as a compressed dataset
                        for column in df.columns:
                            data = df[column].values
                            
                            # Handle different data types
                            if data.dtype == 'object':
                                # Convert objects to strings
                                data = data.astype(str)
                            
                            session_dataset.create_dataset(
                                column, 
                                data=data, 
                                compression='gzip',
                                compression_opts=9
                            )
                        
                        # Store metadata
                        metadata_group.create_dataset(
                            f"session_{session_idx}_id", 
                            data=session_id.encode('utf-8')
                        )
                        
                        session_count += 1
                        total_records += len(session_data)
                        
                    except Exception as e:
                        print(f"[WARNING] Failed to process session {session_idx}: {e}")
                        continue
                    finally:
                        session_idx += 1
                
                # Store cache metadata
                hdf_file.attrs['track_name'] = track_name
                hdf_file.attrs['car_name'] = car_name or 'all_cars'
                hdf_file.attrs['session_count'] = session_count
                hdf_file.attrs['total_records'] = total_records
                hdf_file.attrs['cached_at'] = datetime.now().isoformat()
            
            # Atomically rename temp file to final file
            temp_hdf5_path.rename(hdf5_path)
            print(f"[INFO] Successfully wrote HDF5 file: {hdf5_path}")
            
            # Update metadata database
            file_size_mb = hdf5_path.stat().st_size / (1024 * 1024)
            self._update_cache_metadata(
                cache_key, track_name, car_name, 'hdf5',
                str(hdf5_path), file_size_mb, session_count, total_records
            )
            
            print(f"[INFO] Cached {session_count} sessions, {total_records} records to HDF5 ({file_size_mb:.1f}MB)")
            return True
            
        except Exception as e:
            print(f"[ERROR] HDF5 caching failed: {e}")
            # Clean up any failed files
            for cleanup_path in [hdf5_path, temp_hdf5_path]:
                try:
                    if cleanup_path.exists():
                        cleanup_path.unlink()
                        print(f"[INFO] Cleaned up failed file: {cleanup_path}")
                except Exception as cleanup_error:
                    print(f"[WARNING] Could not clean up failed file {cleanup_path}: {cleanup_error}")
            return False
    
    async def _cache_to_memory_streaming(self, cache_key: str, track_name: str,
                                  car_name: Optional[str],
                                  sessions_iterator: Iterator[Dict[str, Any]]) -> bool:
        """Cache smaller datasets to memory"""
        try:
            sessions_data = []
            session_count = 0
            total_records = 0
            
            async for session in sessions_iterator:
                sessions_data.append(session)
                session_count += 1
                total_records += len(session.get("data", []))
                
                # Prevent memory overflow - switch to HDF5 for large datasets
                if total_records > 50000:  # Limit for memory storage
                    print(f"[INFO] Dataset too large for memory, switching to HDF5")
                    # Create a sync iterator from the data we've collected so far
                    async def remaining_sessions():
                        for remaining_session in sessions_data:
                            yield remaining_session
                        async for remaining_session in sessions_iterator:
                            yield remaining_session
                    
                    return await self._cache_to_hdf5_streaming(
                        cache_key, track_name, car_name, remaining_sessions()
                    )
            
            # Store in memory
            data = {
                "success": True,
                "track_name": track_name,
                "car_name": car_name,
                "sessions": sessions_data,
                "summary": {
                    "total_sessions_retrieved": session_count,
                    "total_telemetry_records": total_records
                }
            }
            
            self._add_to_memory_cache(cache_key, data)
            
            # Update metadata
            self._update_cache_metadata(
                cache_key, track_name, car_name, 'memory',
                None, 0, session_count, total_records
            )
            
            print(f"[INFO] Cached {session_count} sessions to memory ({total_records} records)")
            return True
            
        except Exception as e:
            print(f"[ERROR] Memory caching failed: {e}")
            return False
    
    def _load_from_disk(self, cached_info: Dict[str, Any]) -> Dict[str, Any]:
        """Load cached data from disk"""
        storage_type = cached_info["storage_type"]
        file_path = cached_info["file_path"]
        
        if storage_type == "hdf5":
            return self._load_from_hdf5(file_path, cached_info)
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")
    
    def _load_from_hdf5(self, file_path: str, cached_info: Dict[str, Any]) -> Dict[str, Any]:
        """Load data from HDF5 file - REQUIRED"""
        if not HDF5_AVAILABLE:
            raise RuntimeError("[CRITICAL ERROR] HDF5 loading is required but h5py not available")
            
        try:
            sessions_data = []
            
            with h5py.File(file_path, 'r') as hdf_file:
                sessions_group = hdf_file["sessions"]
                metadata_group = hdf_file["metadata"]
                
                # Load each session
                for session_key in sessions_group.keys():
                    session_dataset = sessions_group[session_key]
                    
                    # Reconstruct DataFrame
                    session_data = {}
                    for column_name in session_dataset.keys():
                        session_data[column_name] = session_dataset[column_name][:]
                    
                    df = pd.DataFrame(session_data)
                    
                    # Get session ID
                    session_id_key = f"{session_key}_id"
                    if session_id_key in metadata_group:
                        session_id = metadata_group[session_id_key][()].decode('utf-8')
                    else:
                        session_id = session_key
                    
                    sessions_data.append({
                        "sessionId": session_id,
                        "data": df.to_dict('records'),
                        "total_telemetry_records": len(df)
                    })
            
            return {
                "success": True,
                "track_name": cached_info["track_name"],
                "car_name": cached_info["car_name"],
                "sessions": sessions_data,
                "summary": {
                    "total_sessions_retrieved": len(sessions_data),
                    "total_telemetry_records": sum(s["total_telemetry_records"] for s in sessions_data)
                }
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to load HDF5 data: {e}")
            raise
    
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
                             storage_type: str, file_path: Optional[str], data_size_mb: float,
                             session_count: int, record_count: int):
        """Update cache metadata in database"""
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cached_datasets 
                (cache_key, track_name, car_name, cached_at, storage_type, file_path,
                 data_size_mb, session_count, record_count, access_count, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """, (
                cache_key, track_name, car_name, datetime.now(),
                storage_type, file_path, data_size_mb, session_count, record_count,
                datetime.now()
            ))
    
    def _update_access(self, cache_key: str):
        """Update access statistics"""
        with sqlite3.connect(self.metadata_db) as conn:
            conn.execute("""
                UPDATE cached_datasets 
                SET access_count = access_count + 1, last_accessed = ?
                WHERE cache_key = ?
            """, (datetime.now(), cache_key))
    
    def _add_to_memory_cache(self, cache_key: str, data: Dict[str, Any]):
        """Add data to memory cache with LRU eviction"""
        # Remove if already exists
        if cache_key in self.memory_cache:
            self.access_order.remove(cache_key)
        
        # Add to memory
        self.memory_cache[cache_key] = data
        self.access_order.append(cache_key)
        
        # Evict oldest if over limit
        while len(self.memory_cache) > self.max_memory_datasets:
            oldest_key = self.access_order.pop(0)
            del self.memory_cache[oldest_key]
            print(f"[INFO] Evicted {oldest_key} from memory cache")
    
    def _remove_cache_entry(self, cache_key: str):
        """Remove corrupted cache entry"""
        # Remove from memory
        if cache_key in self.memory_cache:
            del self.memory_cache[cache_key]
            self.access_order.remove(cache_key)
        
        # Remove from database and disk
        with sqlite3.connect(self.metadata_db) as conn:
            cursor = conn.execute(
                "SELECT file_path FROM cached_datasets WHERE cache_key = ?", 
                (cache_key,)
            )
            row = cursor.fetchone()
            
            if row and row[0]:
                try:
                    Path(row[0]).unlink(missing_ok=True)
                except Exception as e:
                    print(f"[WARNING] Failed to delete cache file {row[0]}: {e}")
            
            conn.execute("DELETE FROM cached_datasets WHERE cache_key = ?", (cache_key,))
    
    def process_large_dataset_streaming(self, track_name: str, 
                                      processing_func: callable,
                                      chunk_size: int = 10000) -> Any:
        """
        Process cached large dataset using streaming with Dask if available
        
        Args:
            track_name: Track to process
            processing_func: Function to apply to data chunks
            chunk_size: Size of processing chunks
            
        Returns:
            Processing results
        """
        cache_key = self._generate_cache_key(track_name)
        cached_info = self._get_cache_metadata(cache_key, max_age_hours=24)
        
        if not cached_info:
            raise ValueError(f"No cached data found for {track_name}")
        
        # Both Dask and HDF5 are required for processing
        if not DASK_AVAILABLE:
            raise RuntimeError("[CRITICAL ERROR] Dask is required for large dataset processing but not available")
        if not HDF5_AVAILABLE:
            raise RuntimeError("[CRITICAL ERROR] h5py is required for large dataset processing but not available")
            
        if cached_info["storage_type"] == "hdf5" and self.enable_dask:
            return self._process_with_dask(cached_info["file_path"], processing_func, chunk_size)
        else:
            return self._process_with_pandas(cached_info, processing_func, chunk_size, cache_key)
    
    def _process_with_dask(self, file_path: str, processing_func: callable, chunk_size: int) -> Any:
        """Process HDF5 data using Dask - REQUIRED"""
        if not HDF5_AVAILABLE:
            raise RuntimeError("[CRITICAL ERROR] HDF5 processing is required but h5py not available")
        if not DASK_AVAILABLE:
            raise RuntimeError("[CRITICAL ERROR] Dask processing is required but not available")
            
        try:
            # Read HDF5 with Dask (simplified approach)
            # Note: This is a basic implementation - more sophisticated handling may be needed
            results = []
            
            with h5py.File(file_path, 'r') as hdf_file:
                sessions_group = hdf_file["sessions"]
                
                for session_key in sessions_group.keys():
                    session_dataset = sessions_group[session_key]
                    
                    # Convert to DataFrame chunk by chunk
                    session_data = {}
                    for column_name in session_dataset.keys():
                        session_data[column_name] = session_dataset[column_name][:]
                    
                    df = pd.DataFrame(session_data)
                    
                    # Process in chunks
                    for i in range(0, len(df), chunk_size):
                        chunk = df.iloc[i:i+chunk_size]
                        result = processing_func(chunk)
                        results.append(result)
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"[CRITICAL ERROR] Dask processing failed with no fallback available: {e}")
    
    def _process_with_pandas(self, cached_info: Dict[str, Any], processing_func: callable, chunk_size: int, cache_key: str) -> Any:
        """Process data using pandas chunking"""
        results = []
        
        # Handle memory vs disk storage
        if cached_info["storage_type"] == "memory":
            # For memory storage, get data directly from memory cache
            if cache_key in self.memory_cache:
                data = self.memory_cache[cache_key]
            else:
                raise ValueError(f"Memory cached data not found for {cache_key}")
        else:
            # For disk storage, load from disk
            data = self._load_from_disk(cached_info)
        
        for session in data["sessions"]:
            session_data = session["data"]
            df = pd.DataFrame(session_data)
            
            # Process in chunks
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size]
                result = processing_func(chunk)
                results.append(result)
        
        return results
    
    def clear_cache(self, track_name: Optional[str] = None):
        """Clear cache for specific track or all tracks"""
        if track_name:
            cache_key = self._generate_cache_key(track_name)
            self._remove_cache_entry(cache_key)
        else:
            # Clear all
            self.memory_cache.clear()
            self.access_order.clear()
            
            # Clear disk storage
            for hdf5_file in self.hdf5_dir.glob("*.h5"):
                hdf5_file.unlink()
            
            # Clear database
            with sqlite3.connect(self.metadata_db) as conn:
                conn.execute("DELETE FROM cached_datasets")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information"""
        with sqlite3.connect(self.metadata_db) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_entries,
                    SUM(data_size_mb) as total_size_mb,
                    storage_type,
                    track_name,
                    session_count,
                    record_count,
                    last_accessed
                FROM cached_datasets 
                GROUP BY track_name, storage_type
                ORDER BY last_accessed DESC
            """)
            
            entries = []
            total_size = 0
            for row in cursor.fetchall():
                entries.append({
                    "track_name": row[3],
                    "storage_type": row[2],
                    "session_count": row[4],
                    "record_count": row[5],
                    "size_mb": row[1] or 0,
                    "last_accessed": row[6]
                })
                total_size += row[1] or 0
        
        return {
            "memory_cache": {
                "entries": len(self.memory_cache),
                "max_entries": self.max_memory_datasets,
                "cached_tracks": list(self.memory_cache.keys())
            },
            "disk_cache": {
                "entries": entries,
                "total_size_mb": total_size,
                "storage_directory": str(self.cache_dir)
            },
            "dask_enabled": self.enable_dask,
            "dask_client": str(self.dask_client) if self.dask_client else None
        }
    
    def __del__(self):
        """Cleanup Dask client"""
        if self.dask_client:
            try:
                self.dask_client.close()
            except:
                pass


# Create shared service instance
hybrid_data_cache = HybridDataCache()

def get_shared_data_cache():
    """
    Get the shared data cache instance
    
    This function provides a consistent way for all services to access 
    the same cache instance, ensuring data is shared across services.
    
    Returns:
        HybridDataCache: The shared cache instance
    """
    return hybrid_data_cache

async def cache_telemetry_sessions(track_name: str, sessions_iterator, estimated_size_mb: float = None):
    """
    Convenience function to cache telemetry sessions using the shared cache
    
    Args:
        track_name: Track name for the sessions
        sessions_iterator: Iterator yielding session data
        estimated_size_mb: Estimated size in MB
        
    Returns:
        bool: True if caching was successful
    """
    return await hybrid_data_cache.cache_sessions_streaming(
        track_name=track_name,
        sessions_iterator=sessions_iterator,
        estimated_size_mb=estimated_size_mb
    )

def get_cached_telemetry_sessions(track_name: str, max_age_hours: int = 24):
    """
    Convenience function to get cached telemetry sessions using the shared cache
    
    Args:
        track_name: Track name to retrieve
        max_age_hours: Maximum age of cached data in hours
        
    Returns:
        Dict or None: Cached session data or None if not found
    """
    return hybrid_data_cache.get_cached_sessions(track_name, max_age_hours=max_age_hours)

def get_shared_cache_info():
    """
    Get information about the shared cache usage
    
    Returns:
        Dict: Cache information including what data is shared across services
    """
    cache_info = hybrid_data_cache.get_cache_info()
    
    # Add information about sharing
    cache_info["sharing_info"] = {
        "is_shared": True,
        "shared_across": [
            "backend_service",
            "full_dataset_ml_service", 
            "imitate_expert_learning_service"
        ],
        "benefits": [
            "Avoids duplicate data caching",
            "Reduces memory usage",
            "Enables data reuse across services",
            "Consistent cache management"
        ]
    }
    
    return cache_info