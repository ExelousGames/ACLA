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

# Dask imports for large dataset processing
try:
    import dask
    import dask.dataframe as dd
    from dask.distributed import LocalCluster, Client
    import h5py
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    print("[WARNING] Dask not available - falling back to basic caching")

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
                 dask_memory_limit: str = "4GB"):
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
        
        # Dask setup
        self.enable_dask = enable_dask and DASK_AVAILABLE
        self.dask_client = None
        self.dask_memory_limit = dask_memory_limit
        
        if self.enable_dask:
            self._setup_dask()
        
        print(f"[INFO] Hybrid data cache initialized at {self.cache_dir}")
        print(f"[INFO] Dask enabled: {self.enable_dask}")
    
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
        """Setup Dask client for distributed processing"""
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
            print(f"[WARNING] Failed to setup Dask: {e}")
            self.enable_dask = False
            self.dask_client = None
    
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
    
    def cache_sessions_streaming(self, track_name: str, sessions_iterator: Iterator[Dict[str, Any]], 
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
            # Choose storage method based on estimated size
            use_hdf5 = (estimated_size_mb and estimated_size_mb > 100) or self.enable_dask
            
            if use_hdf5:
                success = self._cache_to_hdf5_streaming(
                    cache_key, track_name, car_name, sessions_iterator
                )
            else:
                success = self._cache_to_memory_streaming(
                    cache_key, track_name, car_name, sessions_iterator
                )
            
            if success:
                print(f"[INFO] Successfully cached {track_name} using streaming")
                return True
            else:
                print(f"[WARNING] Failed to cache {track_name}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Failed to cache sessions for {track_name}: {e}")
            return False
    
    def _cache_to_hdf5_streaming(self, cache_key: str, track_name: str, 
                                car_name: Optional[str], 
                                sessions_iterator: Iterator[Dict[str, Any]]) -> bool:
        """Stream sessions to HDF5 storage"""
        hdf5_path = self.hdf5_dir / f"{cache_key}.h5"
        
        try:
            session_count = 0
            total_records = 0
            
            with h5py.File(hdf5_path, 'w') as hdf_file:
                # Create groups for organized storage
                sessions_group = hdf_file.create_group("sessions")
                metadata_group = hdf_file.create_group("metadata")
                
                # Process sessions in streaming fashion
                for session_idx, session in enumerate(sessions_iterator):
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
                                compression='lz4',
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
                
                # Store cache metadata
                hdf_file.attrs['track_name'] = track_name
                hdf_file.attrs['car_name'] = car_name or 'all_cars'
                hdf_file.attrs['session_count'] = session_count
                hdf_file.attrs['total_records'] = total_records
                hdf_file.attrs['cached_at'] = datetime.now().isoformat()
            
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
            if hdf5_path.exists():
                hdf5_path.unlink()
            return False
    
    def _cache_to_memory_streaming(self, cache_key: str, track_name: str,
                                  car_name: Optional[str],
                                  sessions_iterator: Iterator[Dict[str, Any]]) -> bool:
        """Cache smaller datasets to memory"""
        try:
            sessions_data = []
            session_count = 0
            total_records = 0
            
            for session in sessions_iterator:
                sessions_data.append(session)
                session_count += 1
                total_records += len(session.get("data", []))
                
                # Prevent memory overflow
                if total_records > 50000:  # Limit for memory storage
                    print(f"[INFO] Dataset too large for memory, switching to HDF5")
                    return self._cache_to_hdf5_streaming(
                        cache_key, track_name, car_name, iter(sessions_data)
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
        """Load data from HDF5 file"""
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
        
        if cached_info["storage_type"] == "hdf5" and self.enable_dask:
            return self._process_with_dask(cached_info["file_path"], processing_func, chunk_size)
        else:
            return self._process_with_pandas(cached_info, processing_func, chunk_size)
    
    def _process_with_dask(self, file_path: str, processing_func: callable, chunk_size: int) -> Any:
        """Process HDF5 data using Dask"""
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
            print(f"[WARNING] Dask processing failed, falling back to pandas: {e}")
            return self._process_with_pandas({"file_path": file_path}, processing_func, chunk_size)
    
    def _process_with_pandas(self, cached_info: Dict[str, Any], processing_func: callable, chunk_size: int) -> Any:
        """Process data using pandas chunking"""
        results = []
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


# Create service instance
hybrid_data_cache = HybridDataCache()