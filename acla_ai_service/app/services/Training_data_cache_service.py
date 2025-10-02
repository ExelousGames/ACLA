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
        """Cache each session as a separate Parquet file"""
        
        try:
            session_count = 0
            total_records = 0
            session_files = []  # Keep track of session files
            
            print(f"[INFO] Starting session-based parquet caching for {cache_key}")
            
            # Process each session separately
            async for session in sessions_iterator:
                session_id = session.get("sessionId", f"session_{session_count}")
                session_data = session.get("data", [])
                
                if not session_data:
                    print(f"[WARNING] Skipping empty session: {session_id}")
                    continue
                
                # Create session-specific file
                session_file = self.parquet_dir / f"{cache_key}_session_{session_id}.parquet"
                
                # Convert session data to DataFrame and save
                session_df = pd.DataFrame(session_data)
                session_df['session_id'] = session_id  # Add session metadata
                
                session_df.to_parquet(session_file, compression=self.compression, index=False)
                session_files.append(session_file)
                
                session_count += 1
                total_records += len(session_data)
                
                print(f"[INFO] Cached session {session_id}: {len(session_data)} records -> {session_file.name}")
            
            if not session_files:
                print(f"[WARNING] No sessions to cache for {cache_key}")
                return False
            
            # Create a manifest file to track all session files
            manifest_file = self.parquet_dir / f"{cache_key}_manifest.txt"
            with open(manifest_file, 'w') as f:
                for session_file in session_files:
                    f.write(f"{session_file.name}\n")
            
            # Calculate total size
            total_size_mb = sum(f.stat().st_size for f in session_files if f.exists()) / (1024 * 1024)
            
            print(f"[INFO] Session-based storage: {total_size_mb:.1f}MB across {len(session_files)} session files")
            
            # Update metadata - use the manifest file as the primary file path
            self._update_cache_metadata(
                cache_key, track_name, car_name, str(manifest_file), 
                total_size_mb, session_count, total_records
            )
            
            print(f"[INFO] Cached {session_count} sessions, {total_records} records ({total_size_mb:.1f}MB)")
            return True
            
        except Exception as e:
            print(f"[ERROR] Session-based parquet caching failed: {e}")
            # Clean up failed files
            try:
                manifest_file = self.parquet_dir / f"{cache_key}_manifest.txt"
                if manifest_file.exists():
                    manifest_file.unlink()
                
                # Clean up any session files that were created
                session_pattern = f"{cache_key}_session_*.parquet"
                for file_path in self.parquet_dir.glob(session_pattern):
                    file_path.unlink()
            except Exception:
                pass
            return False

    def _load_parquet_data(self, cached_info: Dict[str, Any]) -> Dict[str, Any]:
        """Load data from session-based Parquet files using manifest"""
        manifest_file = Path(cached_info["file_path"])
        
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_file}")
        
        print(f"[INFO] Loading session-based parquet dataset using manifest")
        
        # Read manifest to get all session files
        with open(manifest_file, 'r') as f:
            session_files = [line.strip() for line in f.readlines() if line.strip()]
        
        if not session_files:
            raise ValueError(f"Empty manifest file: {manifest_file}")
        
        # Load each session file separately
        sessions_data = []
        total_records = 0
        
        for session_filename in session_files:
            session_path = manifest_file.parent / session_filename
            
            if not session_path.exists():
                print(f"[WARNING] Session file not found: {session_path}")
                continue
            
            # Load session data
            session_df = pd.read_parquet(session_path)
            
            # Extract session_id and clean the data
            if 'session_id' in session_df.columns:
                session_id = session_df['session_id'].iloc[0]  # All rows should have same session_id
                # Remove session_id column from the data
                data_df = session_df.drop(columns=['session_id'])
            else:
                # Extract session_id from filename if not in data
                session_id = session_filename.replace('.parquet', '').split('_session_')[-1]
                data_df = session_df
            
            # Convert to session format
            sessions_data.append({
                "sessionId": str(session_id),
                "data": data_df.to_dict('records')
            })
            
            total_records += len(data_df)
        
        print(f"[INFO] Loaded {len(sessions_data)} sessions with {total_records} total records")
        
        return {
            "success": True,
            "track_name": cached_info.get("track_name"),
            "car_name": cached_info.get("car_name"),
            "sessions": sessions_data,
            "summary": {
                "total_sessions_retrieved": len(sessions_data),
                "total_telemetry_records": total_records
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
    
    def get_cached_data_chunks(self, track_name: str, chunk_size: int = 50000, 
                              car_name: Optional[str] = None, max_age_hours: int = 24) -> Iterator[pd.DataFrame]:
        """
        Get a lazy iterator that yields DataFrame chunks from cached data only when accessed.
        Data is loaded on-demand during iteration to minimize memory usage.
        
        Args:
            track_name: Name of the track to get cached data for
            chunk_size: Size of each chunk to yield
            car_name: Optional car name filter
            max_age_hours: Maximum age of cache to consider valid
            
        Yields:
            pd.DataFrame: Chunks of cached data (loaded only when accessed)
        """
        cache_key = self._generate_cache_key(track_name, car_name)
        cached_info = self._get_cache_metadata(cache_key, max_age_hours)
        
        if not cached_info:
            raise ValueError(f"No cached data found for track: {track_name}")
        
        manifest_file = Path(cached_info["file_path"])
        
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_file}")
        
        # Read manifest to get session file list (minimal memory impact)
        with open(manifest_file, 'r') as f:
            session_files = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"[INFO] Lazy iterator ready for {len(session_files)} sessions with chunk size {chunk_size}")
        
        # Generator function - data is only loaded when yielded
        for session_filename in session_files:
            session_path = manifest_file.parent / session_filename
            
            if not session_path.exists():
                continue
            
            # Load session data only when this iteration is reached
            session_df = pd.read_parquet(session_path)
            session_rows = len(session_df)
            
            if session_rows <= chunk_size:
                # Yield entire session as one chunk
                yield session_df
            else:
                # Yield session in smaller chunks
                for j in range(0, session_rows, chunk_size):
                    chunk = session_df.iloc[j:j+chunk_size].copy()
                    yield chunk
            
            # Clean up memory immediately after processing this session
            del session_df

    
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
        """Remove cache entry and all associated session files"""
        with sqlite3.connect(self.metadata_db) as conn:
            cursor = conn.execute(
                "SELECT file_path FROM cached_datasets WHERE cache_key = ?", 
                (cache_key,)
            )
            row = cursor.fetchone()
            
            if row and row[0]:
                try:
                    manifest_file = Path(row[0])
                    
                    # If manifest exists, remove all session files it references
                    if manifest_file.exists():
                        with open(manifest_file, 'r') as f:
                            session_files = [line.strip() for line in f.readlines() if line.strip()]
                        
                        # Remove each session file
                        for session_filename in session_files:
                            session_path = manifest_file.parent / session_filename
                            session_path.unlink(missing_ok=True)
                        
                        # Remove manifest file
                        manifest_file.unlink(missing_ok=True)
                    
                    # Also clean up any orphaned session files for this cache_key
                    session_pattern = f"{cache_key}_session_*.parquet"
                    for file_path in self.parquet_dir.glob(session_pattern):
                        file_path.unlink(missing_ok=True)
                        
                except Exception as e:
                    print(f"[WARNING] Error cleaning up cache files: {e}")
            
            conn.execute("DELETE FROM cached_datasets WHERE cache_key = ?", (cache_key,))
    

    
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