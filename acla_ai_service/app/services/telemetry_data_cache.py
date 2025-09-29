"""
Telemetry Data Cache Service for ACLA AI Service

This service provides hybrid memory + disk caching for large telemetry datasets
to improve performance and reduce backend load during AI model training.
"""

import json
import gzip
import sqlite3
import hashlib
import warnings
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)


class SQLiteDataCache:
    """SQLite-based local cache for telemetry data"""
    
    def __init__(self, cache_directory: str = "data_cache"):
        self.cache_dir = Path(cache_directory)
        self.cache_dir.mkdir(exist_ok=True)
        self.db_path = self.cache_dir / "telemetry_cache.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cached_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_name TEXT NOT NULL,
                    car_name TEXT,
                    cache_key TEXT UNIQUE NOT NULL,
                    cached_at TIMESTAMP NOT NULL,
                    data_compressed BLOB NOT NULL,
                    session_count INTEGER,
                    data_size_mb REAL,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_track_name ON cached_sessions(track_name)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_key ON cached_sessions(cache_key)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cached_at ON cached_sessions(cached_at)
            """)
    
    def _generate_cache_key(self, track_name: str, car_name: Optional[str] = None, filters: Dict[str, Any] = None) -> str:
        """Generate unique cache key for the data"""
        key_data = f"{track_name}_{car_name or 'all_cars'}_{filters or {}}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _compress_data(self, data: Dict[str, Any]) -> bytes:
        """Compress data for storage"""
        json_str = json.dumps(data, default=str)  # Handle datetime objects
        return gzip.compress(json_str.encode('utf-8'))
    
    def _decompress_data(self, compressed_data: bytes) -> Dict[str, Any]:
        """Decompress data from storage"""
        json_str = gzip.decompress(compressed_data).decode('utf-8')
        return json.loads(json_str)
    
    def get_cached_sessions(self, track_name: str, car_name: Optional[str] = None, 
                          max_age_hours: int = 24, filters: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Get cached session data if available and not expired"""
        cache_key = self._generate_cache_key(track_name, car_name, filters)
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT data_compressed, cached_at, session_count, data_size_mb 
                FROM cached_sessions 
                WHERE cache_key = ? AND cached_at > ?
            """, (cache_key, cutoff_time))
            
            row = cursor.fetchone()
            if row:
                try:
                    data = self._decompress_data(row[0])
                    cached_at = datetime.fromisoformat(row[1]) if isinstance(row[1], str) else row[1]
                    age_hours = (datetime.now() - cached_at).total_seconds() / 3600
                    
                    print(f"[INFO] Loaded {row[2]} cached sessions for {track_name} "
                          f"(age: {age_hours:.1f}h, size: {row[3]:.1f}MB)")
                    return data
                except Exception as e:
                    print(f"[WARNING] Failed to decompress cached data for {track_name}: {e}")
                    # Clean up corrupted cache entry
                    self._remove_corrupted_entry(cache_key)
        
        return None
    
    def _remove_corrupted_entry(self, cache_key: str):
        """Remove corrupted cache entry"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cached_sessions WHERE cache_key = ?", (cache_key,))
            print(f"[INFO] Removed corrupted cache entry: {cache_key}")
        except Exception as e:
            print(f"[WARNING] Failed to remove corrupted cache entry: {e}")
    
    def cache_sessions(self, track_name: str, sessions_data: Dict[str, Any], 
                      car_name: Optional[str] = None, filters: Dict[str, Any] = None):
        """Cache session data locally"""
        cache_key = self._generate_cache_key(track_name, car_name, filters)
        
        try:
            compressed_data = self._compress_data(sessions_data)
            session_count = len(sessions_data.get("sessions", []))
            data_size_mb = len(compressed_data) / (1024 * 1024)
            
            # Calculate original size for metadata
            original_size_mb = len(json.dumps(sessions_data, default=str)) / (1024 * 1024)
            compression_ratio = original_size_mb / data_size_mb if data_size_mb > 0 else 1
            
            metadata = {
                "original_size_mb": original_size_mb,
                "compression_ratio": compression_ratio,
                "filters_applied": filters or {},
                "cached_by": "TelemetryDataCache"
            }
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cached_sessions 
                    (track_name, car_name, cache_key, cached_at, data_compressed, 
                     session_count, data_size_mb, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    track_name, car_name, cache_key, datetime.now(),
                    compressed_data, session_count, data_size_mb,
                    json.dumps(metadata)
                ))
            
            print(f"[INFO] Cached {session_count} sessions for {track_name} "
                  f"({data_size_mb:.1f}MB compressed, {compression_ratio:.1f}x compression)")
            
        except Exception as e:
            print(f"[WARNING] Failed to cache sessions for {track_name}: {e}")
    
    def clear_cache(self, track_name: Optional[str] = None, older_than_hours: Optional[int] = None):
        """Clear cache for specific track or all tracks"""
        with sqlite3.connect(self.db_path) as conn:
            if track_name and older_than_hours:
                cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
                cursor = conn.execute("""
                    DELETE FROM cached_sessions 
                    WHERE track_name = ? AND cached_at < ?
                """, (track_name, cutoff_time))
                deleted_count = cursor.rowcount
                print(f"[INFO] Cleared {deleted_count} old cache entries for {track_name}")
            elif track_name:
                cursor = conn.execute("DELETE FROM cached_sessions WHERE track_name = ?", (track_name,))
                deleted_count = cursor.rowcount
                print(f"[INFO] Cleared {deleted_count} cache entries for {track_name}")
            elif older_than_hours:
                cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
                cursor = conn.execute("DELETE FROM cached_sessions WHERE cached_at < ?", (cutoff_time,))
                deleted_count = cursor.rowcount
                print(f"[INFO] Cleared {deleted_count} old cache entries")
            else:
                cursor = conn.execute("DELETE FROM cached_sessions")
                deleted_count = cursor.rowcount
                print(f"[INFO] Cleared all {deleted_count} cached entries")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with sqlite3.connect(self.db_path) as conn:
            # Get summary statistics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_entries,
                    SUM(data_size_mb) as total_size_mb,
                    AVG(data_size_mb) as avg_size_mb,
                    COUNT(DISTINCT track_name) as unique_tracks
                FROM cached_sessions
            """)
            
            summary = cursor.fetchone()
            
            # Get detailed entries
            cursor = conn.execute("""
                SELECT 
                    track_name,
                    car_name,
                    cached_at,
                    session_count,
                    data_size_mb,
                    metadata
                FROM cached_sessions 
                ORDER BY cached_at DESC
                LIMIT 20
            """)
            
            entries = []
            for row in cursor.fetchall():
                try:
                    metadata = json.loads(row[5]) if row[5] else {}
                except:
                    metadata = {}
                
                entries.append({
                    "track_name": row[0],
                    "car_name": row[1],
                    "cached_at": row[2],
                    "session_count": row[3],
                    "size_mb": row[4],
                    "compression_ratio": metadata.get("compression_ratio", 1.0)
                })
            
            return {
                "total_entries": summary[0] or 0,
                "total_size_mb": summary[1] or 0.0,
                "avg_size_mb": summary[2] or 0.0,
                "unique_tracks": summary[3] or 0,
                "entries": entries
            }
    
    def cleanup_old_entries(self, max_age_days: int = 7) -> int:
        """Clean up entries older than specified days"""
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM cached_sessions WHERE cached_at < ?
            """, (cutoff_time,))
            
            deleted_count = cursor.rowcount
            
            # Vacuum to reclaim space
            conn.execute("VACUUM")
            
        if deleted_count > 0:
            print(f"[INFO] Cleaned up {deleted_count} cache entries older than {max_age_days} days")
        
        return deleted_count


class HybridDataCache:
    """
    Hybrid memory + disk cache for optimal performance
    
    This class provides:
    - Fast in-memory access for recently used data
    - Persistent disk storage for all cached data
    - Automatic memory management with LRU eviction
    - Compression for efficient disk usage
    """
    
    def __init__(self, cache_directory: str = "data_cache", max_memory_sessions: int = 3):
        """
        Initialize hybrid cache
        
        Args:
            cache_directory: Directory for disk cache storage
            max_memory_sessions: Maximum number of sessions to keep in memory
        """
        self.cache_dir = Path(cache_directory)
        self.cache_dir.mkdir(exist_ok=True)
        
        # In-memory cache for recently used data
        self.memory_cache = {}
        self.max_memory_sessions = max_memory_sessions
        self.access_order = []  # LRU tracking
        
        # Disk cache using SQLite
        self.disk_cache = SQLiteDataCache(cache_directory)
        
        print(f"[INFO] Hybrid cache initialized: {cache_directory} "
              f"(max memory sessions: {max_memory_sessions})")
    
    def _generate_memory_key(self, track_name: str, car_name: Optional[str] = None, 
                           filters: Dict[str, Any] = None) -> str:
        """Generate memory cache key"""
        return f"{track_name}_{car_name or 'all_cars'}_{hash(str(sorted((filters or {}).items())))}"
    
    def get_cached_sessions(self, track_name: str, car_name: Optional[str] = None, 
                          max_age_hours: int = 24, filters: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Get cached data from memory first, then disk
        
        Args:
            track_name: Name of the track
            car_name: Optional car name filter
            max_age_hours: Maximum age of cached data in hours
            filters: Optional filters applied to the data
            
        Returns:
            Cached session data or None if not found/expired
        """
        memory_key = self._generate_memory_key(track_name, car_name, filters)
        
        # Check memory cache first
        if memory_key in self.memory_cache:
            # Move to end (most recently used)
            self.access_order.remove(memory_key)
            self.access_order.append(memory_key)
            print(f"[INFO] Memory cache hit for {track_name}")
            return self.memory_cache[memory_key]
        
        # Check disk cache
        data = self.disk_cache.get_cached_sessions(track_name, car_name, max_age_hours, filters)
        if data:
            # Load into memory cache
            self._add_to_memory_cache(memory_key, data)
            print(f"[INFO] Disk cache hit for {track_name}, loaded to memory")
            return data
        
        print(f"[INFO] Cache miss for {track_name}")
        return None
    
    def cache_sessions(self, track_name: str, sessions_data: Dict[str, Any], 
                      car_name: Optional[str] = None, filters: Dict[str, Any] = None):
        """
        Cache data to both memory and disk
        
        Args:
            track_name: Name of the track
            sessions_data: Session data to cache
            car_name: Optional car name
            filters: Optional filters that were applied
        """
        memory_key = self._generate_memory_key(track_name, car_name, filters)
        
        # Cache to disk first (persistent storage)
        self.disk_cache.cache_sessions(track_name, sessions_data, car_name, filters)
        
        # Cache to memory (fast access)
        self._add_to_memory_cache(memory_key, sessions_data)
        
        session_count = len(sessions_data.get("sessions", []))
        print(f"[INFO] Cached {session_count} sessions for {track_name} to memory and disk")
    
    def _add_to_memory_cache(self, memory_key: str, data: Dict[str, Any]):
        """Add data to memory cache with LRU eviction"""
        # Remove if already exists
        if memory_key in self.memory_cache:
            self.access_order.remove(memory_key)
        
        # Add to memory
        self.memory_cache[memory_key] = data
        self.access_order.append(memory_key)
        
        # Evict oldest if over limit
        while len(self.memory_cache) > self.max_memory_sessions:
            oldest_key = self.access_order.pop(0)
            del self.memory_cache[oldest_key]
            print(f"[INFO] Evicted {oldest_key.split('_')[0]} from memory cache")
    
    def clear_cache(self, track_name: Optional[str] = None, clear_memory: bool = True, 
                   clear_disk: bool = True):
        """
        Clear cache (memory and/or disk)
        
        Args:
            track_name: Optional specific track to clear
            clear_memory: Whether to clear memory cache
            clear_disk: Whether to clear disk cache
        """
        if clear_memory:
            if track_name:
                # Remove from memory cache
                keys_to_remove = [key for key in self.memory_cache.keys() 
                                 if key.startswith(f"{track_name}_")]
                for key in keys_to_remove:
                    del self.memory_cache[key]
                    if key in self.access_order:
                        self.access_order.remove(key)
                print(f"[INFO] Cleared memory cache for {track_name}")
            else:
                self.memory_cache.clear()
                self.access_order.clear()
                print("[INFO] Cleared all memory cache")
        
        if clear_disk:
            self.disk_cache.clear_cache(track_name)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information"""
        disk_info = self.disk_cache.get_cache_info()
        
        # Memory cache info
        memory_tracks = set()
        memory_size_estimate = 0
        for key, data in self.memory_cache.items():
            track_name = key.split('_')[0]
            memory_tracks.add(track_name)
            # Rough size estimate (JSON size in MB)
            memory_size_estimate += len(json.dumps(data, default=str)) / (1024 * 1024)
        
        return {
            "memory_cache": {
                "entries": len(self.memory_cache),
                "max_entries": self.max_memory_sessions,
                "cached_tracks": list(memory_tracks),
                "estimated_size_mb": memory_size_estimate,
                "access_order": [key.split('_')[0] for key in self.access_order]
            },
            "disk_cache": disk_info,
            "total_unique_tracks": len(memory_tracks.union(
                set(entry["track_name"] for entry in disk_info.get("entries", []))
            ))
        }
    
    def print_cache_info(self):
        """Print formatted cache information"""
        info = self.get_cache_info()
        
        print("\n" + "="*70)
        print("TELEMETRY DATA CACHE INFORMATION")
        print("="*70)
        
        # Memory cache info
        mem_info = info["memory_cache"]
        print(f"Memory Cache: {mem_info['entries']}/{mem_info['max_entries']} sessions")
        print(f"Memory Size: {mem_info['estimated_size_mb']:.1f}MB")
        if mem_info['access_order']:
            print(f"Recent Access: {' -> '.join(mem_info['access_order'][-3:])}")
        
        # Disk cache info
        disk_info = info["disk_cache"]
        print(f"\nDisk Cache: {disk_info['total_entries']} entries")
        print(f"Disk Size: {disk_info['total_size_mb']:.1f}MB")
        print(f"Avg Entry Size: {disk_info['avg_size_mb']:.2f}MB")
        
        print(f"\nTotal Unique Tracks: {info['total_unique_tracks']}")
        
        # Recent entries
        if disk_info.get("entries"):
            print("\nRecent Cache Entries:")
            for entry in disk_info["entries"][:5]:
                age_hours = (datetime.now() - datetime.fromisoformat(entry["cached_at"])).total_seconds() / 3600
                print(f"  {entry['track_name']}: {entry['session_count']} sessions, "
                      f"{entry['size_mb']:.1f}MB, {age_hours:.1f}h ago")
        
        print("="*70 + "\n")
    
    def optimize_cache(self, max_disk_size_mb: float = 1000, max_age_days: int = 7) -> Dict[str, Any]:
        """
        Optimize cache by cleaning up old/large entries
        
        Args:
            max_disk_size_mb: Maximum disk cache size in MB
            max_age_days: Maximum age of entries to keep
            
        Returns:
            Optimization results
        """
        print(f"[INFO] Optimizing cache (max size: {max_disk_size_mb}MB, max age: {max_age_days} days)")
        
        # Clean up old entries
        old_entries_removed = self.disk_cache.cleanup_old_entries(max_age_days)
        
        # Get current size
        cache_info = self.disk_cache.get_cache_info()
        current_size = cache_info["total_size_mb"]
        
        size_entries_removed = 0
        if current_size > max_disk_size_mb:
            # TODO: Implement size-based cleanup (remove largest/oldest entries)
            print(f"[WARNING] Cache size ({current_size:.1f}MB) exceeds limit ({max_disk_size_mb}MB)")
            print("[INFO] Consider manually clearing some cache entries")
        
        return {
            "old_entries_removed": old_entries_removed,
            "size_entries_removed": size_entries_removed,
            "final_size_mb": cache_info["total_size_mb"],
            "final_entry_count": cache_info["total_entries"]
        }
    
    def preload_track_data(self, track_name: str, car_name: Optional[str] = None) -> bool:
        """
        Preload track data into memory cache
        
        Args:
            track_name: Track to preload
            car_name: Optional car filter
            
        Returns:
            True if data was loaded, False if not found
        """
        data = self.get_cached_sessions(track_name, car_name)
        if data:
            print(f"[INFO] Preloaded {track_name} data into memory cache")
            return True
        else:
            print(f"[INFO] No cached data found for {track_name}")
            return False


# Create a singleton instance for easy importing
telemetry_data_cache = HybridDataCache()

if __name__ == "__main__":
    # Example usage and testing
    print("Telemetry Data Cache initialized. Example usage:")
    
    # Create cache instance
    cache = HybridDataCache("test_cache", max_memory_sessions=2)
    
    # Print cache info
    cache.print_cache_info()
    
    # Example data structure
    example_data = {
        "sessions": [
            {"session_id": "test_1", "lap_count": 10},
            {"session_id": "test_2", "lap_count": 15}
        ]
    }
    
    # Cache some data
    cache.cache_sessions("monza", example_data, "ferrari_488")
    
    # Retrieve data
    retrieved = cache.get_cached_sessions("monza", "ferrari_488")
    print(f"Retrieved: {len(retrieved.get('sessions', []))} sessions")
    
    # Print updated cache info
    cache.print_cache_info()