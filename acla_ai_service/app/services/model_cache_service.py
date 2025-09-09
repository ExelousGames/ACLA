"""
Model Caching Service for AI/ML Models

This service provides intelligent caching capabilities for frequently accessed AI/ML models
to improve prediction performance and reduce backend API calls.
"""

import os
import threading
import time
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib
import logging

# Import cache configuration
try:
    from ..config.cache_config import get_cache_config, get_model_ttl, is_caching_enabled
except ImportError:
    # Fallback if config is not available
    def get_cache_config(environment="development"):
        return {
            "max_cache_size": 100,
            "max_memory_mb": 500,
            "default_ttl_seconds": 3600,
            "cleanup_interval_seconds": 300
        }
    
    def get_model_ttl(model_type, environment="development"):
        return 3600
    
    def is_caching_enabled(operation, environment="development"):
        return True

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Data structure for cache entries"""
    key: str
    data: Any
    metadata: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    ttl_seconds: Optional[int] = None
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired"""
        if self.ttl_seconds is None:
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    def update_access(self):
        """Update access statistics"""
        self.last_accessed = datetime.now()
        self.access_count += 1


class ModelCacheService:
    """
    Thread-safe model caching service with LRU eviction and TTL support
    
    Features:
    - LRU (Least Recently Used) eviction policy
    - TTL (Time To Live) support for automatic expiration
    - Memory usage tracking and limits
    - Thread-safe operations
    - Cache hit/miss statistics
    - Model version tracking
    """
    
    def __init__(self, 
                 environment: str = "development",
                 max_cache_size: Optional[int] = None,
                 max_memory_mb: Optional[int] = None,
                 default_ttl_seconds: Optional[int] = None,
                 cleanup_interval_seconds: Optional[int] = None):
        """
        Initialize the model cache service
        
        Args:
            environment: Environment name (development, testing, production)
            max_cache_size: Override maximum number of models to cache
            max_memory_mb: Override maximum memory usage in MB
            default_ttl_seconds: Override default TTL for cached models
            cleanup_interval_seconds: Override interval for automatic cleanup
        """
        # Load configuration based on environment
        config = get_cache_config(environment)
        
        # Use provided values or fall back to config
        self.environment = environment
        self.max_cache_size = max_cache_size or config["max_cache_size"]
        self.max_memory_mb = max_memory_mb or config["max_memory_mb"]
        self.default_ttl_seconds = default_ttl_seconds or config["default_ttl_seconds"]
        self.cleanup_interval_seconds = cleanup_interval_seconds or config["cleanup_interval_seconds"]
        
        # Store full config for advanced features
        self.config = config
        
        # Thread-safe cache storage using OrderedDict for LRU
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Cache statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'cleanups': 0,
            'total_memory_bytes': 0
        }
        
        # Start background cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._running = True
        self._cleanup_thread.start()
        
        logger.info(f"ModelCacheService initialized for {environment} environment")
        logger.info(f"Configuration - Max size: {self.max_cache_size}, "
                   f"Max memory: {self.max_memory_mb}MB, Default TTL: {self.default_ttl_seconds}s")
    
    def _generate_cache_key(self, model_type: str, track_name: Optional[str] = None, car_name: Optional[str] = None, 
                          model_subtype: Optional[str] = None, 
                          additional_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a unique cache key for a model
        
        Args:
            model_type: Type of model (e.g., 'imitation_learning', 'lap_time_prediction')
            track_name: Track name (optional)
            car_name: Car name (optional)
            model_subtype: Optional subtype (e.g., 'behavior', 'trajectory')
            additional_params: Additional parameters for key generation
            
        Returns:
            Unique cache key string
        """
        key_components = [model_type, track_name or 'any', car_name or 'any']
        
        if model_subtype:
            key_components.append(model_subtype)
        
        if additional_params:
            # Sort for consistent key generation
            sorted_params = sorted(additional_params.items())
            params_str = str(sorted_params)
            key_components.append(params_str)
        
        key = ":".join(key_components)
        
        # Create a hash for very long keys to avoid issues
        if len(key) > 200:
            key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
            key = f"{model_type}:{track_name}:{car_name}:{key_hash}"
        
        return key
    
    def _estimate_memory_size(self, data: Any) -> int:
        """
        Estimate memory size of cached data
        
        Args:
            data: Data to estimate size for
            
        Returns:
            Estimated size in bytes
        """
        try:
            import sys
            import pickle
            
            # Try to get actual size through pickle serialization
            pickled_data = pickle.dumps(data)
            return len(pickled_data)
        except Exception:
            # Fallback to sys.getsizeof with rough estimation
            try:
                return sys.getsizeof(data)
            except Exception:
                # Default estimate
                return 1024  # 1KB default
    
    def get_model_ttl(self, model_type: str) -> int:
        """
        Get TTL for a specific model type based on configuration
        
        Args:
            model_type: Type of model
            
        Returns:
            TTL in seconds
        """
        return get_model_ttl(model_type, self.environment)
    
    def is_caching_enabled_for(self, operation: str) -> bool:
        """
        Check if caching is enabled for a specific operation
        
        Args:
            operation: Operation name
            
        Returns:
            True if caching is enabled, False otherwise
        """
        return is_caching_enabled(operation, self.environment)

    def put(self, 
            model_type: str,
            data: Any,
            track_name: Optional[str] = None, 
            car_name: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            model_subtype: Optional[str] = None,
            ttl_seconds: Optional[int] = None,
            additional_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Cache a model
        
        Args:
            model_type: Type of model
            data: Model data to cache
            track_name: Track name (optional)
            car_name: Car name (optional)
            metadata: Optional metadata
            model_subtype: Optional model subtype
            ttl_seconds: TTL override (None uses default)
            additional_params: Additional parameters for key generation
            
        Returns:
            Cache key used for storage
        """
        cache_key = self._generate_cache_key(
            model_type, track_name, car_name, model_subtype, additional_params
        )
        
        with self._lock:
            # Remove existing entry if present
            if cache_key in self._cache:
                old_entry = self._cache.pop(cache_key)
                self._stats['total_memory_bytes'] -= old_entry.size_bytes
            
            # Estimate memory size
            data_size = self._estimate_memory_size(data)
            
            # Check memory limits before adding
            while (self._stats['total_memory_bytes'] + data_size > self.max_memory_mb * 1024 * 1024 and 
                   len(self._cache) > 0):
                self._evict_lru()
            
            # Check size limits
            while len(self._cache) >= self.max_cache_size and len(self._cache) > 0:
                self._evict_lru()
            
            # Create cache entry
            # Use model-specific TTL if not provided
            model_ttl = ttl_seconds or self.get_model_ttl(model_type)
            
            entry = CacheEntry(
                key=cache_key,
                data=data,
                metadata=metadata or {},
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=0,
                size_bytes=data_size,
                ttl_seconds=model_ttl
            )
            
            # Add to cache
            self._cache[cache_key] = entry
            self._stats['total_memory_bytes'] += data_size
            
            logger.debug(f"Cached model: {cache_key} ({data_size} bytes)")
            
        return cache_key
    
    def get(self, 
            model_type: str,
            track_name: Optional[str] = None,
            car_name: Optional[str] = None,
            model_subtype: Optional[str] = None,
            additional_params: Optional[Dict[str, Any]] = None) -> Optional[Tuple[Any, Dict[str, Any]]]:
        """
        Retrieve a cached model
        
        Args:
            model_type: Type of model
            track_name: Track name (optional)
            car_name: Car name (optional)
            model_subtype: Optional model subtype
            additional_params: Additional parameters for key generation
            
        Returns:
            Tuple of (model_data, metadata) if found, None otherwise
        """
        
        # Generate cache key
        cache_key = self._generate_cache_key(
            model_type, track_name, car_name, model_subtype, additional_params
        )
        
        
        # The with statement in Python is used for resource management, ensuring that resources are properly acquired and released, even 
        with self._lock:
            if cache_key not in self._cache:
                self._stats['misses'] += 1
                logger.debug(f"Cache miss: {cache_key}")
                return None
            
            entry = self._cache[cache_key]
            
            # Check if expired
            if entry.is_expired():
                self._cache.pop(cache_key)
                self._stats['total_memory_bytes'] -= entry.size_bytes
                self._stats['misses'] += 1
                logger.debug(f"Cache expired: {cache_key}")
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)
            entry.update_access()
            
            self._stats['hits'] += 1
            logger.debug(f"Cache hit: {cache_key} (access count: {entry.access_count})")
            
            return entry.data, entry.metadata
    
    def get_by_key(self, cache_key: str) -> Optional[Tuple[Any, Dict[str, Any]]]:
        """
        Retrieve a cached model by exact cache key
        
        Args:
            cache_key: Exact cache key
            
        Returns:
            Tuple of (model_data, metadata) if found, None otherwise
        """
        with self._lock:
            if cache_key not in self._cache:
                self._stats['misses'] += 1
                return None
            
            entry = self._cache[cache_key]
            
            # Check if expired
            if entry.is_expired():
                self._cache.pop(cache_key)
                self._stats['total_memory_bytes'] -= entry.size_bytes
                self._stats['misses'] += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)
            entry.update_access()
            
            self._stats['hits'] += 1
            
            return entry.data, entry.metadata
    
    def invalidate(self, 
                   model_type: str,
                   track_name: Optional[str] = None,
                   car_name: Optional[str] = None,
                   model_subtype: Optional[str] = None,
                   additional_params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Invalidate (remove) a cached model
        
        Args:
            model_type: Type of model
            track_name: Track name (optional)
            car_name: Car name (optional)
            model_subtype: Optional model subtype
            additional_params: Additional parameters for key generation
            
        Returns:
            True if model was found and removed, False otherwise
        """
        cache_key = self._generate_cache_key(
            model_type, track_name, car_name, model_subtype, additional_params
        )
        
        with self._lock:
            if cache_key in self._cache:
                entry = self._cache.pop(cache_key)
                self._stats['total_memory_bytes'] -= entry.size_bytes
                logger.debug(f"Invalidated cache entry: {cache_key}")
                return True
            
            return False
    
    def invalidate_by_pattern(self, pattern: str) -> int:
        """
        Invalidate all cached models matching a pattern
        
        Args:
            pattern: Pattern to match (supports wildcards with *)
            
        Returns:
            Number of models invalidated
        """
        import fnmatch
        
        with self._lock:
            keys_to_remove = []
            for key in self._cache.keys():
                if fnmatch.fnmatch(key, pattern):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                entry = self._cache.pop(key)
                self._stats['total_memory_bytes'] -= entry.size_bytes
            
            logger.info(f"Invalidated {len(keys_to_remove)} cache entries matching pattern: {pattern}")
            return len(keys_to_remove)
    
    def _evict_lru(self):
        """Remove the least recently used item from cache"""
        if not self._cache:
            return
        
        # Remove the first item (least recently used)
        key, entry = self._cache.popitem(last=False)
        self._stats['total_memory_bytes'] -= entry.size_bytes
        self._stats['evictions'] += 1
        logger.debug(f"Evicted LRU entry: {key}")
    
    def _cleanup_worker(self):
        """Background worker for periodic cleanup"""
        while self._running:
            try:
                time.sleep(self.cleanup_interval_seconds)
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
    
    def _cleanup_expired(self):
        """Remove expired entries from cache"""
        with self._lock:
            expired_keys = []
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                entry = self._cache.pop(key)
                self._stats['total_memory_bytes'] -= entry.size_bytes
            
            if expired_keys:
                self._stats['cleanups'] += 1
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def clear(self):
        """Clear all cached models"""
        with self._lock:
            cleared_count = len(self._cache)
            self._cache.clear()
            self._stats['total_memory_bytes'] = 0
            logger.info(f"Cleared {cleared_count} cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            hit_rate = (self._stats['hits'] / (self._stats['hits'] + self._stats['misses']) 
                       if (self._stats['hits'] + self._stats['misses']) > 0 else 0)
            
            return {
                'cache_size': len(self._cache),
                'max_cache_size': self.max_cache_size,
                'memory_usage_mb': self._stats['total_memory_bytes'] / (1024 * 1024),
                'max_memory_mb': self.max_memory_mb,
                'hit_rate': hit_rate,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions'],
                'cleanups': self._stats['cleanups'],
                'entries': [
                    {
                        'key': entry.key,
                        'created_at': entry.created_at.isoformat(),
                        'last_accessed': entry.last_accessed.isoformat(),
                        'access_count': entry.access_count,
                        'size_mb': entry.size_bytes / (1024 * 1024),
                        'ttl_remaining_seconds': (
                            entry.ttl_seconds - (datetime.now() - entry.created_at).total_seconds()
                            if entry.ttl_seconds else None
                        )
                    }
                    for entry in list(self._cache.values())
                ]
            }
    
    def get_cache_info(self, 
                       model_type: str,
                       track_name: Optional[str] = None,
                       car_name: Optional[str] = None,
                       model_subtype: Optional[str] = None,
                       additional_params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Get information about a cached model without accessing it
        
        Args:
            model_type: Type of model
            track_name: Track name (optional)
            car_name: Car name (optional)
            model_subtype: Optional model subtype
            additional_params: Additional parameters for key generation
            
        Returns:
            Model cache information or None if not cached
        """
        cache_key = self._generate_cache_key(
            model_type, track_name, car_name, model_subtype, additional_params
        )
        
        with self._lock:
            if cache_key not in self._cache:
                return None
            
            entry = self._cache[cache_key]
            
            return {
                'key': entry.key,
                'created_at': entry.created_at.isoformat(),
                'last_accessed': entry.last_accessed.isoformat(),
                'access_count': entry.access_count,
                'size_mb': entry.size_bytes / (1024 * 1024),
                'is_expired': entry.is_expired(),
                'ttl_remaining_seconds': (
                    entry.ttl_seconds - (datetime.now() - entry.created_at).total_seconds()
                    if entry.ttl_seconds else None
                ),
                'metadata': entry.metadata
            }
    
    def shutdown(self):
        """Shutdown the cache service"""
        self._running = False
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        self.clear()
        logger.info("ModelCacheService shutdown completed")


# Global cache instance with environment detection
def _detect_environment() -> str:
    """Detect the current environment from environment variables"""
    env = os.getenv("ENVIRONMENT", os.getenv("ENV", "development")).lower()
    valid_environments = ["development", "testing", "production"]
    return env if env in valid_environments else "development"

# Initialize global cache instance
_environment = _detect_environment()
model_cache_service = ModelCacheService(environment=_environment)
