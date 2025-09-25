"""
Configuration file for Model Cache Service
"""

# Cache Configuration
CACHE_CONFIG = {
    # Maximum number of models to cache simultaneously
    "max_cache_size": 100,
    
    # Maximum memory usage in MB
    "max_memory_mb": 4000,  # Increased to 4GB to handle large models
    
    # Default Time To Live for cached models in seconds
    "default_ttl_seconds": 7200,  # 2 hours - longer for large models
    
    # Interval for automatic cleanup of expired models in seconds
    "cleanup_interval_seconds": 300,  # 5 minutes
    
    # Model-specific TTL settings
    "model_specific_ttl": {
        "imitation_learning": 7200,    # 2 hours for large models
        "transformer_expert_action": 7200,  # 2 hours for large models
        "corner_identification": 7200,  # 2 hours 
        "tire_grip_analysis": 7200,    # 2 hours
        "lap_time_prediction": 3600,   # 1 hour
        "performance_classification": 7200,  # 2 hours
        "sklearn": 3600,               # 1 hour (general sklearn models)
    },
    
    # Memory limits for different model types (in MB)
    "model_memory_limits": {
        "imitation_learning": 1000,    # 1GB per imitation model (large models)
        "transformer_expert_action": 1000,  # 1GB per transformer model
        "corner_identification": 200,  # 200MB per corner model
        "tire_grip_analysis": 200,     # 200MB per tire model
        "sklearn": 50,                 # 50MB per sklearn model
        "default": 100                 # 100MB default limit
    },
    
    # Enable/disable caching for different operations
    "cache_enabled": {
        "imitation_learning": True,
        "sklearn_models": True,
        "backend_responses": True,
        "predictions": False  # Typically don't cache predictions themselves
    },
    
    # Preloading configuration
    "preload_config": {
        # Common track/car combinations to preload on startup
        "common_combinations": [
            {"track": "Monza", "car": "BMW_M4_GT3"},
            {"track": "Spa", "car": "Mercedes_AMG_GT3"},
            {"track": "Silverstone", "car": "Audi_R8_LMS"},
            {"track": "Nurburgring", "car": "Porsche_911_GT3_R"},
        ],
        
        # Whether to preload models on service startup
        "preload_on_startup": False,
        
        # Maximum time to spend preloading (seconds)
        "max_preload_time": 60
    },
    
    # Cache warming strategy
    "cache_warming": {
        "enabled": True,
        "warm_on_first_miss": True,
        "background_refresh": False,
        "refresh_threshold": 0.8  # Refresh when TTL is 80% expired
    },
    
    # Logging configuration for cache operations
    "logging": {
        "log_hits": True,
        "log_misses": True,
        "log_evictions": True,
        "log_memory_usage": True,
        "detailed_stats": False
    },
    
    # Performance tuning
    "performance": {
        # Use compression for large models
        "compress_large_models": True,
        "compression_threshold_mb": 100,  # Compress models larger than 100MB
        
        # Async operations
        "async_cache_operations": True,
        "max_concurrent_loads": 2,  # Reduced for large models
        
        # Memory management
        "aggressive_cleanup": False,
        "memory_pressure_threshold": 0.85,  # Start cleanup at 85% memory usage
        "large_model_priority": True  # Prioritize keeping large models in cache
    }
}

# Environment-specific overrides
ENVIRONMENT_OVERRIDES = {
    "development": {
        "max_cache_size": 10,  # Reduced count but higher memory
        "max_memory_mb": 2000,  # 2GB for development
        "default_ttl_seconds": 1800,  # 30 minutes for development
        "logging": {
            "detailed_stats": True
        }
    },
    
    "testing": {
        "max_cache_size": 3,
        "max_memory_mb": 500,  # 500MB for testing
        "default_ttl_seconds": 300,   # 5 minutes for tests
        "cleanup_interval_seconds": 10,
        "cache_enabled": {
            "imitation_learning": True,
            "sklearn_models": False,
            "backend_responses": False,
            "predictions": False
        }
    },
    
    "production": {
        "max_cache_size": 50,  # Fewer models but larger memory
        "max_memory_mb": 6000,  # 6GB for production
        "default_ttl_seconds": 14400,  # 4 hours for production
        "preload_config": {
            "preload_on_startup": True
        },
        "performance": {
            "compress_large_models": True,
            "async_cache_operations": True,
            "aggressive_cleanup": True,
            "large_model_priority": True
        }
    }
}

def get_cache_config(environment: str = "development") -> dict:
    """
    Get cache configuration for the specified environment
    
    Args:
        environment: Environment name (development, testing, production)
        
    Returns:
        Dictionary with cache configuration
    """
    config = CACHE_CONFIG.copy()
    
    # Apply environment-specific overrides
    if environment in ENVIRONMENT_OVERRIDES:
        overrides = ENVIRONMENT_OVERRIDES[environment]
        
        # Deep merge the configurations
        for key, value in overrides.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
    
    return config

def get_model_ttl(model_type: str, environment: str = "development") -> int:
    """
    Get TTL for a specific model type
    
    Args:
        model_type: Type of model
        environment: Environment name
        
    Returns:
        TTL in seconds
    """
    config = get_cache_config(environment)
    return config["model_specific_ttl"].get(model_type, config["default_ttl_seconds"])

def is_caching_enabled(operation: str, environment: str = "development") -> bool:
    """
    Check if caching is enabled for a specific operation
    
    Args:
        operation: Operation name
        environment: Environment name
        
    Returns:
        True if caching is enabled, False otherwise
    """
    config = get_cache_config(environment)
    return config["cache_enabled"].get(operation, False)


# Example usage
if __name__ == "__main__":
    import json
    
    print("Cache Configuration Examples:")
    print("=" * 40)
    
    for env in ["development", "testing", "production"]:
        print(f"\n{env.upper()} Configuration:")
        config = get_cache_config(env)
        
        print(f"  Max Cache Size: {config['max_cache_size']}")
        print(f"  Max Memory: {config['max_memory_mb']} MB") 
        print(f"  Default TTL: {config['default_ttl_seconds']}s")
        print(f"  Imitation Learning TTL: {get_model_ttl('imitation_learning', env)}s")
        print(f"  Preload on Startup: {config['preload_config']['preload_on_startup']}")
    
    print("\nModel Type TTL Examples:")
    print("-" * 25)
    for model_type in ["imitation_learning", "sklearn", "unknown_type"]:
        ttl = get_model_ttl(model_type, "production")
        print(f"  {model_type}: {ttl}s")
    
    print("\nCaching Enabled Check:")
    print("-" * 22)
    for operation in ["imitation_learning", "sklearn_models", "predictions"]:
        enabled = is_caching_enabled(operation, "production")
        print(f"  {operation}: {enabled}")
