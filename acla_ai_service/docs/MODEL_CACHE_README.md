# Model Caching Service

A high-performance, thread-safe caching service designed for AI/ML models in the ACLA system. This service significantly improves prediction performance by caching frequently accessed models in memory, reducing backend API calls and network latency.

## Features

- 🚀 **High Performance**: In-memory caching with LRU eviction
- 🔒 **Thread-Safe**: Safe for concurrent access in multi-threaded environments
- ⏰ **TTL Support**: Configurable Time-To-Live for automatic cache expiration
- 📊 **Memory Management**: Automatic memory usage tracking and limits
- 📈 **Statistics**: Comprehensive cache hit/miss statistics and performance metrics
- 🎛️ **Configurable**: Environment-specific configurations (dev, test, prod)
- 🔧 **Management APIs**: RESTful endpoints for cache monitoring and control

## Quick Start

### Basic Usage

```python
from app.services.scikit_ml_service import TelemetryMLService

# Initialize the service (cache is automatically initialized)
ml_service = TelemetryMLService()

# Get expert guidance (automatically caches the model)
guidance = await ml_service.get_imitation_learning_expert_guidance(
    current_telemetry=telemetry_data,
    trackName="Monza",
    carName="BMW_M4_GT3",
    guidance_type="both"
)

# Subsequent calls will use cached model (much faster!)
guidance_2 = await ml_service.get_imitation_learning_expert_guidance(
    current_telemetry=new_telemetry,
    trackName="Monza", 
    carName="BMW_M4_GT3",
    guidance_type="actions"
)
```

### Cache Management

```python
# Get cache statistics
stats = ml_service.get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Memory usage: {stats['memory_usage_mb']:.1f} MB")

# Check if a specific model is cached
model_info = ml_service.get_model_cache_info(
    model_type="imitation_learning",
    track_name="Monza",
    car_name="BMW_M4_GT3"
)

# Invalidate specific model
ml_service.invalidate_model_cache(
    model_type="imitation_learning",
    track_name="Monza",
    car_name="BMW_M4_GT3"
)

# Clear all cached models
ml_service.clear_all_cache()
```

## Configuration

The cache service uses environment-specific configurations defined in `app/config/cache_config.py`:

### Environment Variables

Set the `ENVIRONMENT` or `ENV` variable to control cache behavior:

```bash
# Development (default)
export ENVIRONMENT=development

# Testing
export ENVIRONMENT=testing  

# Production
export ENVIRONMENT=production
```

### Configuration Options

| Environment | Max Models | Max Memory | Default TTL | Features |
|-------------|------------|------------|-------------|----------|
| Development | 20 | 100 MB | 10 min | Detailed logging |
| Testing | 5 | 50 MB | 1 min | Minimal caching |
| Production | 200 | 1000 MB | 2 hours | Full features |

### Model-Specific TTL

Different model types have different cache lifetimes:

- **Imitation Learning**: 30 minutes
- **Sklearn Models**: 1 hour  
- **Performance Models**: 2 hours

## Performance Benefits

### Before Caching
```
Request 1: Backend API call (500ms) + Deserialization (200ms) = 700ms
Request 2: Backend API call (500ms) + Deserialization (200ms) = 700ms  
Request 3: Backend API call (500ms) + Deserialization (200ms) = 700ms
Total: 2100ms for 3 predictions
```

### With Caching
```
Request 1: Backend API call (500ms) + Deserialization (200ms) + Cache (1ms) = 701ms
Request 2: Cache hit (1ms) = 1ms
Request 3: Cache hit (1ms) = 1ms  
Total: 703ms for 3 predictions (66% faster!)
```

## API Endpoints

If you're using the FastAPI integration, the following endpoints are available:

### Get Cache Statistics
```http
GET /api/cache/stats
```

### Check Model Cache Status
```http
GET /api/cache/info/{model_type}/{track_name}/{car_name}?model_subtype=optional
```

### Invalidate Specific Model
```http
DELETE /api/cache/invalidate/{model_type}/{track_name}/{car_name}
```

### Invalidate All Models for Track
```http
DELETE /api/cache/invalidate/track/{track_name}
```

### Clear All Cache
```http
DELETE /api/cache/clear
```

### Preload Models
```http
POST /api/cache/preload?track_name=Monza&car_name=BMW_M4_GT3&model_types=imitation_learning
```

### Health Check
```http
GET /api/cache/health
```

## Examples

### Example 1: Real-time Telemetry Processing

```python
async def process_real_time_telemetry(session_data):
    """Process real-time telemetry with caching for optimal performance"""
    ml_service = TelemetryMLService()
    
    # Preload models for the session
    await ml_service.preload_models_for_session(
        track_name=session_data["track"],
        car_name=session_data["car"],
        model_types=["imitation_learning"]
    )
    
    # Process telemetry points (subsequent calls use cache)
    for telemetry_point in session_data["telemetry_stream"]:
        guidance = await ml_service.get_imitation_learning_expert_guidance(
            current_telemetry=telemetry_point,
            trackName=session_data["track"],
            carName=session_data["car"],
            guidance_type="both"
        )
        
        # Send guidance to user (very fast due to caching)
        await send_guidance_to_user(guidance)
```

### Example 2: Cache Monitoring

```python
def monitor_cache_performance():
    """Monitor cache performance and health"""
    ml_service = TelemetryMLService()
    
    stats = ml_service.get_cache_stats()
    
    print("Cache Performance Report")
    print("=" * 30)
    print(f"Cache Size: {stats['cache_size']}/{stats['max_cache_size']}")
    print(f"Memory Usage: {stats['memory_usage_mb']:.1f}/{stats['max_memory_mb']} MB")
    print(f"Hit Rate: {stats['hit_rate']:.2%}")
    print(f"Total Requests: {stats['hits'] + stats['misses']}")
    
    # Alert if performance is poor
    if stats['hit_rate'] < 0.5:
        print("⚠️  Low cache hit rate - consider adjusting TTL settings")
    
    if stats['memory_usage_mb'] > stats['max_memory_mb'] * 0.8:
        print("⚠️  High memory usage - consider clearing old entries")
```

### Example 3: Session-based Caching

```python
class RacingSessionManager:
    def __init__(self):
        self.ml_service = TelemetryMLService()
    
    async def start_session(self, track_name: str, car_name: str):
        """Start racing session with model preloading"""
        print(f"Starting session: {track_name} with {car_name}")
        
        # Preload all relevant models
        preload_result = self.ml_service.preload_models_for_session(
            track_name=track_name,
            car_name=car_name,
            model_types=["imitation_learning", "lap_time_prediction"]
        )
        
        print(f"Preloaded {len(preload_result['preloaded_models'])} models")
        return preload_result
    
    async def end_session(self, track_name: str, car_name: str):
        """End session and optionally clean up cache"""
        # Optionally invalidate session-specific cache
        count = self.ml_service.invalidate_track_cache(track_name)
        print(f"Cleaned up {count} cached models for {track_name}")
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Check cache statistics: `ml_service.get_cache_stats()`
   - Clear cache if needed: `ml_service.clear_all_cache()`
   - Adjust memory limits in configuration

2. **Low Hit Rate**
   - Check TTL settings in configuration
   - Consider preloading frequently used models
   - Monitor model usage patterns

3. **Cache Misses**
   - Models may have expired (check TTL)
   - Model parameters might not match exactly
   - Backend connectivity issues

### Debugging

Enable detailed logging:

```python
import logging
logging.getLogger('model_cache_service').setLevel(logging.DEBUG)
```

Check cache health:

```python
stats = ml_service.get_cache_stats()
print(f"Cache entries: {len(stats['entries'])}")
for entry in stats['entries']:
    print(f"  {entry['key']}: {entry['access_count']} accesses")
```

## Best Practices

1. **Preload Models**: Use `preload_models_for_session()` for known usage patterns
2. **Monitor Performance**: Regularly check cache hit rates and memory usage
3. **Environment-Specific Config**: Use appropriate configurations for each environment
4. **Clean Up**: Invalidate cache when models are updated or retrained
5. **Memory Management**: Monitor memory usage in production environments

## Integration with FastAPI

To integrate cache management APIs with your FastAPI application:

```python
from fastapi import FastAPI
from app.api.cache_management import cache_router

app = FastAPI()
app.include_router(cache_router)

# Cache endpoints will be available at /api/cache/*
```

## Performance Metrics

Expected performance improvements:

- **First Request**: Similar to non-cached (slight overhead for caching)
- **Subsequent Requests**: 90-99% faster response times
- **Memory Usage**: Configurable limits with automatic cleanup
- **Network Reduction**: Significant reduction in backend API calls

## Support

For issues or questions about the caching service:

1. Check the logs for error messages
2. Use the health check endpoint for diagnostics
3. Monitor cache statistics for performance insights
4. Review configuration settings for your environment
