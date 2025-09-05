"""
Cache Management API endpoints for monitoring and managing the model cache
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from app.services.scikit_ml_service import TelemetryMLService

# Create router for cache management endpoints
cache_router = APIRouter(prefix="/api/cache", tags=["cache"])

# Initialize ML service
ml_service = TelemetryMLService()

@cache_router.get("/stats")
async def get_cache_statistics() -> Dict[str, Any]:
    """
    Get comprehensive cache statistics
    
    Returns:
        Dictionary with cache statistics
    """
    try:
        stats = ml_service.get_cache_stats()
        return {
            "success": True,
            "data": stats,
            "timestamp": stats.get("entries", [{}])[0].get("created_at") if stats.get("entries") else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")


@cache_router.get("/info/{model_type}/{track_name}/{car_name}")
async def get_model_cache_info(
    model_type: str,
    track_name: str,
    car_name: str,
    model_subtype: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """
    Get information about a specific cached model
    
    Args:
        model_type: Type of model
        track_name: Track name
        car_name: Car name
        model_subtype: Optional model subtype
        
    Returns:
        Model cache information or error if not found
    """
    try:
        info = ml_service.get_model_cache_info(
            model_type=model_type,
            track_name=track_name,
            car_name=car_name,
            model_subtype=model_subtype
        )
        
        if info:
            return {
                "success": True,
                "data": info,
                "cached": True
            }
        else:
            return {
                "success": True,
                "data": None,
                "cached": False,
                "message": "Model not found in cache"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@cache_router.delete("/invalidate/{model_type}/{track_name}/{car_name}")
async def invalidate_model_cache(
    model_type: str,
    track_name: str,
    car_name: str,
    model_subtype: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """
    Invalidate a specific cached model
    
    Args:
        model_type: Type of model to invalidate
        track_name: Track name
        car_name: Car name
        model_subtype: Optional model subtype
        
    Returns:
        Result of invalidation
    """
    try:
        invalidated = ml_service.invalidate_model_cache(
            model_type=model_type,
            track_name=track_name,
            car_name=car_name,
            model_subtype=model_subtype
        )
        
        return {
            "success": True,
            "invalidated": invalidated,
            "message": f"Model {'invalidated' if invalidated else 'not found in cache'}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to invalidate model: {str(e)}")


@cache_router.delete("/invalidate/track/{track_name}")
async def invalidate_track_cache(track_name: str) -> Dict[str, Any]:
    """
    Invalidate all cached models for a specific track
    
    Args:
        track_name: Track name to invalidate
        
    Returns:
        Number of models invalidated
    """
    try:
        count = ml_service.invalidate_track_cache(track_name)
        
        return {
            "success": True,
            "invalidated_count": count,
            "message": f"Invalidated {count} models for track {track_name}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to invalidate track cache: {str(e)}")


@cache_router.delete("/invalidate/car/{car_name}")
async def invalidate_car_cache(car_name: str) -> Dict[str, Any]:
    """
    Invalidate all cached models for a specific car
    
    Args:
        car_name: Car name to invalidate
        
    Returns:
        Number of models invalidated
    """
    try:
        count = ml_service.invalidate_car_cache(car_name)
        
        return {
            "success": True,
            "invalidated_count": count,
            "message": f"Invalidated {count} models for car {car_name}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to invalidate car cache: {str(e)}")


@cache_router.delete("/clear")
async def clear_all_cache() -> Dict[str, Any]:
    """
    Clear all cached models
    
    Returns:
        Confirmation of cache clearing
    """
    try:
        # Get count before clearing
        stats = ml_service.get_cache_stats()
        count = stats.get("cache_size", 0)
        
        ml_service.clear_all_cache()
        
        return {
            "success": True,
            "cleared_count": count,
            "message": f"Cleared {count} models from cache"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@cache_router.post("/preload")
async def preload_models(
    track_name: str = Query(...),
    car_name: str = Query(...),
    model_types: List[str] = Query(default=["imitation_learning"])
) -> Dict[str, Any]:
    """
    Preload models for a racing session
    
    Args:
        track_name: Track name
        car_name: Car name
        model_types: List of model types to preload
        
    Returns:
        Preload results
    """
    try:
        results = ml_service.preload_models_for_session(
            track_name=track_name,
            car_name=car_name,
            model_types=model_types
        )
        
        return {
            "success": True,
            "data": results,
            "message": f"Preloaded {len(results['preloaded_models'])} models"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to preload models: {str(e)}")


@cache_router.get("/health")
async def cache_health_check() -> Dict[str, Any]:
    """
    Get cache service health information
    
    Returns:
        Cache health status
    """
    try:
        stats = ml_service.get_cache_stats()
        
        # Calculate health metrics
        memory_usage_percent = (stats["memory_usage_mb"] / stats["max_memory_mb"]) * 100
        cache_usage_percent = (stats["cache_size"] / stats["max_cache_size"]) * 100
        
        health_status = "healthy"
        if memory_usage_percent > 90 or cache_usage_percent > 95:
            health_status = "critical"
        elif memory_usage_percent > 75 or cache_usage_percent > 80:
            health_status = "warning"
        
        return {
            "success": True,
            "health_status": health_status,
            "metrics": {
                "memory_usage_percent": round(memory_usage_percent, 2),
                "cache_usage_percent": round(cache_usage_percent, 2),
                "hit_rate": round(stats["hit_rate"] * 100, 2),
                "total_requests": stats["hits"] + stats["misses"],
                "active_models": stats["cache_size"]
            },
            "recommendations": _get_health_recommendations(memory_usage_percent, cache_usage_percent, stats["hit_rate"])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache health: {str(e)}")


def _get_health_recommendations(memory_percent: float, cache_percent: float, hit_rate: float) -> List[str]:
    """Generate health recommendations based on metrics"""
    recommendations = []
    
    if memory_percent > 80:
        recommendations.append("Consider clearing old cache entries or increasing memory limit")
    
    if cache_percent > 85:
        recommendations.append("Cache is near capacity - consider increasing max cache size")
    
    if hit_rate < 0.5:
        recommendations.append("Low cache hit rate - review cache TTL settings or preload strategy")
    
    if memory_percent > 90 or cache_percent > 95:
        recommendations.append("URGENT: Cache resources critically low - immediate action required")
    
    if not recommendations:
        recommendations.append("Cache performance is optimal")
    
    return recommendations


# Usage example for integration with FastAPI app
"""
from fastapi import FastAPI
from app.api.cache_management import cache_router

app = FastAPI()
app.include_router(cache_router)

# Then you can access endpoints like:
# GET /api/cache/stats
# GET /api/cache/info/imitation_learning/Monza/BMW_M4_GT3
# DELETE /api/cache/invalidate/imitation_learning/Monza/BMW_M4_GT3
# DELETE /api/cache/clear
# POST /api/cache/preload?track_name=Spa&car_name=Mercedes_AMG_GT3
# GET /api/cache/health
"""

if __name__ == "__main__":
    import asyncio
    
    async def test_cache_endpoints():
        """Test cache management functionality"""
        print("Testing Cache Management Functions")
        print("=" * 40)
        
        # Test getting stats
        print("\\n1. Getting cache statistics:")
        try:
            stats = await get_cache_statistics()
            print(f"   Cache size: {stats['data']['cache_size']}")
            print(f"   Memory usage: {stats['data']['memory_usage_mb']:.2f} MB")
            print(f"   Hit rate: {stats['data']['hit_rate']:.2%}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test health check
        print("\\n2. Health check:")
        try:
            health = await cache_health_check()
            print(f"   Status: {health['health_status']}")
            print(f"   Memory usage: {health['metrics']['memory_usage_percent']:.1f}%")
            print(f"   Recommendations: {len(health['recommendations'])}")
        except Exception as e:
            print(f"   Error: {e}")
        
        print("\\nCache management tests completed!")
    
    asyncio.run(test_cache_endpoints())
