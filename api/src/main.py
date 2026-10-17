"""
FastAPI Backend for Real-Time Text Classification
Features Redis caching and Prometheus metrics
"""

import os
import sys
import json
import hashlib
import logging
import time
from typing import List, Optional, Dict
from datetime import datetime
import asyncio

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import redis.asyncio as redis
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
import uvicorn

# Add model path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../model/src"))
from bert_classifier import BERTTextClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    "classification_predictions_total",
    "Total number of classification predictions",
    ["status", "cached"]
)
PREDICTION_LATENCY = Histogram(
    "classification_prediction_duration_seconds",
    "Prediction latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)
CACHE_HIT_COUNTER = Counter(
    "classification_cache_hits_total",
    "Total number of cache hits"
)
CACHE_MISS_COUNTER = Counter(
    "classification_cache_misses_total",
    "Total number of cache misses"
)
ACTIVE_REQUESTS = Gauge(
    "classification_active_requests",
    "Number of active classification requests"
)

# Pydantic models
class ClassificationRequest(BaseModel):
    """Request model for single text classification."""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to classify")
    
class BatchClassificationRequest(BaseModel):
    """Request model for batch classification."""
    texts: List[str] = Field(..., min_length=1, max_length=100, description="List of texts to classify")
    
class ClassificationResponse(BaseModel):
    """Response model for classification."""
    predicted_class: str = Field(..., description="Predicted class label")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    probabilities: Dict[str, float] = Field(..., description="Probability distribution")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    cached: bool = Field(default=False, description="Whether result was cached")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    cache_connected: bool
    timestamp: str

class StatsResponse(BaseModel):
    """API statistics response."""
    total_predictions: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    avg_latency_ms: float
    model_info: Dict

# Initialize FastAPI app
app = FastAPI(
    title="Real-Time Text Classification API",
    description="BERT-based text classification with Redis caching and Prometheus metrics",
    version="1.0.0"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Initialize Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

# Global state
class AppState:
    model: Optional[BERTTextClassifier] = None
    redis_client: Optional[redis.Redis] = None
    model_path: str = os.getenv("MODEL_PATH", "./models/bert-classifier")
    cache_ttl: int = int(os.getenv("REDIS_CACHE_TTL", "3600"))
    request_count: int = 0
    total_latency: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

state = AppState()

def get_cache_key(text: str) -> str:
    """Generate cache key from text."""
    text_hash = hashlib.md5(text.encode()).hexdigest()
    return f"classification:{text_hash}"

async def init_redis():
    """Initialize Redis connection."""
    try:
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_db = int(os.getenv("REDIS_DB", "0"))
        
        state.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True,
            health_check_interval=30
        )
        
        # Test connection
        await state.redis_client.ping()
        logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
        return True
        
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        state.redis_client = None
        return False

def load_model():
    """Load the BERT classification model."""
    try:
        logger.info(f"Loading model from {state.model_path}")
        
        if os.path.exists(state.model_path):
            state.model = BERTTextClassifier.load(state.model_path)
        else:
            logger.warning("Model not found, initializing new model")
            state.model = BERTTextClassifier()
            state.model.initialize()
        
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting up API server...")
    
    # Load model
    model_loaded = load_model()
    if not model_loaded:
        logger.error("Failed to load model, API may not function correctly")
    
    # Initialize Redis
    cache_connected = await init_redis()
    if not cache_connected:
        logger.warning("Redis not available, caching disabled")
    
    logger.info("API server started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API server...")
    if state.redis_client:
        await state.redis_client.close()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    cache_connected = False
    if state.redis_client:
        try:
            cache_connected = await state.redis_client.ping()
        except:
            cache_connected = False
    
    return HealthResponse(
        status="healthy" if state.model else "degraded",
        model_loaded=state.model is not None,
        cache_connected=cache_connected,
        timestamp=datetime.utcnow().isoformat()
    )

@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get API statistics."""
    total_cache = state.cache_hits + state.cache_misses
    cache_hit_rate = state.cache_hits / total_cache if total_cache > 0 else 0
    avg_latency = (state.total_latency / state.request_count * 1000) if state.request_count > 0 else 0
    
    return StatsResponse(
        total_predictions=state.request_count,
        cache_hits=state.cache_hits,
        cache_misses=state.cache_misses,
        cache_hit_rate=cache_hit_rate,
        avg_latency_ms=avg_latency,
        model_info={
            "model_name": state.model.model_name if state.model else "unknown",
            "num_labels": state.model.num_labels if state.model else 0,
            "max_length": state.model.max_length if state.model else 0
        }
    )

@app.post("/predict", response_model=ClassificationResponse)
async def predict(request: ClassificationRequest):
    """
    Classify a single text.
    
    - Checks Redis cache before inference; latency impact requires measurement
    - Falls back to model inference if cache miss
    - Tracks metrics for Prometheus monitoring
    """
    start_time = time.perf_counter()
    ACTIVE_REQUESTS.inc()
    
    try:
        text = request.text.strip()
        cache_key = get_cache_key(text)
        
        # Check cache first
        if state.redis_client:
            try:
                cached_result = await state.redis_client.get(cache_key)
                if cached_result:
                    result = json.loads(cached_result)
                    result["cached"] = True
                    CACHE_HIT_COUNTER.inc()
                    state.cache_hits += 1
                    PREDICTION_COUNTER.labels(status="success", cached="true").inc()
                    
                    processing_time = (time.perf_counter() - start_time) * 1000
                    PREDICTION_LATENCY.observe(processing_time / 1000)
                    result["processing_time_ms"] = processing_time
                    
                    return ClassificationResponse(**result)
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        
        # Cache miss - run inference
        CACHE_MISS_COUNTER.inc()
        state.cache_misses += 1
        
        if state.model is None:
            raise HTTPException(status_code=503, detail="Model not available")
        
        # Perform inference
        inference_start = time.perf_counter()
        prediction = state.model.predict_single(text)
        inference_time = (time.perf_counter() - inference_start) * 1000
        
        # Build response
        response = ClassificationResponse(
            predicted_class=prediction["predicted_class"],
            confidence=prediction["confidence"],
            probabilities=prediction["probabilities"],
            processing_time_ms=inference_time,
            cached=False
        )
        
        # Cache the result
        if state.redis_client:
            try:
                cache_data = response.dict(exclude={"cached", "timestamp"})
                await state.redis_client.setex(
                    cache_key,
                    state.cache_ttl,
                    json.dumps(cache_data)
                )
            except Exception as e:
                logger.warning(f"Cache write error: {e}")
        
        # Update metrics
        PREDICTION_COUNTER.labels(status="success", cached="false").inc()
        total_time = (time.perf_counter() - start_time) * 1000
        PREDICTION_LATENCY.observe(total_time / 1000)
        
        state.request_count += 1
        state.total_latency += total_time / 1000
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        PREDICTION_COUNTER.labels(status="error", cached="false").inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ACTIVE_REQUESTS.dec()

@app.post("/predict/batch", response_model=List[ClassificationResponse])
async def predict_batch(request: BatchClassificationRequest):
    """
    Classify multiple texts in a batch.
    
    More efficient than individual requests for high-volume scenarios.
    """
    start_time = time.perf_counter()
    ACTIVE_REQUESTS.inc()
    
    try:
        if state.model is None:
            raise HTTPException(status_code=503, detail="Model not available")
        
        texts = [t.strip() for t in request.texts]
        
        # Check cache for each text
        cache_keys = [get_cache_key(t) for t in texts]
        cached_results = {}
        
        if state.redis_client:
            try:
                cached_values = await state.redis_client.mget(*cache_keys)
                for i, (text, cached) in enumerate(zip(texts, cached_values)):
                    if cached:
                        cached_results[i] = json.loads(cached)
                        CACHE_HIT_COUNTER.inc()
                        state.cache_hits += 1
            except Exception as e:
                logger.warning(f"Cache batch read error: {e}")
        
        # Get texts that need inference
        texts_to_infer = [texts[i] for i in range(len(texts)) if i not in cached_results]
        
        # Run batch inference
        if texts_to_infer:
            batch_predictions = state.model.predict(texts_to_infer, batch_size=len(texts_to_infer))
            
            # Merge results
            results = []
            infer_idx = 0
            for i in range(len(texts)):
                if i in cached_results:
                    cached = cached_results[i]
                    cached["cached"] = True
                    results.append(ClassificationResponse(**cached))
                else:
                    pred = batch_predictions[infer_idx]
                    infer_idx += 1
                    
                    response = ClassificationResponse(
                        predicted_class=pred["predicted_class"],
                        confidence=pred["confidence"],
                        probabilities=pred["probabilities"],
                        processing_time_ms=0,
                        cached=False
                    )
                    results.append(response)
                    
                    # Cache individual results
                    if state.redis_client:
                        try:
                            cache_data = response.dict(exclude={"cached", "timestamp"})
                            await state.redis_client.setex(
                                cache_keys[i],
                                state.cache_ttl,
                                json.dumps(cache_data)
                            )
                        except Exception as e:
                            logger.warning(f"Cache write error: {e}")
                    
                    CACHE_MISS_COUNTER.inc()
                    state.cache_misses += 1
        else:
            # All results from cache
            results = [ClassificationResponse(**cached_results[i], cached=True) 
                      for i in range(len(texts))]
        
        # Update metrics
        total_time = (time.perf_counter() - start_time) * 1000
        PREDICTION_LATENCY.observe(total_time / 1000)
        state.request_count += len(texts)
        state.total_latency += total_time / 1000
        
        PREDICTION_COUNTER.labels(status="success", cached="mixed").inc(len(texts))
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        PREDICTION_COUNTER.labels(status="error", cached="false").inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ACTIVE_REQUESTS.dec()

@app.delete("/cache")
async def clear_cache():
    """Clear the classification cache (admin only)."""
    if state.redis_client:
        try:
            # Find and delete all classification keys
            pattern = "classification:*"
            cursor = 0
            deleted = 0
            
            while True:
                cursor, keys = await state.redis_client.scan(cursor, match=pattern, count=1000)
                if keys:
                    await state.redis_client.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break
            
            return {"status": "success", "keys_deleted": deleted}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Cache clear failed: {e}")
    else:
        raise HTTPException(status_code=503, detail="Cache not available")

def main():
    """Run the API server."""
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("API_WORKERS", "1"))
    
    logger.info(f"Starting API server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()
