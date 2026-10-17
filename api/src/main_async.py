"""
异步模式API - FastAPI接收请求并转发给Kafka
适合：批量处理、削峰填谷、不需要立即返回结果的场景
"""

import os
import json
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis.asyncio as redis
from confluent_kafka import Producer, KafkaError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Async Text Classification API",
    description="FastAPI + Kafka异步处理模式",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局状态
class AppState:
    redis_client: Optional[redis.Redis] = None
    kafka_producer: Optional[Producer] = None
    result_topic: str = "classification-results"

state = AppState()

# ============ Pydantic模型 ============

class ClassificationRequest(BaseModel):
    """分类请求"""
    text: str = Field(..., min_length=1, max_length=10000)
    callback_url: Optional[str] = None  # 可选的回调地址
    
class AsyncResponse(BaseModel):
    """异步提交响应"""
    request_id: str
    status: str  # "accepted", "processing", "completed", "failed"
    message: str
    estimated_time_ms: int
    submitted_at: str

class ResultQueryResponse(BaseModel):
    """查询结果响应"""
    request_id: str
    status: str
    result: Optional[Dict] = None
    submitted_at: str
    completed_at: Optional[str] = None

# ============ Kafka相关 ============

def init_kafka():
    """初始化Kafka生产者"""
    config = {
        "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        "batch.num.messages": 1000,
        "linger.ms": 10,
        "compression.type": "lz4",
    }
    state.kafka_producer = Producer(config)
    logger.info("Kafka producer initialized")

def delivery_callback(err, msg):
    """消息投递回调"""
    if err:
        logger.error(f"Message delivery failed: {err}")
    else:
        logger.debug(f"Message delivered to {msg.topic()}")

def send_to_kafka(request_id: str, text: str, callback_url: Optional[str] = None):
    """发送消息到Kafka"""
    message = {
        "request_id": request_id,
        "text": text,
        "callback_url": callback_url,
        "submitted_at": datetime.utcnow().isoformat(),
        "retry_count": 0
    }
    
    state.kafka_producer.produce(
        topic="text-classification-requests",
        key=request_id.encode("utf-8"),
        value=json.dumps(message).encode("utf-8"),
        callback=delivery_callback
    )
    state.kafka_producer.poll(0)  # 触发发送

# ============ Redis相关 ============

async def init_redis():
    """初始化Redis"""
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    state.redis_client = redis.Redis(
        host=redis_host,
        port=redis_port,
        decode_responses=True
    )

# ============ API端点 ============

@app.on_event("startup")
async def startup():
    await init_redis()
    init_kafka()

@app.post("/classify/async", response_model=AsyncResponse)
async def classify_async(request: ClassificationRequest):
    """
    异步提交分类任务
    
    流程：
    1. 生成唯一请求ID
    2. 将请求写入Redis（状态：accepted）
    3. 发送消息到Kafka
    4. 立即返回请求ID给客户端
    
    客户端需要用返回的request_id轮询 /result/{request_id} 获取结果
    """
    request_id = str(uuid.uuid4())
    submitted_at = datetime.utcnow().isoformat()
    
    # 1. 先存入Redis，标记为已接收
    task_info = {
        "request_id": request_id,
        "text": request.text[:100] + "..." if len(request.text) > 100 else request.text,
        "status": "accepted",
        "submitted_at": submitted_at,
        "callback_url": request.callback_url
    }
    
    await state.redis_client.setex(
        f"task:{request_id}",
        3600,  # 1小时过期
        json.dumps(task_info)
    )
    
    # 2. 发送到Kafka队列
    try:
        send_to_kafka(request_id, request.text, request.callback_url)
        
        # 更新状态为处理中
        task_info["status"] = "processing"
        await state.redis_client.setex(
            f"task:{request_id}",
            3600,
            json.dumps(task_info)
        )
        
        return AsyncResponse(
            request_id=request_id,
            status="accepted",
            message="Task submitted successfully. Use /result/{request_id} to query result.",
            estimated_time_ms=200,  # 预估处理时间
            submitted_at=submitted_at
        )
        
    except Exception as e:
        logger.error(f"Failed to send to Kafka: {e}")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")

@app.get("/result/{request_id}", response_model=ResultQueryResponse)
async def query_result(request_id: str):
    """
    查询异步任务的结果
    
    客户端轮询这个接口获取处理结果
    """
    # 从Redis查询任务状态
    task_data = await state.redis_client.get(f"task:{request_id}")
    
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found or expired")
    
    task = json.loads(task_data)
    
    # 如果已完成，尝试获取详细结果
    result = None
    completed_at = None
    
    if task["status"] == "completed":
        result_data = await state.redis_client.get(f"result:{request_id}")
        if result_data:
            result = json.loads(result_data)
            completed_at = result.get("completed_at")
    
    return ResultQueryResponse(
        request_id=request_id,
        status=task["status"],
        result=result,
        submitted_at=task["submitted_at"],
        completed_at=completed_at
    )

@app.post("/classify/batch-async")
async def classify_batch_async(requests: list[ClassificationRequest]):
    """
    批量异步提交
    
    适合一次性提交大量文本（如1000条）
    返回批量request_id列表
    """
    request_ids = []
    
    for req in requests:
        request_id = str(uuid.uuid4())
        request_ids.append(request_id)
        
        # 批量发送到Kafka
        send_to_kafka(request_id, req.text, req.callback_url)
    
    # 触发批量发送
    state.kafka_producer.flush()
    
    return {
        "batch_id": str(uuid.uuid4()),
        "request_ids": request_ids,
        "total": len(request_ids),
        "status": "accepted",
        "query_url": "/result/batch/{batch_id}"
    }

# ============ Consumer示例（需要另外运行） ============
"""
在这种模式下，Consumer不再是调API，而是自己直接处理：

流程：
1. Consumer从Kafka拉取消息
2. Consumer直接加载BERT模型进行推理
3. Consumer将结果写入Redis（供API查询）
4. 如果有callback_url，Consumer回调通知客户端

好处：
- 完全解耦，API只负责收和查
- Consumer可以独立扩展（和API解耦）
- 适合超高并发场景（削峰填谷）
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
