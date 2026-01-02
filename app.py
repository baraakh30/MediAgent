"""
Flask API for Medical Agent + RAG Application
REST API endpoints for the medical AI assistant with caching and session history
"""
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import logging
from typing import Dict, Any, List, Optional
import traceback
from functools import wraps
import time
from datetime import datetime, timezone
import hashlib
import json
import uuid
import redis
from concurrent.futures import ThreadPoolExecutor

from config import get_settings, create_directories
from logger import setup_logging, get_logger
from agent_main import run_medical_assistant, run_medical_research
from rag_tools import get_rag_pipeline

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load configuration
config = get_settings()
app.config['SECRET_KEY'] = config.SECRET_KEY

# Setup logging
setup_logging(config.LOG_LEVEL, config.LOG_FILE)
logger = get_logger(__name__)

# Create necessary directories
create_directories()

# Redis clients
redis_cache_client: Optional[redis.Redis] = None
redis_history_client: Optional[redis.Redis] = None

# In-memory fallback storage (when Redis is not available)
memory_cache: Dict[str, Any] = {}
memory_history: Dict[str, List[Dict]] = {}

# Thread pool for batch processing
executor = ThreadPoolExecutor(max_workers=4)


def initialize_redis():
    """Initialize Redis connections"""
    global redis_cache_client, redis_history_client
    
    if not config.REDIS_ENABLED:
        logger.info("Redis is disabled, using in-memory storage")
        return
    
    try:
        logger.info("Connecting to Redis...")
        redis_cache_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            password=config.REDIS_PASSWORD,
            decode_responses=True
        )
        redis_cache_client.ping()
        
        redis_history_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.HISTORY_DB,
            password=config.REDIS_PASSWORD,
            decode_responses=True
        )
        redis_history_client.ping()
        
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}. Using in-memory storage.")
        redis_cache_client = None
        redis_history_client = None


def get_cache_key(question: str) -> str:
    """Generate cache key from question"""
    return f"cache:{hashlib.md5(question.lower().strip().encode()).hexdigest()}"


def get_from_cache(question: str) -> Optional[Dict[str, Any]]:
    """Get cached response"""
    cache_key = get_cache_key(question)
    
    # Try Redis first
    if redis_cache_client:
        try:
            cached = redis_cache_client.get(cache_key)
            if cached:
                logger.info(f"Cache hit (Redis) for question: {question[:50]}...")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis cache retrieval error: {e}")
    
    # Fallback to memory cache
    if cache_key in memory_cache:
        logger.info(f"Cache hit (memory) for question: {question[:50]}...")
        return memory_cache[cache_key]
    
    return None


def set_cache(question: str, result: Dict[str, Any]):
    """Cache response"""
    cache_key = get_cache_key(question)
    
    # Try Redis first
    if redis_cache_client:
        try:
            redis_cache_client.setex(
                cache_key,
                config.CACHE_TTL,
                json.dumps(result)
            )
            logger.info(f"Cached (Redis) response for: {question[:50]}...")
            return
        except Exception as e:
            logger.warning(f"Redis cache storage error: {e}")
    
    # Fallback to memory cache
    memory_cache[cache_key] = result
    logger.info(f"Cached (memory) response for: {question[:50]}...")


def get_session_id() -> str:
    """Get or create session ID"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']


def save_to_history(session_id: str, question: str, answer: str, sources: List = None, mode: str = 'agent'):
    """Save conversation to history"""
    message = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer,
        "sources": sources or [],
        "mode": mode
    }
    
    # Try Redis first
    if redis_history_client:
        try:
            history_key = f"history:{session_id}"
            redis_history_client.lpush(history_key, json.dumps(message))
            redis_history_client.ltrim(history_key, 0, config.MAX_HISTORY_PER_SESSION - 1)
            redis_history_client.expire(history_key, 86400 * 7)  # 7 days
            logger.info(f"Saved to history (Redis) for session: {session_id[:8]}...")
            return
        except Exception as e:
            logger.warning(f"Redis history storage error: {e}")
    
    # Fallback to memory history
    if session_id not in memory_history:
        memory_history[session_id] = []
    memory_history[session_id].insert(0, message)
    # Keep only max items
    memory_history[session_id] = memory_history[session_id][:config.MAX_HISTORY_PER_SESSION]
    logger.info(f"Saved to history (memory) for session: {session_id[:8]}...")


def get_history(session_id: str, limit: int = 10) -> List[Dict]:
    """Get conversation history"""
    # Try Redis first
    if redis_history_client:
        try:
            history_key = f"history:{session_id}"
            messages = redis_history_client.lrange(history_key, 0, limit - 1)
            return [json.loads(msg) for msg in messages]
        except Exception as e:
            logger.warning(f"Redis history retrieval error: {e}")
    
    # Fallback to memory history
    if session_id in memory_history:
        return memory_history[session_id][:limit]
    return []


def clear_session_history(session_id: str) -> bool:
    """Clear history for a session"""
    cleared = False
    
    # Try Redis first
    if redis_history_client:
        try:
            history_key = f"history:{session_id}"
            redis_history_client.delete(history_key)
            cleared = True
        except Exception as e:
            logger.warning(f"Redis history clear error: {e}")
    
    # Also clear memory history
    if session_id in memory_history:
        del memory_history[session_id]
        cleared = True
    
    return cleared


def error_handler(f):
    """Decorator for error handling"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                "error": str(e),
                "message": "An error occurred processing your request"
            }), 500
    return decorated_function


def timing_decorator(f):
    """Decorator to log execution time"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{f.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return decorated_function


@app.route('/')
def index():
    """Serve the main UI"""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "medical-agent-rag"
    }), 200


@app.route('/api/v1/agent/query', methods=['POST'])
@error_handler
@timing_decorator
def agent_query():
    """
    Query the medical agent
    
    Request body:
    {
        "question": "What are the symptoms of diabetes?"
    }
    """
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({
            "error": "Missing required field",
            "message": "Please provide a 'question' in the request body"
        }), 400
    
    question = data['question']
    
    if not question or not question.strip():
        return jsonify({
            "error": "Empty question",
            "message": "Question cannot be empty"
        }), 400
    
    logger.info(f"Agent query received: {question[:100]}...")
    
    # Get session ID
    session_id = get_session_id()
    
    # Check cache first
    cached_result = get_from_cache(f"agent:{question}")
    if cached_result:
        # Save to history
        save_to_history(session_id, question, cached_result["answer"], mode='agent')
        
        return jsonify({
            "question": question,
            "answer": cached_result["answer"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_time": 0,
            "cached": True
        }), 200
    
    start_time = time.time()
    
    # Run the medical assistant agent
    result = run_medical_assistant(question)
    
    processing_time = time.time() - start_time
    
    # Cache the result
    set_cache(f"agent:{question}", {"answer": result})
    
    # Save to history
    save_to_history(session_id, question, result, mode='agent')
    
    response = {
        "question": question,
        "answer": result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "processing_time": round(processing_time, 2),
        "cached": False
    }
    
    return jsonify(response), 200


@app.route('/api/v1/agent/research', methods=['POST'])
@error_handler
@timing_decorator
def agent_research():
    """
    Research a medical topic using the agent
    
    Request body:
    {
        "topic": "Hypertension"
    }
    """
    data = request.get_json()
    
    if not data or 'topic' not in data:
        return jsonify({
            "error": "Missing required field",
            "message": "Please provide a 'topic' in the request body"
        }), 400
    
    topic = data['topic']
    
    if not topic or not topic.strip():
        return jsonify({
            "error": "Empty topic",
            "message": "Topic cannot be empty"
        }), 400
    
    logger.info(f"Research query received: {topic}")
    
    # Get session ID
    session_id = get_session_id()
    
    # Check cache first
    cached_result = get_from_cache(f"research:{topic}")
    if cached_result:
        save_to_history(session_id, topic, cached_result["report"], mode='research')
        
        return jsonify({
            "topic": topic,
            "report": cached_result["report"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_time": 0,
            "cached": True
        }), 200
    
    start_time = time.time()
    
    # Run the medical research
    result = run_medical_research(topic)
    
    processing_time = time.time() - start_time
    
    # Cache the result
    set_cache(f"research:{topic}", {"report": result})
    
    # Save to history
    save_to_history(session_id, topic, result, mode='research')
    
    response = {
        "topic": topic,
        "report": result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "processing_time": round(processing_time, 2),
        "cached": False
    }
    
    return jsonify(response), 200


@app.route('/api/v1/rag/search', methods=['POST'])
@error_handler
@timing_decorator
def rag_search():
    """
    Direct RAG search without agent
    
    Request body:
    {
        "query": "diabetes symptoms",
        "top_k": 5
    }
    """
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({
            "error": "Missing required field"
        }), 400
    
    query = data['query']
    top_k = data.get('top_k', 5)
    
    if not query or not query.strip():
        return jsonify({
            "error": "Empty query"
        }), 400
    
    logger.info(f"RAG search query: {query[:100]}...")
    
    # Get RAG pipeline and search
    rag = get_rag_pipeline()
    results = rag.similarity_search(query, k=min(top_k, 10))
    
    response = {
        "query": query,
        "results": results,
        "count": len(results),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    return jsonify(response), 200


@app.route('/api/v1/rag/query', methods=['POST'])
@error_handler
@timing_decorator
def rag_query():
    """
    Direct RAG query with answer generation
    
    Request body:
    {
        "question": "What causes diabetes?"
    }
    """
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({
            "error": "Missing required field"
        }), 400
    
    question = data['question']
    
    if not question or not question.strip():
        return jsonify({
            "error": "Empty question"
        }), 400
    
    logger.info(f"RAG query: {question[:100]}...")
    
    start_time = time.time()
    
    # Get RAG pipeline and query
    rag = get_rag_pipeline()
    result = rag.query(question)
    
    processing_time = time.time() - start_time
    
    response = {
        "question": result["question"],
        "answer": result["answer"],
        "sources": result.get("source_documents", []),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "processing_time": round(processing_time, 2)
    }
    
    return jsonify(response), 200


@app.route('/api/v1/stats', methods=['GET'])
@error_handler
def get_stats():
    """Get system statistics"""
    cache_enabled = redis_cache_client is not None
    history_enabled = redis_history_client is not None
    
    stats = {
        "service": "medical-agent-rag",
        "agent_model": config.AGENT_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "vector_store_type": config.VECTOR_STORE_TYPE,
        "top_k_results": config.TOP_K_RESULTS,
        "cache_enabled": cache_enabled,
        "history_enabled": history_enabled,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    return jsonify(stats), 200


@app.route('/api/v1/history', methods=['GET'])
@error_handler
def get_conversation_history():
    """
    Get conversation history for current session
    
    Query params:
        limit: Number of messages to retrieve (default: 10)
    """
    session_id = get_session_id()
    limit = request.args.get('limit', 10, type=int)
    
    if limit < 1 or limit > 100:
        return jsonify({
            "error": "Invalid limit",
            "message": "Limit must be between 1 and 100"
        }), 400
    
    history = get_history(session_id, limit)
    
    return jsonify({
        "session_id": session_id,
        "messages": history,
        "count": len(history),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }), 200


@app.route('/api/v1/history', methods=['DELETE'])
@error_handler
def clear_history():
    """Clear conversation history for current session"""
    session_id = get_session_id()
    
    cleared = clear_session_history(session_id)
    
    if cleared:
        logger.info(f"Cleared history for session: {session_id[:8]}...")
        return jsonify({
            "message": "History cleared successfully",
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 200
    else:
        return jsonify({
            "message": "No history to clear",
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 200


@app.route('/api/v1/cache/clear', methods=['POST'])
@error_handler
def clear_cache():
    """Clear all cached responses"""
    global memory_cache
    count = 0
    
    # Clear Redis cache if available
    if redis_cache_client:
        try:
            keys = redis_cache_client.keys("cache:*")
            if keys:
                count = redis_cache_client.delete(*keys)
        except Exception as e:
            logger.warning(f"Error clearing Redis cache: {e}")
    
    # Clear memory cache
    memory_count = len(memory_cache)
    memory_cache = {}
    count += memory_count
    
    logger.info(f"Cleared {count} cache entries")
    
    return jsonify({
        "message": "Cache cleared successfully",
        "keys_deleted": count,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }), 200


@app.route('/api/v1/cache/stats', methods=['GET'])
@error_handler
def cache_stats():
    """Get cache statistics"""
    redis_count = 0
    memory_count = len(memory_cache)
    
    if redis_cache_client:
        try:
            keys = redis_cache_client.keys("cache:*")
            redis_count = len(keys)
        except Exception as e:
            logger.warning(f"Error getting Redis cache stats: {e}")
    
    return jsonify({
        "cache_enabled": True,
        "cached_items": redis_count + memory_count,
        "redis_items": redis_count,
        "memory_items": memory_count,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }), 200


def process_single_query(question: str, mode: str = 'agent') -> Dict[str, Any]:
    """Process a single query (used for batch processing)"""
    try:
        cache_key = f"{mode}:{question}"
        cached_result = get_from_cache(cache_key)
        
        if cached_result:
            return {
                "question": question,
                "answer": cached_result.get("answer") or cached_result.get("report", ""),
                "cached": True,
                "processing_time": 0
            }
        
        start_time = time.time()
        
        if mode == 'agent':
            result = run_medical_assistant(question)
        elif mode == 'research':
            result = run_medical_research(question)
        else:
            rag = get_rag_pipeline()
            rag_result = rag.query(question)
            result = rag_result["answer"]
        
        processing_time = time.time() - start_time
        
        # Cache result
        if mode == 'research':
            set_cache(cache_key, {"report": result})
        else:
            set_cache(cache_key, {"answer": result})
        
        return {
            "question": question,
            "answer": result,
            "cached": False,
            "processing_time": round(processing_time, 2)
        }
    except Exception as e:
        logger.error(f"Error processing query '{question}': {e}")
        return {
            "question": question,
            "error": str(e),
            "processing_time": 0
        }


@app.route('/api/v1/batch', methods=['POST'])
@error_handler
def batch_process():
    """
    Process multiple questions in batch
    
    Request body:
    {
        "questions": ["question 1", "question 2", ...],
        "mode": "agent"  (optional: agent, research, rag)
    }
    """
    data = request.get_json()
    
    if not data or 'questions' not in data:
        return jsonify({
            "error": "Missing required field",
            "message": "Please provide 'questions' array in the request body"
        }), 400
    
    questions = data['questions']
    mode = data.get('mode', 'agent')
    
    if not isinstance(questions, list):
        return jsonify({
            "error": "Invalid format",
            "message": "'questions' must be an array"
        }), 400
    
    if len(questions) == 0:
        return jsonify({
            "error": "Empty questions array"
        }), 400
    
    if len(questions) > 20:
        return jsonify({
            "error": "Too many questions",
            "message": "Maximum 20 questions per batch"
        }), 400
    
    logger.info(f"Processing batch of {len(questions)} questions in {mode} mode")
    
    start_time = time.time()
    
    # Process questions in parallel
    results = list(executor.map(
        lambda q: process_single_query(q, mode),
        questions
    ))
    
    total_time = time.time() - start_time
    
    successful = sum(1 for r in results if 'error' not in r)
    failed = len(results) - successful
    
    response = {
        "results": results,
        "total": len(results),
        "successful": successful,
        "failed": failed,
        "total_time": round(total_time, 2),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    logger.info(f"Batch processed: {successful} successful, {failed} failed in {total_time:.2f}s")
    
    return jsonify(response), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Not found",
        "message": "The requested endpoint does not exist"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500


if __name__ == '__main__':
    logger.info("Starting Medical Agent + RAG API...")
    
    # Initialize Redis
    initialize_redis()
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=config.API_PORT,
        debug=(config.FLASK_ENV == 'development')
    )
