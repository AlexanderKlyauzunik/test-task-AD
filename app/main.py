"""
FastAPI Backend for Agentic RAG System
Foundation Stage - Ollama Proxy API for OpenWebUI integration
"""

import logging
import os
import shutil
from pathlib import Path

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import CLEAR_DOCS_DIR, OLLAMA_HOST, RAW_DOCS_DIR
from app.conversation_memory import DEFAULT_USER_ID, append_message_to_history, fetch_conversation_history  # Import cache helpers
from app.document_processor import RecursiveTextSplitter

# Initialize Phoenix observability
from app.tracing import active_trace_span
from app.vector_store import PostgresVectorStoreManager

# Setup logging
logging.basicConfig(level=logging.INFO)
app_logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Agentic RAG Backend",
    description="Backend API for RAG system with OpenWebUI integration",
    version="0.1.0"
)
document_chunker = RecursiveTextSplitter()
vector_db_manager = PostgresVectorStoreManager()
# Configure CORS for OpenWebUI frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ServiceStatusResponse(BaseModel):
    status: str
    service: str

class DocumentProcessingResult(BaseModel):
    filename: str
    status: str
    total_chunks: int
    chunk_size: int
    message: str # Added message field for better feedback


# HTTP client for Ollama communication
async def get_http_client():
    return httpx.AsyncClient(timeout=30.0)

@app.get("/health")
async def check_service_status():
    """Health check endpoint for service monitoring"""
    return ServiceStatusResponse(
        status="healthy",
        service="agentic-rag-backend"
    )

# Ollama Proxy Endpoints
@app.get("/api/tags")
async def get_available_ollama_models():
    """Fetch available models from Ollama"""
    async with await get_http_client() as client:
        try:
            response = await client.get(f"{OLLAMA_HOST}/api/tags")
            return response.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Ollama connection failed: {e!s}")

@app.post("/api/pull")
async def request_ollama_model_pull(request: Request):
    """Initiate a model pull from Ollama"""
    body = await request.json()
    async with await get_http_client() as client:
        try:
            response = await client.post(f"{OLLAMA_HOST}/api/pull", json=body)
            return response.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Ollama connection failed: {e!s}")

@app.post("/api/generate")
async def generate_ollama_response(request: Request):
    """Request a generation response from Ollama"""
    body = await request.json()
    async with await get_http_client() as client:
        try:
            response = await client.post(f"{OLLAMA_HOST}/api/generate", json=body)
            return response.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Ollama connection failed: {e!s}")

@app.post("/api/chat")
async def rag_chat_completion(request: Request):
    """Chat completion endpoint with RAG agents for OpenWebUI integration"""
    body = await request.json()
    load_dotenv()
    print("Request body:", body)

    # Force non-streaming mode for agent processing
    body["stream"] = False

    with active_trace_span("rag.chat", {"model": body.get("model", "unknown")}):
        try:
            # Detect request format: OpenWebUI uses "messages", direct API uses "prompt" or "query"
            user_input_query = None
            if "messages" in body:
                # Extract user query from messages array (OpenWebUI format)
                messages = body.get("messages", [])
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        user_input_query = msg.get("content")
                        break

                if not user_input_query:
                    raise HTTPException(status_code=400, detail="No user message found in messages")
            else:
                # Direct API format (for curl/testing)
                user_input_query = body.get("prompt", body.get("query", ""))
                if not user_input_query:
                    raise HTTPException(status_code=400, detail="No query provided")


            with active_trace_span("db.retrieve", {"query": user_input_query[:100], "limit": 3}):
                retrieved_nodes = vector_db_manager.perform_similarity_search(user_input_query, 3)

            # Format retrieved documents with metadata
            context_documents = [
            {
                "text": node.text,
                "score": node.score if hasattr(node, 'score') else 0.0,
                "metadata": node.metadata if hasattr(node, 'metadata') and node.metadata else {},
                "source": node.metadata.get('source_file', 'unknown') if hasattr(node, 'metadata') and node.metadata else 'unknown',
                "document_id": node.metadata.get('id', node.metadata.get('document_id', None)) if hasattr(node, 'metadata') and node.metadata else None
            }
            for node in retrieved_nodes
            ]

            # Process with agents (synchronous execution via asyncio.to_thread)
            import asyncio
            from datetime import datetime, timezone

            from app.crew_agents import process_with_agents

            with active_trace_span("agents.process", {
                "model": body["model"],
                "query_length": len(user_input_query),
                "docs_count": len(context_documents)
            }):
                agent_processing_output = await asyncio.to_thread(process_with_agents, body["model"], user_input_query, context_documents)

            await append_message_to_history(DEFAULT_USER_ID, "assistant", agent_processing_output.get('response', ''))
            history = await fetch_conversation_history(DEFAULT_USER_ID)
            app_logger.info(f"Conversation memory updated for user {DEFAULT_USER_ID}")

            # Format response based on request type
            if "messages" in body:
                # Return Ollama-compatible format for OpenWebUI
                return {
                    "model": body["model"],
                    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "message": {
                        "role": "assistant",
                        "content": agent_processing_output.get('response', '')
                    },
                    "done": True
                }
            else:
                # Return custom format for direct API calls
                return agent_processing_output

        except Exception as e:
            app_logger.error(f"Chat failed: {e!s}", exc_info=True)
            # In case of DB or Agent error, return a standard error response
            if "messages" in body:
                 return {
                    "model": body.get("model", "unknown"),
                    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "message": {
                        "role": "assistant",
                        "content": f"An unexpected error occurred during RAG processing: {e!s}"
                    },
                    "done": True
                }
            raise HTTPException(status_code=503, detail=f"Chat failed: {e!s}")

@app.get("/api/version")
async def get_ollama_version():
    """Get Ollama version"""
    async with await get_http_client() as client:
        try:
            response = await client.get(f"{OLLAMA_HOST}/api/version")
            return response.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Ollama connection failed: {e!s}")

@app.post("/api/rag")
async def run_rag_query(request: Request):
    """Perform RAG query using Ollama embeddings directly against the vector store"""
    body = await request.json()
    load_dotenv()

    try:
        retrieved_nodes = vector_db_manager.perform_similarity_search(body["query"])
        # Format response for generation module with metadata
        retrieval_response = [
            {
                "text": node.text,
                "score": node.score if hasattr(node, 'score') else 0.0,
                "metadata": node.metadata if hasattr(node, 'metadata') and node.metadata else {},
                "source": node.metadata.get('source_file', 'unknown') if hasattr(node, 'metadata') and node.metadata else 'unknown',
                "document_id": node.metadata.get('id', node.metadata.get('document_id', None)) if hasattr(node, 'metadata') and node.metadata else None
            }
            for node in retrieved_nodes
        ]
        return {
            "response": retrieval_response,
            "query": body["query"],
            "count": len(retrieval_response)
        }
    except Exception as e:
        app_logger.error(f"RAG query failed: {e!s}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"RAG query failed: {e!s}")


@app.get("/")
async def get_api_info():
    """Root endpoint with basic API information"""
    return {
        "message": "Agentic RAG Backend API",
        "version": "0.1.0",
        "stage": "Foundation",
        "ollama_host": OLLAMA_HOST,
        "endpoints": {
            "health": "/health",
            "ollama_proxy": "/api/*",
            "docs": "/docs"
        }
    }

@app.post("/api/upload-and-chunk", response_model=DocumentProcessingResult)
async def process_document_upload(
    document_file: UploadFile = File(...),
    max_chunk_size: int = Form(1000, ge=100, le=10000, description="Maximum chunk size in characters (default: 1000, range: 100-10000)")
):
    """
    Unified endpoint for document upload, chunking, and indexing.
    
    Uploads a document, chunks it into smaller pieces, and automatically indexes 
    the chunks into the vector database.
    
    Args:
        document_file: Document file to upload (supports .pdf, .docx, .doc, .md)
        max_chunk_size: Maximum size of each chunk in characters. 
                   Default: 1000 characters (~170 words, ~250 tokens).
                   Range: 100-10000 characters.
    
    Returns:
        DocumentProcessingResult with processing status and chunk count
    """
    allowed_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md'}
    file_extension = Path(document_file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_extension} not supported. Allowed: {allowed_extensions}"
        )

    # Ensure RAW_DOCS_DIR exists
    Path(RAW_DOCS_DIR).mkdir(parents=True, exist_ok=True)

    file_path = Path(RAW_DOCS_DIR) / document_file.filename

    try:
        with active_trace_span("document.upload", {
            "filename": document_file.filename,
            "chunk_size": max_chunk_size
        }) as span:
            # 1. Save the raw file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(document_file.file, buffer)

            file_size = file_path.stat().st_size
            if span:
                span.set_attribute("file_size", file_size)

            # 2. Process and Chunk the document
            chunks = document_chunker.process_and_save_chunks(document_file.filename, max_chunk_size)

            if span:
                span.set_attribute("total_chunks", len(chunks))

            # 3. Index chunks into vector database
            chunk_filename = f"{Path(document_file.filename).stem}_chunks.json"
            chunk_file_path = Path(CLEAR_DOCS_DIR) / chunk_filename

            indexed_count = 0
            indexing_error_message = None
            try:
                indexed_count = vector_db_manager.full_json_to_db_pipeline(
                    json_file_path=str(chunk_file_path),
                    source_name=Path(document_file.filename).stem
                )
                app_logger.info(f"Indexed {indexed_count} chunks from {chunk_filename}")
                if span:
                    span.set_attribute("indexed_count", indexed_count)
            except Exception as index_error:
                indexing_error_message = str(index_error)
                app_logger.error(f"Indexing failed (chunks saved but not indexed): {indexing_error_message}", exc_info=True)
                if span:
                    span.set_attribute("indexing_error", indexing_error_message)

        return DocumentProcessingResult(
            filename=document_file.filename,
            status="completed",
            total_chunks=len(chunks),
            chunk_size=max_chunk_size,
            message=f"Successfully processed {len(chunks)} chunks" +
                   (f", indexed {indexed_count}" if indexed_count else "") +
                   (f". Warning: Indexing failed: {indexing_error_message}" if indexing_error_message else "")
        )

    except Exception as e:
        # Clean up raw file on error
        if file_path.exists():
            file_path.unlink()
        app_logger.error(f"Document processing failed: {e!s}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {e!s}"
        )

if __name__ == "__main__":
    # Get configuration from environment variables
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", "8000"))

    # Run the application
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,  # Set to True for development
        log_level="info"
    )
