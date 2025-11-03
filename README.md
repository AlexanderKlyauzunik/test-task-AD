# Agentic RAG System

Docker-based Retrieval-Augmented Generation system with CrewAI agents, Ollama LLM, and PostgreSQL vector store.

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/AlexanderKlyauzunik/test-task-AD.git
cd test-task-AD
```

### 2. Environment Configuration

```bash
cp .env.example .env
```

### 3. Start Services

```bash
# First run or after code changes
sudo docker compose up --build -d

# Regular start
docker compose up -d

# Verify services
docker compose ps
```

## 🌐 Service URLs

- **Backend API**: http://localhost:8000 (FastAPI + API docs)
- **OpenWebUI**: http://localhost:3000 (Chat interface)
- **Phoenix**: http://localhost:6006 (Observability)
- **PostgreSQL**: http://localhost:5432 (Database with pgvector)
- **Ollama**: http://localhost:11434 (LLM hosting)

## 🤖 Setup AI Models

Pull required models from Ollama:

```bash
# Embedding model (768 dimensions, required for document indexing)
docker exec ollama ollama pull nomic-embed-text

# Generation model (required for chat responses)
docker exec ollama ollama pull qwen3:0.6b
```

## 📄 Document Upload & Indexing

The `/api/upload-and-chunk` endpoint automatically:
1. Saves original file to `data/raw_docs/`
2. Chunks document using Docling library
3. Saves chunks to `data/clear_docs/{filename}_chunks.json`
4. Indexes chunks into PostgreSQL vector database

### Upload Commands

**Supported formats:** `.pdf`, `.docx`, `.doc`, `.txt`, `.md`

```bash
# Upload with default chunk size (1000 characters = ~170 words, ~250 tokens)
curl -X POST \
  -F "document_file=@document.pdf" \
  http://localhost:8000/api/upload-and-chunk

# Upload with custom chunk size (range: 100-10000 characters)
curl -X POST \
  -F "document_file=@report.docx" \
  -F "max_chunk_size=500" \
  http://localhost:8000/api/upload-and-chunk

# Upload multiple documents
curl -X POST -F "document_file=@doc1.pdf" http://localhost:8000/api/upload-and-chunk
curl -X POST -F "document_file=@doc2.txt" -F "max_chunk_size=800" http://localhost:8000/api/upload-and-chunk
curl -X POST -F "document_file=@doc3.md" http://localhost:8000/api/upload-and-chunk
```

**Important notes:**
- Parameter name is `document_file` (not `file`)
- Parameter name is `max_chunk_size` (not `chunk_size`)
- Documents are immediately searchable after upload
- All documents share the same vector index
- Chunking uses 10% overlap for context preservation

## 🔍 Testing & Usage

### Test RAG Retrieval

```bash
# Query vector database directly
curl -X POST http://localhost:8000/api/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?"}'

# Response includes:
# - text: retrieved chunk text
# - score: relevance score (0-1)
# - metadata: {id, source, source_file, document_id, chunk_index}
# - document_id: unique chunk identifier
```

### Test Agentic Chat

Uses two CrewAI agents for enhanced quality:
1. **Researcher Agent** - Validates facts from retrieved documents
2. **Finisher Agent** - Formats response with citations [1][2][3]

```bash
# Direct API call
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain the main concepts", "model": "qwen3:0.6b"}'

# OpenWebUI format (with messages array)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:0.6b",
    "messages": [{"role": "user", "content": "What are the key points?"}]
  }'
```

## 🛠️ Development Commands

### Docker Management

```bash
# View backend logs
docker compose logs -f backend

# View Ollama logs
docker compose logs -f ollama

# Restart backend after code changes (hot reload enabled)
docker compose restart backend

# Stop all services
docker compose down

# Stop and remove volumes
docker compose down -v
```

### Database Management

```bash
# Check document count
docker exec postgres_db psql -U rag_user -d rag_db -c "SELECT COUNT(*) FROM data_rag_vectors;"

# View database statistics
docker exec postgres_db psql -U rag_user -d rag_db -c "
SELECT 
  metadata->>'source' as source,
  COUNT(*) as chunk_count,
  AVG(LENGTH(text)) as avg_length
FROM data_rag_vectors 
GROUP BY metadata->>'source';"

# Drop and rebuild index (removes ALL documents)
docker exec postgres_db psql -U rag_user -d rag_db -c "DROP TABLE IF EXISTS data_rag_vectors CASCADE;"

# Connect to database for manual queries
docker exec -it postgres_db psql -U rag_user -d rag_db
```

### Code Quality

```bash
# Check linting issues
uv run ruff check

# Auto-fix linting issues
uv run ruff check --fix
```

## 📊 Observability with Phoenix

Phoenix UI: http://localhost:6006

### Instrumented Operations

1. **`document.upload`** - File upload, chunking, indexing
   - Attributes: `filename`, `chunk_size`, `file_size`, `total_chunks`, `indexed_count`

2. **`rag.retrieve`** - Vector similarity search
   - Attributes: `query`, `limit`, `documents_found`

3. **`agents.process`** - CrewAI agent execution
   - Attributes: `model`, `query_length`, `docs_count`

4. **`rag.chat`** - Complete RAG pipeline
   - Attributes: `model`
   - Child spans: `db.retrieve` + `agents.process`

5. **LLM calls** - Ollama API interactions (auto-instrumented)

### Using Phoenix

1. Upload documents and run queries
2. Open http://localhost:6006
3. View traces with timing, metadata, and performance metrics

## 🏗️ Architecture

### Components

- **Backend (FastAPI)** - `/app/main.py` - API orchestration
- **Document Processor** - `/app/document_processor.py` - Document chunking with Docling
- **Vector Store Manager** - `/app/vector_store.py` - PostgreSQL vector operations
- **CrewAI Agents** - `/app/crew_agents.py` - Multi-agent orchestration
- **Tracing** - `/app/tracing.py` - Phoenix/OpenTelemetry observability
- **Configuration** - `/app/config.py` - Environment settings
- **Conversation Memory** - `/app/conversation_memory.py` - Chat history cache
