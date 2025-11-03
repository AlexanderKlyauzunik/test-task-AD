import os

"""
Global Configuration Settings
Defines model names, service hosts, and data directory paths.
All language models are configured to load from the local Ollama container.
"""

EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "qwen3:0.6b"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

# Agent configuration parameters
AGENT_TEMPERATURE = 0.7
AGENT_MAX_TOKENS = 500

# Data directories configuration
DATA_DIR = os.getenv("DATA_DIR", "/app/data")

# Test files configuration (used for internal testing)
DATA_QUERIES_FILE = os.getenv("DATA_QUERIES_FILE", "queries10_corrected_double.json")

# Document directories (inside DATA_DIR)
RAW_DOCS_DIR = os.path.join(DATA_DIR, "raw_docs")
CLEAR_DOCS_DIR = os.path.join(DATA_DIR, "clear_docs")
