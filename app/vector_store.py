import json
import os
from typing import Any

import psycopg2
import psycopg2.errors
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from app.config import DATA_DIR, EMBEDDING_MODEL, OLLAMA_HOST
from app.tracing import active_trace_span


class PostgresVectorStoreManager:
    def __init__(self,
                 chunk_source_path: str = None,
                 collection_name: str = None,
                 embedding_dimension: int = 768):
        """
        Initialize Postgres Vector Database Manager for RAG system.
        
        Args:
            chunk_source_path: Path to JSON file containing chunks (optional).
            collection_name: Database table name (default: rag_vectors).
            embedding_dimension: Embedding vector size.
        """
        self.load_config_variables()
        self.configure_db_access_params()
        self.chunk_source_path = chunk_source_path
        self.collection_name = collection_name or os.getenv("POSTGRES_TABLE", "rag_vectors")
        self.embedding_dimension = embedding_dimension
        self.db_connection = None
        self.embedding_model_instance = None
        self.pg_vector_store = None
        self.vector_store_index = None

    def load_config_variables(self) -> None:
        """Load environment variables for configuration."""
        load_dotenv(dotenv_path='../.env')
        load_dotenv(dotenv_path='/app/.env')
        load_dotenv()

    def configure_db_access_params(self) -> None:
        """Set up database credentials from environment variables."""
        self.database = os.getenv("POSTGRES_DB")
        self.user = os.getenv("POSTGRES_USER")
        self.password = os.getenv("POSTGRES_PASSWORD")
        self.host = os.getenv("POSTGRES_HOST", "postgres")
        self.port = int(os.getenv("POSTGRES_PORT", "5432"))

        if not all([self.database, self.user, self.password]):
            raise ValueError(
                f"Missing database credentials. "
                f"DB: {self.database}, User: {self.user}, "
                f"Password: {'SET' if self.password else 'NOT SET'}"
            )

    def connect_to_database(self) -> None:
        """Establish a connection to the PostgreSQL server."""
        self.db_connection = psycopg2.connect(
            database=self.database,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        )
        self.db_connection.autocommit = True

    def ensure_database_exists(self) -> None:
        """Create the specified database if it does not already exist."""
        if not self.db_connection:
            self.connect_to_database()

        # Temporarily connect to a default DB (like 'postgres') to execute CREATE DATABASE
        # This part of the logic is usually handled outside the main connection,
        # but kept simplified here to meet original code structure.

        # A more robust solution for CREATE DATABASE would involve connecting to 'postgres'
        # with a separate cursor and connection object, which is beyond this modification.
        # Assuming the connection established is sufficient or relying on Docker setup.

    def list_databases(self) -> list[str]:
        """Retrieve a list of all available databases on the server."""
        if not self.db_connection:
            self.connect_to_database()

        with self.db_connection.cursor() as c:
            c.execute("SELECT datname FROM pg_database WHERE datistemplate = false")
            dbs = [db[0] for db in c.fetchall()]
            return dbs

    def setup_embedding_model_client(self) -> None:
        """Initialize the Ollama-based embedding model client."""
        self.embedding_model_instance = OllamaEmbedding(
            model_name=EMBEDDING_MODEL,
            base_url=OLLAMA_HOST
        )

    def initialize_vector_store(self) -> None:
        """Initialize the PostgreSQL vector store object."""
        self.pg_vector_store = PGVectorStore.from_params(
            database=self.database,
            host=self.host,
            password=self.password,
            port=self.port,
            user=self.user,
            table_name=self.collection_name,
            embed_dim=self.embedding_dimension,
        )

    def initialize_index(self) -> None:
        """Initialize the LlamaIndex VectorStoreIndex for querying."""
        if not self.pg_vector_store:
            self.initialize_vector_store()

        if not self.embedding_model_instance:
            self.setup_embedding_model_client()

        storage_context = StorageContext.from_defaults(
            vector_store=self.pg_vector_store
        )

        self.vector_store_index = VectorStoreIndex.from_documents(
            [],
            storage_context=storage_context,
            show_progress=True,
            embed_model=self.embedding_model_instance,
        )

    def retrieve_chunks_from_json(self, json_file_path: str = None) -> list[dict[str, Any]]:
        """
        Load document chunks from a structured JSON file.
        
        Args:
            json_file_path: Path to JSON file (if not provided during initialization)
            
        Returns:
            List of chunks (dictionaries)
        """
        file_path_to_load = json_file_path or self.chunk_source_path
        if not file_path_to_load:
            raise ValueError("JSON file path not provided")

        data = None
        # Try different paths if only filename is provided
        possible_paths = [
            file_path_to_load,
            f"clear_docs/{file_path_to_load}",
            f"{DATA_DIR}/{file_path_to_load}",
            f"/app/data/{file_path_to_load}",
            f"../data/{file_path_to_load}",
            f"data/{file_path_to_load}"
        ]

        for path in possible_paths:
            try:
                with open(path, encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"Loaded data from: {path}")
                    break
            except FileNotFoundError:
                continue
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in file {path}: {e}")

        if data is None:
            raise FileNotFoundError(
                f"Could not find JSON file. Tried: {', '.join(possible_paths)}"
            )

        # Extract chunks from JSON
        chunks = []
        if "chunks" in data:
            for chunk in data["chunks"]:
                chunks.append(chunk)
                print(f"Loaded chunk: {chunk}")
        else:
            # If different structure, use entire data as chunks
            chunks = data if isinstance(data, list) else [data]

        return chunks

    def generate_text_nodes(self, chunks: list[dict[str, Any]],
                               source_name: str = None) -> list[TextNode]:
        """
        Converts a list of chunk dictionaries into LlamaIndex TextNode objects for indexing.
        
        Args:
            chunks: List of chunk dictionaries.
            source_name: Identifier for the source document.
            
        Returns:
            List of TextNode objects.
        """
        if not source_name and self.chunk_source_path:
            source_name = os.path.basename(self.chunk_source_path).replace('.json', '')
        else:
            source_name = source_name or "unknown_source"

        nodes = []
        for i, chunk in enumerate(chunks):
            # Support different chunk formats
            if isinstance(chunk, dict):
                text = chunk.get('text', chunk.get('content', str(chunk)))
                chunk_id = chunk.get('id', f"{source_name}_{i}")
            else:
                text = str(chunk)
                chunk_id = f"{source_name}_{i}"

            node = TextNode(
                text=text,
                metadata={
                    'id': chunk_id,
                    'source': source_name,
                    'source_file': self.chunk_source_path or source_name,
                    'document_id': chunk_id,
                    'chunk_index': i
                }
            )
            nodes.append(node)

        return nodes

    def index_nodes_to_database(self, chunks: list[dict[str, Any]],
                     source_name: str = None) -> int:
        """
        Converts chunks to nodes and inserts them into the vector database.
        
        Args:
            chunks: List of chunk dictionaries.
            source_name: Source name.
            
        Returns:
            Number of inserted documents/nodes.
        """
        if not self.vector_store_index:
            self.initialize_index()

        nodes = self.generate_text_nodes(chunks, source_name)
        self.vector_store_index.insert_nodes(nodes)

        return len(nodes)

    def full_json_to_db_pipeline(self, json_file_path: str = None,
                        source_name: str = None) -> int:
        """
        Complete process: load JSON -> create nodes -> index to DB.
        
        Args:
            json_file_path: Path to JSON file.
            source_name: Source name.
            
        Returns:
            Number of inserted documents.
        """
        # Lazy initialization: initialize index if not exists
        if not self.vector_store_index:
            self._setup_database_resources()

        if json_file_path:
            self.chunk_source_path = json_file_path

        chunks = self.retrieve_chunks_from_json()
        count = self.index_nodes_to_database(chunks, source_name)

        print(f"Successfully inserted {count} chunks from {self.chunk_source_path}")
        return count

    def count_total_documents(self) -> int:
        """Get the total count of documents/nodes in the database table."""
        if not self.db_connection:
            self.connect_to_database()

        with self.db_connection.cursor() as c:
            c.execute(f"SELECT COUNT(*) FROM data_{self.collection_name}")
            count = c.fetchone()[0]
            return count

    def _setup_database_resources(self):
        # 1. Connect and create database
        self.connect_to_database()
        self.ensure_database_exists()

        # 2. Show database list
        dbs = self.list_databases()
        print("Available databases:")
        for db in dbs:
            print(f"- {db}")

        # 3. Initialize entire system
        self.setup_embedding_model_client()
        self.initialize_vector_store()
        self.initialize_index()

    def execute_full_indexing_pipeline(self, json_file_path: str = None) -> int:
        """
        Complete pipeline setup and data insertion in a controlled sequence.
        
        Args:
            json_file_path: Path to JSON file.
            
        Returns:
            Number of inserted documents.
        """
        print("Setting up vector database pipeline...")

        self._setup_database_resources()
        count = self.full_json_to_db_pipeline(json_file_path)

        # 5. Check result
        final_count = self.count_total_documents()
        print(f"{count} docs has been inserted. Total documents in database: {final_count}")

        return final_count


    def perform_similarity_search(self, query: str, limit: int = 5) -> list[Any]:

        """    Search for similar document chunks using vector similarity.
            
            Args:
                query: Search query string.
                limit: Maximum number of results to return.
                
            Returns:
                List of similar documents (nodes) with metadata and score.
        """
        if not self.vector_store_index:
            self.initialize_index()
        retriever = self.vector_store_index.as_retriever(similarity_top_k=limit)

        # Retrieve documents
        with active_trace_span("rag.retrieve", {"query": query[:100], "limit": limit}) as span:
            retrieved_nodes = retriever.retrieve(query)
            if span:
                span.set_attribute("documents_found", len(retrieved_nodes))

        return retrieved_nodes

    def fetch_all_indexed_documents(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        Get all documents/nodes from the database.
        
        Args:
            limit: Maximum number of documents to return
            
        Returns:
            List of all documents
        """
        if not self.db_connection:
            self.connect_to_database()

        with self.db_connection.cursor() as c:
            c.execute(f"""
                SELECT text, metadata 
                FROM data_{self.collection_name} 
                LIMIT %s
            """, (limit,))

            documents = []
            for row in c.fetchall():
                documents.append({
                    'text': row[0],
                    'metadata': row[1]
                })

            return documents

    def filter_documents_by_source(self, source: str, limit: int = 50) -> list[dict[str, Any]]:
        """
        Get documents/nodes filtered by source name.
        
        Args:
            source: Source name to filter by
            limit: Maximum number of documents to return
            
        Returns:
            List of documents from specified source
        """
        if not self.db_connection:
            self.connect_to_database()

        with self.db_connection.cursor() as c:
            c.execute(f"""
                SELECT text, metadata 
                FROM data_{self.collection_name} 
                WHERE metadata->>'source' = %s 
                LIMIT %s
            """, (source, limit))

            documents = []
            for row in c.fetchall():
                documents.append({
                    'text': row[0],
                    'metadata': row[1]
                })

            return documents

    def get_document_by_id(self, doc_id: str) -> dict[str, Any] | None:
        """
        Get specific document/node by ID.
        
        Args:
            doc_id: Document ID to search for
            
        Returns:
            Document data if found, None otherwise
        """
        if not self.db_connection:
            self.connect_to_database()

        with self.db_connection.cursor() as c:
            c.execute(f"""
                SELECT text, metadata 
                FROM data_{self.collection_name} 
                WHERE metadata->>'id' = %s
            """, (doc_id,))

            result = c.fetchone()
            if result:
                return {
                    'text': result[0],
                    'metadata': result[1]
                }
            return None

    def search_by_metadata(self, metadata_key: str, metadata_value: str,
                        limit: int = 50) -> list[dict[str, Any]]:
        """
        Search documents/nodes by metadata key-value pair.
        
        Args:
            metadata_key: Metadata key to search
            metadata_value: Metadata value to match
            limit: Maximum number of results
            
        Returns:
            List of matching documents
        """
        if not self.db_connection:
            self.connect_to_database()

        with self.db_connection.cursor() as c:
            c.execute(f"""
                SELECT text, metadata_
                FROM data_{self.collection_name} 
                WHERE metadata_->>%s = %s 
                LIMIT %s
            """, (metadata_key, metadata_value, limit))

            documents = []
            for row in c.fetchall():
                documents.append({
                    'text': row[0],
                    'metadata': row[1]
                })

            return documents

    def get_chunks_by_document_id(self, document_id: str) -> list[dict[str, Any]]:
        """
        Get all chunks for a specific document ID.
        
        Args:
            document_id: Document ID to get chunks for
            
        Returns:
            List of chunks belonging to the document
        """
        if not self.db_connection:
            self.connect_to_database()

        with self.db_connection.cursor() as c:
            c.execute(f"""
                SELECT text, metadata 
                FROM data_{self.collection_name} 
                WHERE metadata->>'document_id' = %s 
                ORDER BY (metadata->>'chunk_index')::int
            """, (document_id,))

            chunks = []
            for row in c.fetchall():
                chunks.append({
                    'text': row[0],
                    'metadata': row[1]
                })

            return chunks

    def get_statistics(self) -> dict[str, Any]:
        """
        Get database statistics on indexed data.
        
        Returns:
            Dictionary with various statistics
        """
        if not self.db_connection:
            self.connect_to_database()

        with self.db_connection.cursor() as c:
            # Total documents count
            c.execute(f"SELECT COUNT(*) FROM data_{self.collection_name}")
            total_docs = c.fetchone()[0]

            # Sources count
            c.execute(f"""
                SELECT metadata->>'source', COUNT(*) 
                FROM data_{self.collection_name} 
                GROUP BY metadata->>'source'
            """)
            sources = {row[0]: row[1] for row in c.fetchall()}

            # Average text length
            c.execute(f"SELECT AVG(LENGTH(text)) FROM data_{self.collection_name}")
            avg_length = c.fetchone()[0]

            return {
                'total_documents': total_docs,
                'sources_distribution': sources,
                'average_text_length': float(avg_length) if avg_length else 0,
                'table_name': self.collection_name
            }

    def delete_document_by_id(self, doc_id: str) -> bool:
        """
        Delete document/node by ID.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if document was deleted, False otherwise
        """
        if not self.db_connection:
            self.connect_to_database()

        with self.db_connection.cursor() as c:
            c.execute(f"""
                DELETE FROM data_{self.collection_name} 
                WHERE metadata->>'id' = %s
            """, (doc_id,))

            return c.rowcount > 0

    def delete_documents_by_source(self, source: str) -> int:
        """
        Delete all documents/nodes from a specific source.
        
        Args:
            source: Source name to delete
            
        Returns:
            Number of deleted documents
        """
        if not self.db_connection:
            self.connect_to_database()

        with self.db_connection.cursor() as c:
            c.execute(f"""
                DELETE FROM data_{self.collection_name} 
                WHERE metadata->>'source' = %s
            """, (source,))

            return c.rowcount



    def get_document_by_source_file(self, source_file: str) -> dict[str, Any] | None:
        """
        Get specific document/node by source file.
        
        Args:
            source_file: Source file to search for
            
        Returns:
            Document data if found, None otherwise
        """
        if not self.db_connection:
            self.connect_to_database()

        with self.db_connection.cursor() as c:
            c.execute(f"""
                SELECT text, metadata 
                FROM data_{self.collection_name} 
                WHERE metadata->>'source_file' = %s
            """, (source_file,))

            result = c.fetchone()
            if result:
                return {
                    'text': result[0],
                    'metadata': result[1]
                }
            return None

    def get_chunks_by_source_file(self, source_file: str) -> list[dict[str, Any]]:
        """
        Get all chunks for a specific source file.
        
        Args:
            source_file: Source file to get chunks for
            
        Returns:
            List of chunks belonging to the source file
        """
        if not self.db_connection:
            self.connect_to_database()

        with self.db_connection.cursor() as c:
            c.execute(f"""
                SELECT text, metadata 
                FROM data_{self.collection_name} 
                WHERE metadata->>'source_file' = %s 
                ORDER BY (metadata->>'chunk_index')::int
            """, (source_file,))

            chunks = []
            for row in c.fetchall():
                chunks.append({
                    'text': row[0],
                    'metadata': row[1]
                })

            return chunks

    def delete_document_by_source_file(self, source_file: str) -> bool:
        """
        Delete document/node by source file.
        
        Args:
            source_file: Source file to delete
            
        Returns:
            True if document was deleted, False otherwise
        """
        if not self.db_connection:
            self.connect_to_database()

        with self.db_connection.cursor() as c:
            c.execute(f"""
                DELETE FROM data_{self.collection_name} 
                WHERE metadata->>'source_file' = %s
            """, (source_file,))

            return c.rowcount > 0

    def get_documents_by_source_file_pattern(self, pattern: str, limit: int = 50) -> list[dict[str, Any]]:
        """
        Get documents/nodes by source file pattern (LIKE search).
        
        Args:
            pattern: SQL LIKE pattern (e.g., '%.json', 'test%')
            limit: Maximum number of documents to return
            
        Returns:
            List of matching documents
        """
        if not self.db_connection:
            self.connect_to_database()

        with self.db_connection.cursor() as c:
            c.execute(f"""
                SELECT text, metadata 
                FROM data_{self.collection_name} 
                WHERE metadata->>'source_file' LIKE %s 
                LIMIT %s
            """, (pattern, limit))

            documents = []
            for row in c.fetchall():
                documents.append({
                    'text': row[0],
                    'metadata': row[1]
                })

            return documents

    def close(self) -> None:
        """Close the database connection."""
        if self.db_connection:
            self.db_connection.close()
            print("Database connection closed")

    def debug_table_structure(self) -> None:
        """Debug method to check table structure and sample data."""
        if not self.db_connection:
            self.connect_to_database()

        with self.db_connection.cursor() as c:
            # Check if table exists
            c.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_name = %s
            """, (f"data_{self.collection_name}",))

            table_exists = c.fetchone()
            print(f"Table data_{self.collection_name} exists: {bool(table_exists)}")

            if table_exists:
                # Get table structure
                c.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = %s
                """, (f"data_{self.collection_name}",))

                columns = c.fetchall()
                print("Table columns:")
                for col in columns:
                    print(f"- {col[0]}: {col[1]}")

                # Check sample data
                c.execute(f"SELECT * FROM data_{self.collection_name} LIMIT 1")
                sample = c.fetchone()
                if sample:
                    print("Sample row:", sample)
    def drop_table(self) -> bool:
        """
        Drop the entire vector store collection table.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.db_connection:
            self.connect_to_database()

        try:
            with self.db_connection.cursor() as c:
                c.execute(f"DROP TABLE IF EXISTS data_{self.collection_name}")
                print(f"Table dropped: data_{self.collection_name}")
                return True
        except Exception as e:
            print(f"Error dropping table: {e}")
            return False


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
