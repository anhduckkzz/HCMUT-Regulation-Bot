"""Milvus vector database integration for HCMUT regulation indexing.

This module provides functionality to:
1. Connect to Milvus database running on Docker
2. Validate embedding dimensions match between Gemini embeddings and Milvus collection
3. Index crawled PDF chunks into Milvus
4. Query regulations from Milvus
"""

import argparse
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set

import google.generativeai as genai
from dotenv import load_dotenv
from pymilvus import MilvusException, Collection, connections, utility

from cache_config import CacheConfig
from document_processor import DocumentProcessor, process_documents_for_vector_store

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MilvusVectorStore:
    """Embed processed regulation chunks and persist them in Milvus database."""

    # Gemini embedding model (models/gemini-embedding-001) produces 3072-dimensional vectors
    EMBEDDING_DIMENSION = 3072
    COLLECTION_NAME = "hcmut_regulations"
    PARTITION_NAME = "regulations_partition"

    def __init__(
        self,
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
        vector_db_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        crawled_cache_file: Optional[str] = None,
    ) -> None:
        """Initialize Milvus Vector Store.

        Args:
            milvus_host: Milvus server host (env: MILVUS_HOST, default: localhost)
            milvus_port: Milvus server port (env: MILVUS_PORT, default: 19530)
            vector_db_path: Optional path to vector DB cache (env: VECTOR_DB_PATH)
            cache_dir: Optional base cache directory (env: CACHE_DIR)
            crawled_cache_file: Optional path to crawled cache file (env: CRAWLED_CACHE_FILE)
        """
        # Get configuration from environment or use defaults
        self.milvus_host = os.getenv("MILVUS_HOST", milvus_host)
        self.milvus_port = int(os.getenv("MILVUS_PORT", milvus_port))

        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("Missing GEMINI_API_KEY in .env")

        genai.configure(api_key=self.gemini_api_key)

        self.embedding_model = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
        self.embedding_task_type = os.getenv("EMBEDDING_TASK_TYPE", "retrieval_document")
        self.embedding_retry_count = int(os.getenv("EMBEDDING_RETRY_COUNT", "3"))
        self.embedding_retry_delay = float(os.getenv("EMBEDDING_RETRY_DELAY", "1.5"))
        self.verbose_logs = os.getenv("VERBOSE_LOGS", "true").lower() == "true"

        # Initialize cache configuration
        self.cache_config = CacheConfig(
            cache_dir=cache_dir,
            crawled_cache_file=crawled_cache_file,
            vector_db_path=vector_db_path,
        )
        self.cache_config.ensure_directories_exist()

        self.upsert_batch_size = int(os.getenv("UPSERT_BATCH_SIZE", "32"))

        # Connect to Milvus
        self._connect_to_milvus()

        # Verify and create collection if needed
        self._initialize_collection()

        self._log(f"Initialized Milvus connection at {self.milvus_host}:{self.milvus_port}")

    def _log(self, message: str) -> None:
        """Log message if verbose logging is enabled."""
        if self.verbose_logs:
            logger.info(f"[milvus_store] {message}")

    def _connect_to_milvus(self) -> None:
        """Establish connection to Milvus server."""
        try:
            connections.connect(
                alias="default",
                host=self.milvus_host,
                port=self.milvus_port,
                timeout=10,
            )
            self._log(f"Connected to Milvus at {self.milvus_host}:{self.milvus_port}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to Milvus at {self.milvus_host}:{self.milvus_port}: {e}"
            )

    def _get_embedding_dimension_from_gemini(self) -> int:
        """Get actual embedding dimension from Gemini API.

        This validates that the embedding model produces the expected dimension (768).

        Returns:
            Embedding dimension (should be 768)

        Raises:
            RuntimeError: If unable to get embedding or dimension mismatch
        """
        try:
            self._log("Validating Gemini embedding dimension...")
            test_text = "Test embedding validation for HCMUT regulations"
            response = genai.embed_content(
                model=self.embedding_model,
                content=test_text,
                task_type=self.embedding_task_type,
            )
            embedding = response.get("embedding", [])
            dimension = len(embedding)

            if dimension != self.EMBEDDING_DIMENSION:
                raise RuntimeError(
                    f"Embedding dimension mismatch! Expected {self.EMBEDDING_DIMENSION}, "
                    f"but got {dimension} from Gemini API"
                )

            self._log(f"✓ Gemini embedding dimension validated: {dimension}")
            return dimension
        except Exception as e:
            raise RuntimeError(f"Failed to validate Gemini embedding dimension: {e}")

    def _initialize_collection(self) -> None:
        """Initialize Milvus collection with proper schema.

        Creates collection if it doesn't exist, validating that the embedding
        dimension matches Gemini's output (768).
        """
        from pymilvus import (
            FieldSchema,
            CollectionSchema,
            DataType,
        )

        # First, validate Gemini embedding dimension
        actual_dimension = self._get_embedding_dimension_from_gemini()

        # Check if collection already exists
        if utility.has_collection(self.COLLECTION_NAME):
            self._log(f"Collection '{self.COLLECTION_NAME}' already exists. Loading...")
            collection = Collection(self.COLLECTION_NAME)
            collection.load()

            # Verify schema matches expected dimension
            schema = collection.schema
            for field in schema.fields:
                if field.name == "embedding":
                    field_dim = field.params.get("dim", 0)
                    if field_dim != self.EMBEDDING_DIMENSION:
                        raise RuntimeError(
                            f"Collection embedding dimension mismatch! "
                            f"Expected {self.EMBEDDING_DIMENSION}, but collection has {field_dim}"
                        )
            self._log(f"✓ Collection schema verified: embedding dimension = {actual_dimension}")
        else:
            # Create new collection with schema
            self._log(f"Creating new collection '{self.COLLECTION_NAME}' with dimension {actual_dimension}...")

            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=actual_dimension),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="source_url", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="doc_title", dtype=DataType.VARCHAR, max_length=512),
            ]

            schema = CollectionSchema(fields=fields, description="HCMUT Regulations Vector Store")
            collection = Collection(name=self.COLLECTION_NAME, schema=schema)

            # Create index on embedding field
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024},
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            self._log(f"✓ Collection created with IVF_FLAT index")

            # Load collection into memory
            collection.load()
            self._log(f"✓ Collection '{self.COLLECTION_NAME}' loaded")

    def _get_embedding(self, text: str) -> List[float]:
        """Create embedding vector using Gemini API with retry logic.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (768-dimensional for Gemini)

        Raises:
            RuntimeError: If embedding fails after retries
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, self.embedding_retry_count + 1):
            try:
                response = genai.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type=self.embedding_task_type,
                )
                embedding = response.get("embedding", [])
                if not embedding:
                    raise ValueError("Gemini returned empty embedding.")
                return embedding
            except Exception as exc:
                last_error = exc
                self._log(f"Embedding attempt {attempt}/{self.embedding_retry_count} failed: {exc}")
                if attempt < self.embedding_retry_count:
                    time.sleep(self.embedding_retry_delay)

        raise RuntimeError(f"Embedding failed after retries: {last_error}")

    def _get_indexed_ids(self) -> Set[str]:
        """Get all IDs currently indexed in the collection.

        Returns:
            Set of indexed record IDs
        """
        try:
            collection = Collection(self.COLLECTION_NAME)
            # Query all IDs with minimal fields
            results = collection.query(
                expr="id != ''",
                output_fields=["id"],
            )
            ids = [result["id"] for result in results]
            return set(ids) if ids else set()
        except Exception as e:
            self._log(f"Error fetching indexed IDs: {e}")
            return set()

    def _upsert_batch(self, records: List[Dict[str, Any]]) -> int:
        """Upsert a batch of records into Milvus.

        Args:
            records: List of record dictionaries with id, text, metadata

        Returns:
            Number of records upserted

        Raises:
            RuntimeError: If upsert fails
        """
        if not records:
            return 0

        first_id = records[0]["id"]
        self._log(f"Embedding and upserting batch size={len(records)} (first id: {first_id})")

        ids = [record["id"] for record in records]
        documents = [record["text"] for record in records]
        metadatas = [record.get("metadata", {}) for record in records]

        # Generate embeddings for all documents in batch
        embeddings = []
        for i, text in enumerate(documents):
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
            if (i + 1) % 5 == 0:
                self._log(f"  Generated {i + 1}/{len(documents)} embeddings...")

        # Prepare data for Milvus
        data = [
            ids,
            embeddings,
            documents,
            [m.get("source_url", "") for m in metadatas],
            [m.get("chunk_id", "") for m in metadatas],
            [m.get("doc_title", "") for m in metadatas],
        ]

        try:
            collection = Collection(self.COLLECTION_NAME)
            collection.upsert(data)
            collection.flush()
            self._log(f"✓ Batch upserted successfully ({len(records)} records)")
            return len(records)
        except Exception as e:
            raise RuntimeError(f"Failed to upsert batch: {e}")

    def ingest_processed_records(self, records: List[Dict[str, Any]]) -> int:
        """Upsert only new records into Milvus (incremental indexing).

        Args:
            records: List of processed records from document processor

        Returns:
            Number of records upserted
        """
        if not records:
            self._log("No records to ingest.")
            return 0

        # Remove duplicates in current batch
        unique_records: List[Dict[str, Any]] = []
        seen_ids = set()
        for record in records:
            record_id = record.get("id")
            if not record_id or record_id in seen_ids:
                continue
            seen_ids.add(record_id)
            unique_records.append(record)

        if len(unique_records) != len(records):
            self._log(f"Removed {len(records) - len(unique_records)} duplicate records in batch.")

        records = unique_records

        # Skip records already indexed for incremental indexing
        indexed_ids = self._get_indexed_ids()
        new_records = [r for r in records if r.get("id") not in indexed_ids]

        if len(new_records) != len(records):
            skipped_count = len(records) - len(new_records)
            self._log(f"Skipped {skipped_count} already-indexed records (incremental indexing).")

        if not new_records:
            self._log("All records already indexed. No embedding needed.")
            return 0

        records = new_records

        # Upsert in batches
        total_upserted = 0
        total_batches = (len(records) + self.upsert_batch_size - 1) // self.upsert_batch_size
        for i in range(0, len(records), self.upsert_batch_size):
            batch = records[i : i + self.upsert_batch_size]
            batch_idx = i // self.upsert_batch_size + 1
            self._log(f"Processing batch {batch_idx}/{total_batches}...")
            total_upserted += self._upsert_batch(batch)
            self._log(f"Progress: {total_upserted}/{len(records)} records upserted.")

        return total_upserted

    def ingest_from_document_processor(
        self,
        only_new: bool = True,
        max_documents: Optional[int] = None,
    ) -> Dict[str, int]:
        """Pull records from document_processor and store them in Milvus.

        Args:
            only_new: Only ingest new documents (default: True)
            max_documents: Optional limit on documents to process

        Returns:
            Dictionary with statistics about ingestion
        """
        records = process_documents_for_vector_store(
            only_new=only_new,
            max_documents=max_documents,
            cache_file=self.cache_config.crawled_cache_file,
            cache_dir=self.cache_config.cache_dir,
        )
        self._log(f"Fetched {len(records)} records from document processor.")
        upserted = self.ingest_processed_records(records)
        self._log("Ingestion pipeline completed.")
        return {
            "records_fetched": len(records),
            "records_upserted": upserted,
            "collection_size": self.get_collection_count(),
        }

    def ingest_from_cache_only(self, max_documents: Optional[int] = None) -> Dict[str, int]:
        """Index only from local cache without crawling latest documents.

        Args:
            max_documents: Optional limit on documents to process

        Returns:
            Dictionary with statistics about ingestion
        """
        processor = DocumentProcessor(
            cache_file=self.cache_config.crawled_cache_file,
            cache_dir=self.cache_config.cache_dir,
        )
        records = processor.get_cached_records_for_indexing(limit_sources=max_documents)
        self._log(
            f"Cache-only mode: fetched {len(records)} records from local cache (no crawling)."
        )
        upserted = self.ingest_processed_records(records)
        self._log("Cache-only ingestion completed.")
        return {
            "records_fetched": len(records),
            "records_upserted": upserted,
            "collection_size": self.get_collection_count(),
        }

    def search(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Search for regulations similar to query text.

        Args:
            query_text: Query text to search for
            top_k: Number of top results to return (default: 5)

        Returns:
            Dictionary with search results (ids, texts, sources, similarities)
        """
        # Get embedding for query
        query_embedding = self._get_embedding(query_text)

        # Search in Milvus
        try:
            collection = Collection(self.COLLECTION_NAME)
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10},
            }

            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["id", "text", "source_url", "doc_title"],
            )

            # Extract and format results
            search_results = {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "distances": [],
            }

            if results and len(results) > 0 and len(results[0]) > 0:
                for hit in results[0]:
                    search_results["ids"].append(hit.id)
                    search_results["distances"].append(float(hit.distance))

                    # Get full document and metadata
                    entities = hit.entity
                    search_results["documents"].append(entities.get("text", ""))
                    search_results["metadatas"].append({
                        "source_url": entities.get("source_url", ""),
                        "doc_title": entities.get("doc_title", ""),
                        "distance": float(hit.distance),
                    })

            self._log(f"Search returned {len(search_results['ids'])} results")
            return search_results

        except Exception as e:
            raise RuntimeError(f"Search failed: {e}")

    def get_collection_count(self) -> int:
        """Get total number of records in collection.

        Returns:
            Number of records in collection
        """
        try:
            collection = Collection(self.COLLECTION_NAME)
            return collection.num_entities
        except Exception as e:
            self._log(f"Error getting collection count: {e}")
            return 0

    def delete_collection(self) -> None:
        """Delete the collection (use with caution!)."""
        try:
            utility.drop_collection(self.COLLECTION_NAME)
            self._log(f"✓ Deleted collection '{self.COLLECTION_NAME}'")
        except Exception as e:
            self._log(f"Error deleting collection: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the Milvus collection.

        Returns:
            Dictionary with collection statistics
        """
        return {
            "collection_name": self.COLLECTION_NAME,
            "num_entities": self.get_collection_count(),
            "embedding_dimension": self.EMBEDDING_DIMENSION,
            "embedding_model": self.embedding_model,
            "milvus_host": self.milvus_host,
            "milvus_port": self.milvus_port,
        }


def ingest_documents_to_milvus(
    only_new: bool = True,
    max_documents: Optional[int] = None,
    milvus_host: str = "localhost",
    milvus_port: int = 19530,
    vector_db_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    crawled_cache_file: Optional[str] = None,
) -> Dict[str, int]:
    """Convenience function for one-shot ingestion into Milvus.

    Args:
        only_new: Only ingest new documents
        max_documents: Optional limit on documents to process
        milvus_host: Milvus server host (default: localhost)
        milvus_port: Milvus server port (default: 19530)
        vector_db_path: Optional custom vector DB path
        cache_dir: Optional custom cache directory
        crawled_cache_file: Optional custom crawled cache file path

    Returns:
        Dictionary with ingestion statistics
    """
    store = MilvusVectorStore(
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        vector_db_path=vector_db_path,
        cache_dir=cache_dir,
        crawled_cache_file=crawled_cache_file,
    )
    return store.ingest_from_document_processor(only_new=only_new, max_documents=max_documents)


def _run_cli() -> None:
    """CLI for Milvus operations."""
    parser = argparse.ArgumentParser(description="Milvus vector store management")

    # Milvus connection arguments
    parser.add_argument(
        "--milvus-host",
        type=str,
        default="localhost",
        help="Milvus server host (env: MILVUS_HOST, default: localhost)",
    )
    parser.add_argument(
        "--milvus-port",
        type=int,
        default=19530,
        help="Milvus server port (env: MILVUS_PORT, default: 19530)",
    )

    # Cache location arguments
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Base cache directory (env: CACHE_DIR, default: ../cache/)",
    )
    parser.add_argument(
        "--crawled-cache-file",
        type=str,
        default=None,
        help="Path to crawled cache file (env: CRAWLED_CACHE_FILE, default: <cache-dir>/processed_records.jsonl)",
    )
    parser.add_argument(
        "--vector-db-path",
        type=str,
        default=None,
        help="Path to vector DB (env: VECTOR_DB_PATH, default: ../database/vectors/)",
    )

    # Operation arguments
    parser.add_argument(
        "--only-new",
        action="store_true",
        default=True,
        help="Only ingest new documents (default: True)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest all documents (overrides --only-new)",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Maximum number of documents to process",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Index only from local cache (no crawling)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print collection statistics and exit",
    )
    parser.add_argument(
        "--validate-dimension",
        action="store_true",
        help="Validate embedding dimension and exit",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete the collection (use with caution!)",
    )

    args = parser.parse_args()

    # Determine only_new flag
    only_new = not args.all

    # Initialize Milvus store
    try:
        store = MilvusVectorStore(
            milvus_host=args.milvus_host,
            milvus_port=args.milvus_port,
            vector_db_path=args.vector_db_path,
            cache_dir=args.cache_dir,
            crawled_cache_file=args.crawled_cache_file,
        )
    except Exception as e:
        logger.error(f"Failed to initialize Milvus store: {e}")
        return

    # Handle special operations
    if args.validate_dimension:
        try:
            dimension = store._get_embedding_dimension_from_gemini()
            print(f"\n✓ Embedding dimension validation successful!")
            print(f"  Dimension: {dimension}")
            print(f"  Model: {store.embedding_model}")
        except Exception as e:
            print(f"\n✗ Embedding dimension validation failed: {e}")
        return

    if args.stats:
        stats = store.get_stats()
        print("\n--- Milvus Collection Statistics ---")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return

    if args.delete:
        confirm = input(
            f"\n⚠️  Are you sure you want to delete collection '{store.COLLECTION_NAME}'? "
            "This cannot be undone. (yes/no): "
        )
        if confirm.lower() == "yes":
            store.delete_collection()
            print(f"✓ Collection deleted")
        else:
            print("✗ Deletion cancelled")
        return

    # Perform ingestion
    print("\n--- Milvus Ingestion Starting ---")
    print(f"  Host: {args.milvus_host}")
    print(f"  Port: {args.milvus_port}")
    print(f"  Only New: {only_new}")
    print(f"  Cache Only: {args.cache_only}")
    if args.max_documents:
        print(f"  Max Documents: {args.max_documents}")

    try:
        if args.cache_only:
            result = store.ingest_from_cache_only(max_documents=args.max_documents)
        else:
            result = store.ingest_from_document_processor(
                only_new=only_new,
                max_documents=args.max_documents,
            )

        print("\n--- Ingestion Complete ---")
        print(f"  Records Fetched: {result['records_fetched']}")
        print(f"  Records Upserted: {result['records_upserted']}")
        print(f"  Collection Size: {result['collection_size']}")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")


if __name__ == "__main__":
    _run_cli()
