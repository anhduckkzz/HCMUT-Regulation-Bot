import argparse
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set

import google.generativeai as genai
from dotenv import load_dotenv
from pymilvus import Collection, connections, utility

from cache_config import CacheConfig
from document_processor import DocumentProcessor, process_documents_for_vector_store

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MilvusVectorStore:
    """Embed processed regulation chunks and persist them in Milvus database."""

    EMBEDDING_DIMENSION = 3072
    COLLECTION_NAME = "hcmut_regulations"

    def __init__(self) -> None:
        self.milvus_host = os.getenv("MILVUS_HOST", "localhost")
        self.milvus_port = int(os.getenv("MILVUS_PORT", "19530"))

        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("Missing GEMINI_API_KEY in .env")

        genai.configure(api_key=self.gemini_api_key)

        self.embedding_model = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
        self.embedding_task_type = os.getenv("EMBEDDING_TASK_TYPE", "retrieval_document")
        self.embedding_retry_count = int(os.getenv("EMBEDDING_RETRY_COUNT", "3"))
        self.embedding_retry_delay = float(os.getenv("EMBEDDING_RETRY_DELAY", "1.5"))
        self.verbose_logs = os.getenv("VERBOSE_LOGS", "true").lower() == "true"

        self.cache_config = CacheConfig()
        self.cache_config.ensure_directories_exist()

        self.upsert_batch_size = int(os.getenv("UPSERT_BATCH_SIZE", "32"))

        self._connect_to_milvus()
        self._initialize_collection()

        self._log(f"Initialized Milvus connection at {self.milvus_host}:{self.milvus_port}")

    def _log(self, message: str) -> None:
        if self.verbose_logs:
            logger.info(f"[milvus_store] {message}")

    def _connect_to_milvus(self) -> None:
        try:
            connections.connect(
                alias="default",
                host=self.milvus_host,
                port=self.milvus_port,
                timeout=10,
            )
            self._log(f"Connected to Milvus at {self.milvus_host}:{self.milvus_port}")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to connect to Milvus at {self.milvus_host}:{self.milvus_port}: {exc}"
            )

    def _get_embedding_dimension_from_gemini(self) -> int:
        try:
            self._log("Validating Gemini embedding dimension...")
            response = genai.embed_content(
                model=self.embedding_model,
                content="Test embedding validation for HCMUT regulations",
                task_type=self.embedding_task_type,
            )
            embedding = response.get("embedding", [])
            dimension = len(embedding)

            if dimension != self.EMBEDDING_DIMENSION:
                raise RuntimeError(
                    f"Embedding dimension mismatch! Expected {self.EMBEDDING_DIMENSION}, "
                    f"but got {dimension} from Gemini API"
                )

            self._log(f"Gemini embedding dimension validated: {dimension}")
            return dimension
        except Exception as exc:
            raise RuntimeError(f"Failed to validate Gemini embedding dimension: {exc}")

    def _initialize_collection(self) -> None:
        from pymilvus import CollectionSchema, DataType, FieldSchema

        actual_dimension = self._get_embedding_dimension_from_gemini()

        if utility.has_collection(self.COLLECTION_NAME):
            self._log(f"Collection '{self.COLLECTION_NAME}' already exists. Loading...")
            collection = Collection(self.COLLECTION_NAME)
            collection.load()

            schema = collection.schema
            for field in schema.fields:
                if field.name == "embedding":
                    field_dim = field.params.get("dim", 0)
                    if field_dim != self.EMBEDDING_DIMENSION:
                        raise RuntimeError(
                            "Collection embedding dimension mismatch! "
                            f"Expected {self.EMBEDDING_DIMENSION}, but collection has {field_dim}"
                        )
            self._log(f"Collection schema verified: embedding dimension = {actual_dimension}")
            return

        self._log(
            f"Creating new collection '{self.COLLECTION_NAME}' with dimension {actual_dimension}..."
        )

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

        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()

        self._log("Collection created with IVF_FLAT index")

    def _get_embedding(self, text: str) -> List[float]:
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
        try:
            collection = Collection(self.COLLECTION_NAME)
            results = collection.query(expr="id != ''", output_fields=["id"])
            return {result["id"] for result in results}
        except Exception as exc:
            self._log(f"Error fetching indexed IDs: {exc}")
            return set()

    def _upsert_batch(self, records: List[Dict[str, Any]]) -> int:
        if not records:
            return 0

        first_id = records[0]["id"]
        self._log(f"Embedding and upserting batch size={len(records)} (first id: {first_id})")

        ids = [record["id"] for record in records]
        documents = [record["text"] for record in records]
        metadatas = [record.get("metadata", {}) for record in records]

        embeddings = [self._get_embedding(text) for text in documents]

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
            self._log(f"Batch upserted successfully ({len(records)} records)")
            return len(records)
        except Exception as exc:
            raise RuntimeError(f"Failed to upsert batch: {exc}")

    def ingest_processed_records(self, records: List[Dict[str, Any]]) -> int:
        if not records:
            self._log("No records to ingest.")
            return 0

        unique_records: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()
        for record in records:
            record_id = record.get("id")
            if not record_id or record_id in seen_ids:
                continue
            seen_ids.add(record_id)
            unique_records.append(record)

        records = unique_records

        indexed_ids = self._get_indexed_ids()
        records = [r for r in records if r.get("id") not in indexed_ids]

        if not records:
            self._log("All records already indexed. No embedding needed.")
            return 0

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
        records = process_documents_for_vector_store(
            only_new=only_new,
            max_documents=max_documents,
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
        processor = DocumentProcessor()
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

    def search(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        query_embedding = self._get_embedding(query_text)

        try:
            collection = Collection(self.COLLECTION_NAME)
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                limit=top_k,
                output_fields=["id", "text", "source_url", "doc_title"],
            )

            search_results = {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "distances": [],
            }

            if results and results[0]:
                for hit in results[0]:
                    search_results["ids"].append(hit.id)
                    search_results["distances"].append(float(hit.distance))
                    entity = hit.entity
                    search_results["documents"].append(entity.get("text", ""))
                    search_results["metadatas"].append(
                        {
                            "source_url": entity.get("source_url", ""),
                            "doc_title": entity.get("doc_title", ""),
                            "distance": float(hit.distance),
                        }
                    )

            self._log(f"Search returned {len(search_results['ids'])} results")
            return search_results
        except Exception as exc:
            raise RuntimeError(f"Search failed: {exc}")

    def get_collection_count(self) -> int:
        try:
            collection = Collection(self.COLLECTION_NAME)
            return collection.num_entities
        except Exception as exc:
            self._log(f"Error getting collection count: {exc}")
            return 0

    def delete_collection(self) -> None:
        try:
            utility.drop_collection(self.COLLECTION_NAME)
            self._log(f"Deleted collection '{self.COLLECTION_NAME}'")
        except Exception as exc:
            self._log(f"Error deleting collection: {exc}")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "collection_name": self.COLLECTION_NAME,
            "num_entities": self.get_collection_count(),
            "embedding_dimension": self.EMBEDDING_DIMENSION,
            "embedding_model": self.embedding_model,
            "milvus_host": self.milvus_host,
            "milvus_port": self.milvus_port,
        }


class VectorStoreManager(MilvusVectorStore):
    """Backward-compatible alias for Milvus-backed vector store manager."""


def ingest_documents_to_milvus(
    only_new: bool = True,
    max_documents: Optional[int] = None,
) -> Dict[str, int]:
    store = VectorStoreManager()
    return store.ingest_from_document_processor(only_new=only_new, max_documents=max_documents)


def _run_cli() -> None:
    parser = argparse.ArgumentParser(description="Milvus vector store ingestion and cache maintenance")

    parser.add_argument("--only-new", action="store_true", help="Ingest only new regulations")
    parser.add_argument("--all", action="store_true", help="Ingest all documents")
    parser.add_argument(
        "--index-from-cache-only",
        "--cache-only",
        action="store_true",
        dest="cache_only",
        help="Index only from local cache (no crawling)",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Optional max number of source documents to process",
    )

    parser.add_argument("--stats", action="store_true", help="Print collection statistics and exit")
    parser.add_argument(
        "--validate-dimension",
        action="store_true",
        help="Validate embedding dimension and exit",
    )
    parser.add_argument("--delete", action="store_true", help="Delete the collection")

    parser.add_argument("--cache-stats", action="store_true", help="Show document cache stats")
    parser.add_argument("--clear-cache", action="store_true", help="Clear document cache file")
    parser.add_argument("--dedupe-cache", action="store_true", help="Dedupe document cache entries")
    parser.add_argument(
        "--rebuild-cache-from-state",
        action="store_true",
        help="Backfill cache for links in state file that are missing in cache",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for --rebuild-cache-from-state",
    )

    args = parser.parse_args()

    cache_command = any(
        [
            args.cache_stats,
            args.clear_cache,
            args.dedupe_cache,
            args.rebuild_cache_from_state,
        ]
    )

    if cache_command:
        processor = DocumentProcessor()
        print(f"[INFO] Using cache configuration: {processor.cache_config}")

        if args.clear_cache:
            processor.clear_cache()
            print("Cache cleared.")
            return

        if args.dedupe_cache:
            result = processor.dedupe_cache_file()
            print("Cache deduplicated.")
            print(result)
            return

        if args.rebuild_cache_from_state:
            result = processor.rebuild_cache_from_state(limit=args.limit)
            print("Cache rebuild from state completed.")
            print(result)
            return

        if args.cache_stats:
            print(processor.get_cache_stats())
            return

    try:
        store = VectorStoreManager()
    except Exception as exc:
        logger.error(f"Failed to initialize Milvus store: {exc}")
        return

    if args.validate_dimension:
        try:
            dimension = store._get_embedding_dimension_from_gemini()
            print("\nEmbedding dimension validation successful")
            print(f"  Dimension: {dimension}")
            print(f"  Model: {store.embedding_model}")
        except Exception as exc:
            print(f"\nEmbedding dimension validation failed: {exc}")
        return

    if args.stats:
        stats = store.get_stats()
        print("\n--- Milvus Collection Statistics ---")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return

    if args.delete:
        confirm = input(
            f"\nAre you sure you want to delete collection '{store.COLLECTION_NAME}'? "
            "This cannot be undone. (yes/no): "
        )
        if confirm.lower() == "yes":
            store.delete_collection()
            print("Collection deleted")
        else:
            print("Deletion cancelled")
        return

    only_new = args.only_new and not args.all

    print("\n--- Milvus Ingestion Starting ---")
    print(f"  Host: {store.milvus_host}")
    print(f"  Port: {store.milvus_port}")
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
    except Exception as exc:
        logger.error(f"Ingestion failed: {exc}")


if __name__ == "__main__":
    _run_cli()
