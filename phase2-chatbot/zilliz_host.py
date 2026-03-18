"""Zilliz Cloud vector database integration for HCMUT regulation indexing.

Zilliz Cloud is a managed Milvus service. This module provides:
1. Connection to Zilliz Cloud cluster
2. Collection management with proper schema
3. Vector upload and indexing
4. Query functionality
"""

import argparse
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set

import google.generativeai as genai
from dotenv import load_dotenv
from pymilvus import MilvusException, Collection, connections, utility
from pymilvus import CollectionSchema, FieldSchema, DataType

from cache_config import CacheConfig
from document_processor import process_documents_for_vector_store

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalMilvusConnection:
    """Helper class to connect to local Milvus instance."""

    def __init__(self):
        """Initialize local Milvus connection.
        """
        self.milvus_host = os.getenv("MILVUS_HOST", "localhost")
        self.milvus_port = int(os.getenv("MILVUS_PORT", "19530"))
        self.collection_name = "hcmut_regulations"

    def connect(self) -> Collection:
        """Connect to local Milvus and return collection."""
        try:
            connections.connect(
                alias="local_milvus",
                host=self.milvus_host,
                port=self.milvus_port,
                timeout=10,
            )
            collection = Collection(self.collection_name, using="local_milvus")
            return collection
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to local Milvus at {self.milvus_host}:{self.milvus_port}: {e}"
            )

    def get_all_vectors(self) -> List[Dict[str, Any]]:
        """Retrieve all vectors from local Milvus collection.
        
        Returns:
            List of dictionaries with id, embedding, text, metadata
        """
        collection = self.connect()
        
        try:
            # Query all records with embedding field
            results = collection.query(
                expr="id != ''",
                output_fields=["id", "embedding", "text", "source_url", "chunk_id", "doc_title"],
                limit=16384,  # Adjust if you have more than 100k records
            )
            
            print(f"✅ Retrieved {len(results)} vectors from local Milvus")
            return results
        except Exception as e:
            print(f"❌ Error retrieving vectors: {e}")
            raise
        finally:
            connections.disconnect(alias="local_milvus")


class ZillizCloudVectorStore:
    """Embed processed regulation chunks and persist them in Zilliz Cloud."""

    # Gemini embedding model (models/gemini-embedding-001) produces 3072-dimensional vectors
    EMBEDDING_DIMENSION = 3072
    COLLECTION_NAME = "hcmut_regulations"
    PARTITION_NAME = "regulations_partition"

    def __init__(
        self,
        zilliz_endpoint: Optional[str] = None,
        zilliz_api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        """Initialize Zilliz Cloud Vector Store.

        Args:
            zilliz_endpoint: Zilliz Cloud public endpoint (env: ZILLIZ_CLOUD_ENDPOINT)
            zilliz_api_key: Zilliz Cloud API key (env: ZILLIZ_CLOUD_API_KEY)
            collection_name: Collection name (env: ZILLIZ_COLLECTION_NAME, default: hcmut_regulations)
        """
        # Get Zilliz Cloud configuration from environment
        self.zilliz_endpoint = zilliz_endpoint or os.getenv("ZILLIZ_CLOUD_ENDPOINT")
        self.zilliz_api_key = zilliz_api_key or os.getenv("ZILLIZ_CLOUD_API_KEY")

        if not self.zilliz_endpoint:
            raise ValueError(
                "Missing ZILLIZ_CLOUD_ENDPOINT in .env. "
                "Get it from Zilliz Cloud dashboard: https://cloud.zilliz.com"
            )
        if not self.zilliz_api_key:
            raise ValueError(
                "Missing ZILLIZ_CLOUD_API_KEY in .env. "
                "Generate it from Zilliz Cloud dashboard"
            )

        # Get Gemini configuration
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("Missing GEMINI_API_KEY in .env")

        genai.configure(api_key=self.gemini_api_key)

        self.embedding_model = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
        self.embedding_task_type = os.getenv("EMBEDDING_TASK_TYPE", "retrieval_document")
        self.embedding_retry_count = int(os.getenv("EMBEDDING_RETRY_COUNT", "3"))
        self.embedding_retry_delay = float(os.getenv("EMBEDDING_RETRY_DELAY", "1.5"))
        self.verbose_logs = os.getenv("VERBOSE_LOGS", "true").lower() == "true"

        # Collection configuration
        self.collection_name = collection_name or os.getenv(
            "ZILLIZ_COLLECTION_NAME", self.COLLECTION_NAME
        )

        # Initialize cache configuration
        self.cache_config = CacheConfig()
        self.cache_config.ensure_directories_exist()

        self.upsert_batch_size = int(os.getenv("UPSERT_BATCH_SIZE", "32"))

        # Connect to Zilliz Cloud
        self._connect_to_zilliz()
        self._ensure_collection()

        self._log(f"✅ Connected to Zilliz Cloud collection '{self.collection_name}'")

    def _log(self, message: str) -> None:
        """Log message with prefix."""
        if self.verbose_logs:
            print(f"[zilliz_host] {message}")
            logger.info(message)

    def _connect_to_zilliz(self) -> None:
        """Establish connection to Zilliz Cloud."""
        try:
            # Disconnect any existing connections
            connections.disconnect(alias="default")
        except Exception:
            pass

        try:
            connections.connect(
                alias="default",
                uri=self.zilliz_endpoint,
                token=self.zilliz_api_key,
                timeout=30,
            )
            self._log("✅ Connected to Zilliz Cloud")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Zilliz Cloud: {e}")

    def _ensure_collection(self) -> None:
        """Create or get collection with proper schema."""
        try:
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                self._log(f"📦 Using existing collection: {self.collection_name}")
            else:
                self._create_collection()
        except Exception as e:
            self._log(f"⚠️ Error checking collection: {e}")
            self._create_collection()

    def _create_collection(self) -> None:
        """Create collection with proper schema for regulation embeddings."""
        schema = CollectionSchema(
            fields=[
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=500,
                    description="Unique chunk identifier",
                ),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.EMBEDDING_DIMENSION,
                    description="Gemini embedding vector (3072 dimensions)",
                ),
                FieldSchema(
                    name="text",
                    dtype=DataType.VARCHAR,
                    max_length=65535,
                    description="Chunk text content",
                ),
                FieldSchema(
                    name="source_url",
                    dtype=DataType.VARCHAR,
                    max_length=500,
                    description="PDF source URL",
                ),
                FieldSchema(
                    name="chunk_id",
                    dtype=DataType.VARCHAR,
                    max_length=500,
                    description="Original chunk identifier",
                ),
                FieldSchema(
                    name="doc_title",
                    dtype=DataType.VARCHAR,
                    max_length=500,
                    description="Document title",
                ),
            ],
            description="HCMUT regulation embeddings collection",
            enable_dynamic_field=False,
        )

        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 1024},
        }

        try:
            self.collection = Collection(
                name=self.collection_name,
                schema=schema,
                using="default",
            )
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params,
            )
            self._log(f"✅ Created collection '{self.collection_name}'")
        except Exception as e:
            raise RuntimeError(f"Failed to create collection: {e}")

    def _get_indexed_ids(self) -> Set[str]:
        """Get all IDs currently indexed in the collection."""
        try:
            all_data = self.collection.query(
                expr="",
                output_fields=["id"],
                limit=100000,
            )
            indexed_ids = set(doc["id"] for doc in all_data if "id" in doc)
            self._log(f"📊 Found {len(indexed_ids)} indexed documents")
            return indexed_ids
        except Exception as e:
            self._log(f"⚠️ Error fetching indexed IDs: {e}")
            return set()

    def _embed_text(self, text: str) -> List[float]:
        """Create one embedding vector using Gemini Text Embeddings."""
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
                self._log(
                    f"⚠️ Embedding attempt {attempt}/{self.embedding_retry_count} failed: {exc}"
                )
                if attempt < self.embedding_retry_count:
                    time.sleep(self.embedding_retry_delay)

        raise RuntimeError(f"Embedding failed after retries: {last_error}")

    def _upsert_batch(self, records: List[Dict[str, Any]]) -> int:
        """Upsert a batch of records into Zilliz Cloud."""
        if not records:
            return 0

        try:
            # Prepare data for upsert
            ids = []
            embeddings = []
            texts = []
            source_urls = []
            chunk_ids = []
            doc_titles = []

            for record in records:
                ids.append(record["id"])
                embeddings.append(record["embedding"])
                texts.append(record["text"])
                source_urls.append(record.get("source_url", ""))
                chunk_ids.append(record.get("chunk_id", ""))
                doc_titles.append(record.get("doc_title", ""))

            # Upsert to Zilliz Cloud
            self.collection.upsert(
                data=[
                    ids,
                    embeddings,
                    texts,
                    source_urls,
                    chunk_ids,
                    doc_titles,
                ]
            )

            self._log(f"✅ Upserted {len(records)} records to Zilliz Cloud")
            return len(records)
        except Exception as e:
            self._log(f"❌ Error upserting batch: {e}")
            raise

    def index_documents(self, only_new: bool = True) -> int:
        """Index documents into Zilliz Cloud collection.

        Args:
            only_new: If True, only index new documents not in collection

        Returns:
            Number of documents indexed
        """
        self._log("🔄 Starting document indexing to Zilliz Cloud...")

        # Get processed documents
        try:
            records = process_documents_for_vector_store()
            self._log(f"📄 Loaded {len(records)} records from cache")
        except Exception as e:
            self._log(f"❌ Error loading documents: {e}")
            raise

        if not records:
            self._log("⚠️ No documents to index")
            return 0

        # Filter only new records if requested
        if only_new:
            indexed_ids = self._get_indexed_ids()
            new_records = [r for r in records if r["id"] not in indexed_ids]
            self._log(f"🆕 Found {len(new_records)} new records to index")
            records = new_records

        if not records:
            self._log("✅ No new documents to index")
            return 0

        # Embed and upsert in batches
        total_indexed = 0
        batches = [
            records[i : i + self.upsert_batch_size]
            for i in range(0, len(records), self.upsert_batch_size)
        ]

        for batch_idx, batch in enumerate(batches, 1):
            self._log(f"⏳ Processing batch {batch_idx}/{len(batches)}...")

            # Embed texts in batch
            for record in batch:
                try:
                    record["embedding"] = self._embed_text(record["text"])
                except Exception as e:
                    self._log(f"⚠️ Error embedding record {record['id']}: {e}")
                    continue

            # Upsert batch
            try:
                indexed = self._upsert_batch(batch)
                total_indexed += indexed
            except Exception as e:
                self._log(f"❌ Error upserting batch {batch_idx}: {e}")
                continue

        # Flush collection
        try:
            self.collection.flush()
            self._log("✅ Collection flushed")
        except Exception as e:
            self._log(f"⚠️ Error flushing collection: {e}")

        self._log(f"✅ Indexing complete. Total: {total_indexed} documents")
        return total_indexed

    def migrate_from_local_milvus(self, only_new: bool = True) -> int:
        """Migrate vectors from local Milvus to Zilliz Cloud.
        
        This is the recommended way to migrate your existing indexed vectors.
        
        Args:
            only_new: If True, only migrate vectors not already in Zilliz Cloud
            
        Returns:
            Number of vectors migrated
        """
        self._log("🚀 Starting migration from local Milvus to Zilliz Cloud...")
        
        try:
            # Connect to local Milvus and retrieve all vectors
            local_milvus = LocalMilvusConnection()
            records = local_milvus.get_all_vectors()
            
            if not records:
                self._log("⚠️ No vectors found in local Milvus")
                return 0
                
            self._log(f"📦 Retrieved {len(records)} vectors from local Milvus")
            
            # Filter only new records if requested
            if only_new:
                indexed_ids = self._get_indexed_ids()
                new_records = [r for r in records if r.get("id") not in indexed_ids]
                self._log(f"🆕 Found {len(new_records)} new vectors to migrate")
                self._log(f"⏭️  Skipping {len(records) - len(new_records)} already-migrated vectors")
                records = new_records
            
            if not records:
                self._log("✅ All vectors already migrated")
                return 0
            
            # Upsert in batches (embeddings already exist, no need to regenerate)
            total_migrated = 0
            batches = [
                records[i : i + self.upsert_batch_size]
                for i in range(0, len(records), self.upsert_batch_size)
            ]
            
            self._log(f"⏳ Migrating in {len(batches)} batches...")
            
            for batch_idx, batch in enumerate(batches, 1):
                self._log(f"⏳ Migrating batch {batch_idx}/{len(batches)} ({len(batch)} vectors)...")
                
                try:
                    migrated = self._upsert_batch(batch)
                    total_migrated += migrated
                except Exception as e:
                    self._log(f"❌ Error migrating batch {batch_idx}: {e}")
                    continue
            
            # Flush collection
            try:
                self.collection.flush()
                self._log("✅ Collection flushed")
            except Exception as e:
                self._log(f"⚠️ Error flushing collection: {e}")
            
            self._log(f"✅ Migration complete! Migrated {total_migrated} vectors")
            return total_migrated
            
        except Exception as e:
            self._log(f"❌ Migration failed: {e}")
            raise

    def search(
        self,
        query_text: str,
        k: int = 5,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Search for similar documents in Zilliz Cloud.

        Args:
            query_text: Query text
            k: Number of results to return
            search_params: Custom search parameters

        Returns:
            Search results with documents and metadata
        """
        try:
            # Embed query
            query_vector = self._embed_text(query_text)

            # Default search parameters
            if search_params is None:
                search_params = {
                    "metric_type": "L2",
                    "params": {"nprobe": 10},
                }

            # Search
            results = self.collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=k,
                output_fields=["id", "text", "source_url", "doc_title"],
            )

            # Format results
            formatted_results = {
                "documents": [[]],
                "metadatas": [[]],
                "ids": [],
                "distances": [],
            }

            if results and len(results) > 0:
                for hit in results[0]:
                    formatted_results["documents"][0].append(hit.entity.get("text", ""))
                    formatted_results["metadatas"][0].append({
                        "source_url": hit.entity.get("source_url", ""),
                        "doc_title": hit.entity.get("doc_title", ""),
                    })
                    formatted_results["ids"].append(hit.id)
                    formatted_results["distances"].append(float(hit.distance))

            self._log(f"🔍 Search returned {len(formatted_results['ids'])} results")
            return formatted_results
        except Exception as e:
            self._log(f"❌ Error during search: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics and info."""
        try:
            num_entities = self.collection.num_entities
            collection_info = {
                "name": self.collection_name,
                "num_entities": num_entities,
                "embedding_dimension": self.EMBEDDING_DIMENSION,
                "status": "ready",
            }
            self._log(f"📊 Collection info: {collection_info}")
            return collection_info
        except Exception as e:
            self._log(f"❌ Error getting collection info: {e}")
            raise

    def delete_collection(self) -> None:
        """Delete the collection from Zilliz Cloud."""
        try:
            utility.drop_collection(self.collection_name)
            self._log(f"✅ Deleted collection '{self.collection_name}'")
        except Exception as e:
            self._log(f"❌ Error deleting collection: {e}")
            raise


def main():
    """CLI for Zilliz Cloud vector store operations."""
    parser = argparse.ArgumentParser(
        description="Zilliz Cloud Vector Store Management"
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        default=False,
        help="🚀 Migrate vectors from local Milvus to Zilliz Cloud (RECOMMENDED for first-time setup)",
    )
    parser.add_argument(
        "--only-new",
        action="store_true",
        default=False,
        help="Index/migrate only new documents not in collection",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        default=False,
        help="Display collection information",
    )
    parser.add_argument(
        "--delete-collection",
        action="store_true",
        default=False,
        help="Delete the collection from Zilliz Cloud",
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Search query text",
    )
    parser.add_argument(
        "--search-k",
        type=int,
        default=5,
        help="Number of search results (default: 5)",
    )

    args = parser.parse_args()

    try:
        # Initialize Zilliz Cloud vector store
        vector_store = ZillizCloudVectorStore()

        if args.delete_collection:
            confirm = input(
                f"⚠️  Are you sure you want to delete collection '{vector_store.collection_name}'? (yes/no): "
            )
            if confirm.lower() == "yes":
                vector_store.delete_collection()
            else:
                print("❌ Deletion cancelled")
            return

        if args.info:
            info = vector_store.get_collection_info()
            print("\n📊 Collection Information:")
            for key, value in info.items():
                print(f"  {key}: {value}")
            return

        if args.migrate:
            print("\n🚀 Migration Mode: Transferring vectors from local Milvus → Zilliz Cloud")
            migrated_count = vector_store.migrate_from_local_milvus(only_new=args.only_new)
            print(f"\n✅ Successfully migrated {migrated_count} vectors to Zilliz Cloud")
            return

        if args.search:
            print(f"\n🔍 Searching for: '{args.search}'")
            results = vector_store.search(args.search, k=args.search_k)
            print(f"\n✅ Found {len(results['ids'])} results:")
            for i, (doc_id, distance, text) in enumerate(
                zip(results["ids"], results["distances"], results["documents"][0]), 1
            ):
                print(f"\n[{i}] Distance: {distance:.4f}")
                print(f"    ID: {doc_id}")
                print(f"    Text: {text[:200]}...")
            return

        # Default: index documents from scratch (generates new embeddings)
        print("\n📝 Default Mode: Indexing documents (generating embeddings from scratch)")
        indexed_count = vector_store.index_documents(only_new=args.only_new)
        print(f"\n✅ Successfully indexed {indexed_count} documents to Zilliz Cloud")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Fatal error")
        exit(1)


if __name__ == "__main__":
    main()
