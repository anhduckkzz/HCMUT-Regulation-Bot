import argparse
import os
from typing import Any, Dict, List, Optional, Set

import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

from document_processor import DocumentProcessor, process_documents_for_vector_store


load_dotenv()


class VectorStoreManager:
	"""Embed processed regulation chunks and persist them in ChromaDB."""

	def __init__(self) -> None:
		self.gemini_api_key = os.getenv("GEMINI_API_KEY")
		if not self.gemini_api_key:
			raise ValueError("Missing GEMINI_API_KEY in .env")

		genai.configure(api_key=self.gemini_api_key)

		self.embedding_model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
		self.embedding_task_type = os.getenv("EMBEDDING_TASK_TYPE", "retrieval_document")
		self.embedding_retry_count = int(os.getenv("EMBEDDING_RETRY_COUNT", "3"))
		self.embedding_retry_delay = float(os.getenv("EMBEDDING_RETRY_DELAY", "1.5"))
		self.verbose_logs = os.getenv("VERBOSE_LOGS", "true").lower() == "true"

		self.persist_directory = os.getenv(
			"VECTOR_DB_PATH",
			os.path.join(os.path.dirname(__file__), "..", "database", "vectors"),
		)
		self.collection_name = os.getenv("VECTOR_COLLECTION_NAME", "hcmut_regulations")
		self.upsert_batch_size = int(os.getenv("UPSERT_BATCH_SIZE", "32"))

		self.client = chromadb.PersistentClient(path=self.persist_directory)
		self.collection = self.client.get_or_create_collection(name=self.collection_name)
		self._log(
			f"Initialized Chroma collection '{self.collection_name}' at '{self.persist_directory}'."
		)

	def _log(self, message: str) -> None:
		if self.verbose_logs:
			print(f"[vector_store] {message}")

	def _get_indexed_ids(self) -> Set[str]:
		"""Get all IDs currently indexed in the collection."""
		try:
			all_ids = self.collection.get()["ids"]
			return set(all_ids) if all_ids else set()
		except Exception as e:
			self._log(f"Error fetching indexed IDs: {e}")
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
				self._log(f"Embedding attempt {attempt}/{self.embedding_retry_count} failed: {exc}")
				if attempt < self.embedding_retry_count:
					import time
					time.sleep(self.embedding_retry_delay)

		raise RuntimeError(f"Embedding failed after retries: {last_error}")

	def _upsert_batch(self, records: List[Dict[str, Any]]) -> int:
		if not records:
			return 0

		first_id = records[0]["id"]
		self._log(f"Embedding and upserting batch size={len(records)} (first id: {first_id})")

		ids = [record["id"] for record in records]
		documents = [record["text"] for record in records]
		metadatas = [record.get("metadata", {}) for record in records]
		embeddings = [self._embed_text(text) for text in documents]

		self.collection.upsert(
			ids=ids,
			documents=documents,
			metadatas=metadatas,
			embeddings=embeddings,
		)
		self._log(f"Batch upserted successfully ({len(records)} records).")
		return len(records)

	def ingest_processed_records(self, records: List[Dict[str, Any]]) -> int:
		"""Upsert only new records into ChromaDB (incremental indexing)."""
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

		# Skip records already indexed to enable incremental indexing
		indexed_ids = self._get_indexed_ids()
		new_records = [r for r in records if r.get("id") not in indexed_ids]

		if len(new_records) != len(records):
			skipped_count = len(records) - len(new_records)
			self._log(f"Skipped {skipped_count} already-indexed records (incremental indexing).")

		if not new_records:
			self._log("All records already indexed. No embedding needed.")
			return 0

		records = new_records

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
		"""Pull records from document_processor and store them in ChromaDB."""
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
			"collection_size": self.collection.count(),
		}

	def ingest_from_cache_only(self, max_documents: Optional[int] = None) -> Dict[str, int]:
		"""Index only from local cache without crawling latest documents."""
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
			"collection_size": self.collection.count(),
		}


def ingest_documents_to_chromadb(
	only_new: bool = True,
	max_documents: Optional[int] = None,
) -> Dict[str, int]:
	"""Convenience function for one-shot ingestion into ChromaDB."""
	store = VectorStoreManager()
	return store.ingest_from_document_processor(only_new=only_new, max_documents=max_documents)


def _run_cli() -> None:
	parser = argparse.ArgumentParser(description="Vector store ingestion and cache maintenance")
	parser.add_argument("--only-new", action="store_true", help="Ingest only new regulations")
	parser.add_argument(
		"--index-from-cache-only",
		action="store_true",
		help="Index only from local cache (no crawling, downloading, or PDF extraction)",
	)
	parser.add_argument(
		"--max-documents",
		type=int,
		default=None,
		help="Optional max number of source documents to process",
	)

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

	if args.index_from_cache_only:
		store = VectorStoreManager()
		result = store.ingest_from_cache_only(max_documents=args.max_documents)
	else:
		result = ingest_documents_to_chromadb(
			only_new=args.only_new,
			max_documents=args.max_documents,
		)
	print("Vector ingestion completed")
	print(f"Records fetched: {result['records_fetched']}")
	print(f"Records upserted: {result['records_upserted']}")
	print(f"Collection size: {result['collection_size']}")


if __name__ == "__main__":
	_run_cli()