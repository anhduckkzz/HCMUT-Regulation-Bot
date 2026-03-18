"""
Reindex doc_title field in Milvus from cached processed_records.jsonl.

This script reads the cached PDF chunks that contain title information and updates
the Milvus doc_title field for matching records.
"""

import argparse
import json
import logging
import os
from typing import Dict, List

from dotenv import load_dotenv
from pymilvus import Collection, connections

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class TitleReindexer:
    """Reindex doc_title field in Milvus from cached records."""

    def __init__(
        self,
    ):
        """Initialize Title Reindexer.
        """
        self.milvus_host = os.getenv("MILVUS_HOST", "localhost")
        self.milvus_port = int(os.getenv("MILVUS_PORT", "19530"))
        self.collection_name = os.getenv("MILVUS_COLLECTION_NAME", "hcmut_regulations")
        self.processed_records_file = os.getenv("CRAWLED_CACHE_FILE", "cache/processed_records.jsonl")

        # Connect to Milvus
        connections.connect(
            alias="default",
            host=self.milvus_host,
            port=self.milvus_port,
            timeout=10,
        )
        self.collection = Collection(self.collection_name)
        self.collection.load()
        logger.info(
            f"Connected to Milvus at {self.milvus_host}:{self.milvus_port}, collection: {self.collection_name}"
        )

    def load_cached_titles(self) -> Dict[str, str]:
        """Load title mapping from cached processed_records.jsonl.

        Returns:
            Dictionary mapping record_id -> title
        """
        titles_map = {}
        record_count = 0

        if not os.path.exists(self.processed_records_file):
            logger.error(f"Processed records file not found: {self.processed_records_file}")
            return titles_map

        logger.info(f"Loading titles from {self.processed_records_file}...")

        try:
            with open(self.processed_records_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        if "records" not in data:
                            continue

                        for record in data["records"]:
                            record_id = record.get("id")
                            title = record.get("metadata", {}).get("title", "")

                            if record_id:
                                titles_map[record_id] = title
                                record_count += 1

                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error reading processed records file: {e}")
            return titles_map

        logger.info(f"✓ Loaded {record_count} record titles from cache")
        return titles_map

    def get_existing_ids(self) -> Dict[str, str]:
        """Get all existing IDs from Milvus collection with their current doc_title and all fields.

        Returns:
            Dictionary mapping id -> dict with all record fields
        """
        try:
            # Query all records to get all fields needed for upsert
            results = self.collection.query(
                expr="id != ''",
                output_fields=["id", "doc_title", "text", "embedding", "source_url", "chunk_id"],
                limit=16384,  # Adjust if needed
            )
            id_map = {r["id"]: r for r in results}
            logger.info(f"✓ Retrieved {len(id_map)} existing records from Milvus")
            return id_map
        except Exception as e:
            logger.error(f"Error fetching existing IDs from Milvus: {e}")
            return {}

    def reindex_titles(self, dry_run: bool = False) -> Dict[str, int]:
        """Reindex titles from cached records into Milvus.

        Args:
            dry_run: If True, only show what would be updated without actually updating

        Returns:
            Dictionary with statistics
        """
        titles_map = self.load_cached_titles()
        if not titles_map:
            logger.error("No titles loaded from cache. Aborting.")
            return {"error": 1, "updated": 0, "skipped": 0, "not_found": 0}

        existing_ids = self.get_existing_ids()
        if not existing_ids:
            logger.error("No existing records in Milvus. Aborting.")
            return {"error": 1, "updated": 0, "skipped": 0, "not_found": 0}

        logger.info(f"Comparing {len(titles_map)} cached titles with {len(existing_ids)} Milvus records...")

        # Find records to update
        updates_needed = []
        already_set = []
        not_in_milvus = 0
        empty_titles = 0

        for record_id, title in titles_map.items():
            if not title:
                empty_titles += 1
                continue

            if record_id not in existing_ids:
                not_in_milvus += 1
                continue

            current_record = existing_ids[record_id]
            current_doc_title = current_record.get("doc_title", "")
            if current_doc_title != title:
                updates_needed.append((record_id, title, current_record))
            else:
                already_set.append(record_id)

        logger.info(f"\n=== REINDEX SUMMARY ===")
        logger.info(f"Total cached records: {len(titles_map)}")
        logger.info(f"Empty titles in cache: {empty_titles}")
        logger.info(f"Records not found in Milvus: {not_in_milvus}")
        logger.info(f"Records with correct title already set: {len(already_set)}")
        logger.info(f"Records needing title update: {len(updates_needed)}")

        if dry_run:
            logger.info("\n[DRY RUN] Would update the following records:")
            for record_id, title, _ in updates_needed[:10]:  # Show first 10
                logger.info(f"  {record_id}: {title[:60]}")
            if len(updates_needed) > 10:
                logger.info(f"  ... and {len(updates_needed) - 10} more records")
            return {
                "total": len(titles_map),
                "would_update": len(updates_needed),
                "already_set": len(already_set),
                "not_found": not_in_milvus,
                "empty": empty_titles,
            }

        # Actually update Milvus
        if not updates_needed:
            logger.info("No updates needed!")
            return {
                "total": len(titles_map),
                "updated": 0,
                "already_set": len(already_set),
                "not_found": not_in_milvus,
                "empty": empty_titles,
            }

        logger.info(f"\n[UPDATING] Starting to update {len(updates_needed)} records...")
        batch_size = 50
        successful_updates = 0
        failed_updates = 0

        for i in range(0, len(updates_needed), batch_size):
            batch = updates_needed[i : i + batch_size]
            batch_idx = i // batch_size + 1
            total_batches = (len(updates_needed) + batch_size - 1) // batch_size

            try:
                # Prepare data for batch upsert
                # Upsert requires all fields, so we preserve existing data and only update doc_title
                entities = []
                for record_id, new_title, current_record in batch:
                    entity = {
                        "id": current_record.get("id"),
                        "text": current_record.get("text"),
                        "embedding": current_record.get("embedding"),
                        "source_url": current_record.get("source_url"),
                        "chunk_id": current_record.get("chunk_id"),
                        "doc_title": new_title,  # Update the title
                    }
                    entities.append(entity)

                # Upsert the batch
                self.collection.upsert(entities)
                successful_updates += len(batch)
                logger.info(f"Batch {batch_idx}/{total_batches}: Upserted {len(batch)} records")

            except Exception as e:
                logger.error(f"Error upserting batch {batch_idx}: {e}")
                failed_updates += len(batch)

        logger.info(f"\n=== UPDATE RESULTS ===")
        logger.info(f"Successfully updated: {successful_updates}")
        logger.info(f"Failed updates: {failed_updates}")

        return {
            "total": len(titles_map),
            "updated": successful_updates,
            "failed": failed_updates,
            "already_set": len(already_set),
            "not_found": not_in_milvus,
            "empty": empty_titles,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reindex doc_title field in Milvus from cached processed_records.jsonl"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without actually updating Milvus",
    )

    args = parser.parse_args()

    reindexer = TitleReindexer()

    results = reindexer.reindex_titles(dry_run=args.dry_run)

    if args.dry_run:
        print("\n[DRY RUN COMPLETE] No changes made to Milvus.")
        print("Run without --dry-run to actually update records.")
    else:
        print(f"\n✓ Reindexing complete!")
        print(f"Updated records: {results.get('updated', 0)}")
        if results.get("failed", 0) > 0:
            print(f"⚠️  Failed updates: {results.get('failed', 0)}")
