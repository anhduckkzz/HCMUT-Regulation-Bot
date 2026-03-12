"""
Centralized cache configuration for HCMUT Regulation Bot.

This module manages all cache-related paths and settings, supporting:
- Environment variable defaults
- CLI argument overrides
- Automatic directory creation
"""

import os
from pathlib import Path
from typing import Optional


class CacheConfig:
	"""Manages cache configuration with env var fallback and CLI override support."""

	def __init__(
		self,
		cache_dir: Optional[str] = None,
		crawled_cache_file: Optional[str] = None,
		vector_db_path: Optional[str] = None,
	):
		"""
		Initialize cache configuration.

		Args:
			cache_dir: Base cache directory. Defaults to env var CACHE_DIR or "cache/"
			crawled_cache_file: Path to crawled PDFs cache file. Defaults to CRAWLED_CACHE_FILE env var
			vector_db_path: Path to vector DB. Defaults to env var VECTOR_DB_PATH

		The priority order is:
		1. Explicit parameter passed to __init__
		2. Environment variable
		3. Default value
		"""
		# Base cache directory
		self.cache_dir = (
			cache_dir
			or os.getenv("CACHE_DIR", os.path.join(os.path.dirname(__file__), "..", "cache"))
		)
		self.cache_dir = os.path.abspath(self.cache_dir)

		# Crawled PDFs cache file (processed_records.jsonl)
		default_crawled = os.path.join(self.cache_dir, "processed_records.jsonl")
		self.crawled_cache_file = (
			crawled_cache_file or os.getenv("CRAWLED_CACHE_FILE", default_crawled)
		)
		self.crawled_cache_file = os.path.abspath(self.crawled_cache_file)

		# Vector DB path (ChromaDB persistent storage)
		default_vector_db = os.path.join(os.path.dirname(__file__), "..", "database", "vectors")
		self.vector_db_path = vector_db_path or os.getenv("VECTOR_DB_PATH", default_vector_db)
		self.vector_db_path = os.path.abspath(self.vector_db_path)

		# Collection name
		self.collection_name = os.getenv("VECTOR_COLLECTION_NAME", "hcmut_regulations")

	def ensure_directories_exist(self) -> None:
		"""Create cache and vector DB directories if they don't exist."""
		os.makedirs(self.cache_dir, exist_ok=True)
		os.makedirs(os.path.dirname(self.crawled_cache_file), exist_ok=True)
		os.makedirs(self.vector_db_path, exist_ok=True)

	def get_crawled_cache_dir(self) -> str:
		"""Get directory containing the crawled cache file."""
		return os.path.dirname(self.crawled_cache_file)

	def to_dict(self) -> dict:
		"""Return configuration as dictionary for logging/debugging."""
		return {
			"cache_dir": self.cache_dir,
			"crawled_cache_file": self.crawled_cache_file,
			"vector_db_path": self.vector_db_path,
			"collection_name": self.collection_name,
		}

	def __repr__(self) -> str:
		"""String representation for debugging."""
		return f"CacheConfig({self.to_dict()})"


def get_cache_config(
	cache_dir: Optional[str] = None,
	crawled_cache_file: Optional[str] = None,
	vector_db_path: Optional[str] = None,
) -> CacheConfig:
	"""
	Convenience function to get a CacheConfig instance.

	Args:
		cache_dir: Optional base cache directory override
		crawled_cache_file: Optional crawled cache file path override
		vector_db_path: Optional vector DB path override

	Returns:
		CacheConfig instance
	"""
	return CacheConfig(
		cache_dir=cache_dir,
		crawled_cache_file=crawled_cache_file,
		vector_db_path=vector_db_path,
	)
