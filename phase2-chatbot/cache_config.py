"""Centralized cache configuration for HCMUT Regulation Bot."""

import os


class CacheConfig:
	"""Manages cache configuration with environment variable defaults."""

	def __init__(self):
		"""Initialize cache paths from environment variables."""
		self.cache_dir = os.getenv("CACHE_DIR", os.path.join(os.path.dirname(__file__), "..", "cache"))
		self.cache_dir = os.path.abspath(self.cache_dir)

		default_crawled = os.path.join(self.cache_dir, "processed_records.jsonl")
		self.crawled_cache_file = os.getenv("CRAWLED_CACHE_FILE", default_crawled)
		self.crawled_cache_file = os.path.abspath(self.crawled_cache_file)
		self.collection_name = os.getenv("MILVUS_COLLECTION_NAME", "hcmut_regulations")

	def ensure_directories_exist(self) -> None:
		"""Create cache directories if they don't exist."""
		os.makedirs(self.cache_dir, exist_ok=True)
		os.makedirs(os.path.dirname(self.crawled_cache_file), exist_ok=True)

	def get_crawled_cache_dir(self) -> str:
		"""Get directory containing the crawled cache file."""
		return os.path.dirname(self.crawled_cache_file)

	def to_dict(self) -> dict:
		"""Return configuration as dictionary for logging/debugging."""
		return {
			"cache_dir": self.cache_dir,
			"crawled_cache_file": self.crawled_cache_file,
			"collection_name": self.collection_name,
		}

	def __repr__(self) -> str:
		"""String representation for debugging."""
		return f"CacheConfig({self.to_dict()})"
def get_cache_config() -> CacheConfig:
	"""Convenience function to get a CacheConfig instance."""
	return CacheConfig()
