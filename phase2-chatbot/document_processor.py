import hashlib
import json
import os
import re
import tempfile
import time
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import google.generativeai as genai
import fitz
import requests
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager


load_dotenv()


@dataclass
class RegulationSource:
	title: str
	link: str


@dataclass
class ProcessedChunk:
	chunk_id: str
	text: str
	metadata: Dict[str, Any]


@dataclass
class ProcessedDocument:
	doc_id: str
	title: str
	source_url: str
	full_text: str
	chunks: List[ProcessedChunk]
	metadata: Dict[str, Any]


class DocumentProcessor:
	"""Crawl regulation PDFs, preprocess content, and output vector-ready records."""

	def __init__(self) -> None:
		self.regulation_url = os.getenv(
			"REGULATION_PAGE_URL", "https://hcmut.edu.vn/dao-tao/quy-che-quy-dinh"
		)
		self.headless_mode = os.getenv("HEADLESS_MODE", "true").lower() == "true"
		self.wait_time = int(os.getenv("WAIT_TIME", "8"))
		self.pdf_download_timeout = int(os.getenv("PDF_DOWNLOAD_TIMEOUT", "60"))
		self.delay_between_requests = int(os.getenv("DELAY_BETWEEN_REQUESTS", "5"))
		self.state_file = os.getenv(
			"PROCESSOR_STATE_FILE",
			os.path.join(os.path.dirname(__file__), "processor_seen_laws.json"),
		)
		self.cache_file = os.getenv(
			"PROCESSOR_CACHE_FILE",
			os.path.join(os.path.dirname(__file__), ".local_cache", "processed_records.jsonl"),
		)
		self.cache_enabled = os.getenv("PROCESSOR_CACHE_ENABLED", "true").lower() == "true"

		self.chunk_size = int(os.getenv("CHUNK_SIZE", "1200"))
		self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))

		self.main_model_name = os.getenv("MAIN_MODEL", "gemini-3-flash-preview")
		self.fallback_model_name = os.getenv("FALLBACK_MODEL", "gemini-2.5-flash")
		self.pdf_extract_prompt = os.getenv(
			"PDF_EXTRACT_PROMPT",
			(
				"You are extracting university regulation PDFs for RAG indexing. "
				"Return full plain text in Vietnamese, keep section titles and appendix labels, "
				"and remove only decorative repetition (headers/footers/page numbers)."
			),
		)
		self.extraction_mode = os.getenv("EXTRACTION_MODE", "hybrid").lower()
		self.pymupdf_threshold = float(os.getenv("PYMUPDF_QUALITY_THRESHOLD", "0.75"))
		self.expected_min_chars = int(os.getenv("EXPECTED_MIN_CHARS", "1000"))
		self.expected_min_chunks = int(os.getenv("EXPECTED_MIN_CHUNKS", "2"))

		self.gemini_api_key = os.getenv("GEMINI_API_KEY")
		self.verbose_logs = os.getenv("VERBOSE_LOGS", "true").lower() == "true"
		self.main_model = None
		self.fallback_model = None
		self._cache_index: Optional[Dict[str, List[Dict[str, Any]]]] = None
		if self.gemini_api_key:
			genai.configure(api_key=self.gemini_api_key)
			self.main_model = genai.GenerativeModel(self.main_model_name)
			self.fallback_model = genai.GenerativeModel(self.fallback_model_name)

		if self.cache_enabled:
			os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

	def _log(self, message: str) -> None:
		if self.verbose_logs:
			print(f"[document_processor] {message}")

	def _load_seen_links(self) -> List[Dict[str, str]]:
		if not os.path.exists(self.state_file):
			return []
		try:
			with open(self.state_file, "r", encoding="utf-8") as f:
				return json.load(f)
		except Exception:
			return []

	def _save_seen_links(self, links: List[Dict[str, str]]) -> None:
		with open(self.state_file, "w", encoding="utf-8") as f:
			json.dump(links, f, ensure_ascii=False, indent=2)

	def _load_cache_index(self) -> Dict[str, List[Dict[str, Any]]]:
		if self._cache_index is not None:
			return self._cache_index

		index: Dict[str, List[Dict[str, Any]]] = {}
		if not self.cache_enabled or not os.path.exists(self.cache_file):
			self._cache_index = index
			return index

		try:
			with open(self.cache_file, "r", encoding="utf-8") as f:
				for line in f:
					line = line.strip()
					if not line:
						continue
					entry = json.loads(line)
					source_url = entry.get("source_url")
					records = entry.get("records", [])
					if source_url and records:
						index[source_url] = records
		except Exception as exc:
			self._log(f"Warning: failed to load cache file ({exc}).")

		self._cache_index = index
		self._log(f"Cache loaded with {len(index)} source entries.")
		return index

	def _get_cached_records(self, source_url: str) -> Optional[List[Dict[str, Any]]]:
		cache_index = self._load_cache_index()
		return cache_index.get(source_url)

	def _append_cache_entry(self, source_url: str, records: List[Dict[str, Any]]) -> None:
		if not self.cache_enabled or not records:
			return

		cache_index = self._load_cache_index()
		if source_url in cache_index:
			return

		entry = {"source_url": source_url, "records": records}
		with open(self.cache_file, "a", encoding="utf-8") as f:
			f.write(json.dumps(entry, ensure_ascii=False) + "\n")

		cache_index[source_url] = records
		self._log(f"Cached records for source: {source_url}")

	def get_cache_stats(self) -> Dict[str, int]:
		cache_index = self._load_cache_index()
		total_records = sum(len(records) for records in cache_index.values())
		return {
			"cache_entries": len(cache_index),
			"cache_records": total_records,
			"cache_file_exists": int(os.path.exists(self.cache_file)),
		}

	def clear_cache(self) -> None:
		if os.path.exists(self.cache_file):
			os.remove(self.cache_file)
			self._log(f"Cleared cache file: {self.cache_file}")
		self._cache_index = {}

	def dedupe_cache_file(self) -> Dict[str, int]:
		if not self.cache_enabled or not os.path.exists(self.cache_file):
			self._cache_index = {}
			return {"entries_before": 0, "entries_after": 0, "records_after": 0}

		entries_by_url: Dict[str, List[Dict[str, Any]]] = {}
		entries_before = 0
		with open(self.cache_file, "r", encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				entries_before += 1
				entry = json.loads(line)
				source_url = entry.get("source_url")
				records = entry.get("records", [])
				if source_url and records:
					entries_by_url[source_url] = self._dedupe_records_by_id(records)

		with open(self.cache_file, "w", encoding="utf-8") as f:
			for source_url, records in entries_by_url.items():
				f.write(json.dumps({"source_url": source_url, "records": records}, ensure_ascii=False) + "\n")

		self._cache_index = entries_by_url
		records_after = sum(len(records) for records in entries_by_url.values())
		self._log(
			f"Deduped cache: entries {entries_before} -> {len(entries_by_url)}, records={records_after}"
		)
		return {
			"entries_before": entries_before,
			"entries_after": len(entries_by_url),
			"records_after": records_after,
		}

	def rebuild_cache_from_state(self, limit: Optional[int] = None) -> Dict[str, int]:
		seen_links = self._load_seen_links()
		if not seen_links:
			return {
				"state_total": 0,
				"missing_before": 0,
				"rebuilt": 0,
				"failed": 0,
			}

		unique_state: List[RegulationSource] = []
		seen_urls = set()
		for item in seen_links:
			link = item.get("link", "")
			title = item.get("title", "")
			if not link or link in seen_urls:
				continue
			seen_urls.add(link)
			unique_state.append(RegulationSource(title=title, link=link))

		cache_index = self._load_cache_index()
		missing = [law for law in unique_state if law.link not in cache_index]
		if limit is not None:
			missing = missing[:limit]

		rebuilt = 0
		failed = 0
		self._log(
			f"Rebuilding cache from state: total_state={len(unique_state)}, missing={len(missing)}"
		)

		for idx, law in enumerate(missing, start=1):
			self._log(f"[rebuild {idx}/{len(missing)}] {law.title or law.link}")
			doc = self.process_single_regulation(law)
			if not doc:
				failed += 1
				continue

			records = self.to_vector_store_records([doc])
			self._append_cache_entry(law.link, records)
			rebuilt += 1

			if self.delay_between_requests > 0:
				time.sleep(self.delay_between_requests)

		return {
			"state_total": len(unique_state),
			"missing_before": len(missing),
			"rebuilt": rebuilt,
			"failed": failed,
		}

	def get_cached_records_for_indexing(self, limit_sources: Optional[int] = None) -> List[Dict[str, Any]]:
		"""Return flattened cached records only (no crawling/download/extraction)."""
		cache_index = self._load_cache_index()
		source_urls = list(cache_index.keys())
		if limit_sources is not None:
			source_urls = source_urls[:limit_sources]

		all_records: List[Dict[str, Any]] = []
		seen_ids = set()
		for source_url in source_urls:
			records = cache_index.get(source_url, [])
			for record in records:
				record_id = record.get("id")
				if not record_id or record_id in seen_ids:
					continue
				seen_ids.add(record_id)
				all_records.append(record)

		self._log(
			f"Loaded {len(all_records)} unique cached records from {len(source_urls)} sources."
		)
		return all_records

	@staticmethod
	def _dedupe_regulations_by_link(regulations: List[RegulationSource]) -> List[RegulationSource]:
		seen_links = set()
		unique: List[RegulationSource] = []
		for law in regulations:
			if law.link in seen_links:
				continue
			seen_links.add(law.link)
			unique.append(law)
		return unique

	@staticmethod
	def _dedupe_records_by_id(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		seen_ids = set()
		unique_records: List[Dict[str, Any]] = []
		for record in records:
			record_id = record.get("id")
			if not record_id or record_id in seen_ids:
				continue
			seen_ids.add(record_id)
			unique_records.append(record)
		return unique_records

	@staticmethod
	def _extract_drive_id(url: str) -> Optional[str]:
		match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
		return match.group(1) if match else None

	def crawl_regulations(self) -> List[RegulationSource]:
		self._log(f"Starting crawl: {self.regulation_url}")
		options = Options()
		if self.headless_mode:
			options.add_argument("--headless")
		options.add_argument("--log-level=3")
		options.add_argument("--disable-gpu")

		driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
		regulations: List[RegulationSource] = []

		try:
			driver.get(self.regulation_url)
			self._log(f"Page loaded, waiting {self.wait_time}s for dynamic content...")
			time.sleep(self.wait_time)

			elements = driver.find_elements(By.CSS_SELECTOR, "div.sub-link a")
			for el in elements:
				href = el.get_attribute("href")
				text = (el.text or "").strip()
				if not text:
					text = (el.get_attribute("textContent") or "").strip()
				if href and ("drive.google.com" in href or "docs.google.com" in href):
					regulations.append(RegulationSource(title=text, link=href))
		finally:
			driver.quit()

		self._log(f"Crawl completed, found {len(regulations)} regulation links.")

		return regulations

	def get_new_regulations(self, regulations: List[RegulationSource]) -> List[RegulationSource]:
		seen_links = self._load_seen_links()
		seen_urls = {law.get("link", "") for law in seen_links}
		return [law for law in regulations if law.link not in seen_urls]

	def mark_processed(self, regulations: List[RegulationSource]) -> None:
		if not regulations:
			return
		seen_links = self._load_seen_links()
		existing = {law.get("link", "") for law in seen_links}
		for law in regulations:
			if law.link not in existing:
				seen_links.append({"title": law.title, "link": law.link})
		self._save_seen_links(seen_links)

	def download_pdf(self, drive_url: str) -> Optional[bytes]:
		file_id = self._extract_drive_id(drive_url)
		if not file_id:
			self._log(f"Skip download, invalid Google Drive URL: {drive_url}")
			return None

		download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
		try:
			self._log(f"Downloading PDF for file id {file_id[:8]}...")
			response = requests.get(download_url, timeout=self.pdf_download_timeout)
			if response.status_code != 200:
				self._log(f"Download failed with status {response.status_code}.")
				return None
			self._log(f"Download success ({len(response.content)} bytes).")
			return response.content
		except Exception as exc:
			self._log(f"Download error: {exc}")
			return None

	def _extract_text_with_gemini(self, pdf_bytes: bytes, title: str) -> str:
		if not self.main_model or not self.fallback_model:
			raise ValueError("Missing GEMINI_API_KEY. Cannot preprocess PDF text for vector indexing.")

		with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
			temp_file.write(pdf_bytes)
			temp_path = temp_file.name

		try:
			self._log(f"Uploading PDF to Gemini for text extraction: {title}")
			uploaded_doc = genai.upload_file(path=temp_path, mime_type="application/pdf")
			instruction = (
				f"{self.pdf_extract_prompt}\n"
				f"Document title: {title}\n"
				"Output only extracted text, no explanations."
			)
			try:
				self._log(f"Extracting text with main model: {self.main_model_name}")
				response = self.main_model.generate_content([instruction, uploaded_doc])
			except Exception as exc:
				self._log(f"Main model failed ({exc}), retrying with {self.fallback_model_name}")
				response = self.fallback_model.generate_content([instruction, uploaded_doc])

			text = (response.text or "").strip()
			self._log(f"Text extraction complete ({len(text)} chars).")
			genai.delete_file(uploaded_doc.name)
			return text
		finally:
			if os.path.exists(temp_path):
				os.remove(temp_path)

	def _extract_text_with_pymupdf(self, pdf_bytes: bytes, title: str) -> str:
		self._log(f"Extracting text with PyMuPDF: {title}")
		doc = fitz.open(stream=pdf_bytes, filetype="pdf")
		try:
			pages: List[str] = []
			for page in doc:
				pages.append(page.get_text("text"))
			text = "\n".join(pages).strip()
			self._log(f"PyMuPDF extraction complete ({len(text)} chars).")
			return text
		finally:
			doc.close()

	def _score_extraction_quality(self, text: str) -> Dict[str, float]:
		char_count = len(text)
		chunk_count = len(self._chunk_text(text)) if text else 0

		content_score = (
			min(1.0, char_count / self.expected_min_chars)
			if self.expected_min_chars > 0
			else 1.0
		)

		avg_chunk_size = (char_count / chunk_count) if chunk_count > 0 else 0
		target_size = self.chunk_size if self.chunk_size > 0 else 1
		size_diff_ratio = abs(avg_chunk_size - target_size) / target_size
		chunking_score = max(0.0, 1.0 - size_diff_ratio)

		structure_score = (
			min(1.0, chunk_count / self.expected_min_chunks)
			if self.expected_min_chunks > 0
			else 1.0
		)

		overall = content_score * 0.4 + chunking_score * 0.3 + structure_score * 0.3
		return {
			"content_score": round(content_score, 4),
			"chunking_score": round(chunking_score, 4),
			"structure_score": round(structure_score, 4),
			"overall": round(overall, 4),
			"char_count": float(char_count),
			"chunk_count": float(chunk_count),
		}

	def _extract_text_hybrid(self, pdf_bytes: bytes, title: str) -> tuple[str, str, Dict[str, float]]:
		"""Prefer PyMuPDF (free), fallback to Gemini when quality is low or PyMuPDF fails."""
		if self.extraction_mode == "gemini":
			text = self._extract_text_with_gemini(pdf_bytes, title)
			return text, "gemini", {"overall": 1.0}

		if self.extraction_mode == "pymupdf_only":
			text = self._extract_text_with_pymupdf(pdf_bytes, title)
			score = self._score_extraction_quality(text)
			return text, "pymupdf", score

		try:
			pymupdf_text = self._extract_text_with_pymupdf(pdf_bytes, title)
			pymupdf_score = self._score_extraction_quality(pymupdf_text)
			self._log(
				"PyMuPDF quality score "
				f"{pymupdf_score['overall']:.2f} (threshold={self.pymupdf_threshold:.2f})"
			)

			if pymupdf_score["overall"] >= self.pymupdf_threshold:
				self._log("Using PyMuPDF result.")
				return pymupdf_text, "pymupdf", pymupdf_score

			self._log("PyMuPDF quality below threshold, falling back to Gemini.")
		except Exception as exc:
			self._log(f"PyMuPDF failed ({exc}), falling back to Gemini.")

		gemini_text = self._extract_text_with_gemini(pdf_bytes, title)
		gemini_score = self._score_extraction_quality(gemini_text)
		return gemini_text, "gemini", gemini_score

	@staticmethod
	def _normalize_text(raw_text: str) -> str:
		text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
		text = re.sub(r"[ \t]+", " ", text)
		text = re.sub(r"\n{3,}", "\n\n", text)
		return text.strip()

	@staticmethod
	def _split_paragraphs(text: str) -> List[str]:
		paragraphs = [p.strip() for p in text.split("\n\n")]
		return [p for p in paragraphs if p]

	def _chunk_text(self, text: str) -> List[str]:
		paragraphs = self._split_paragraphs(text)
		if not paragraphs:
			return []

		chunks: List[str] = []
		current = ""

		for paragraph in paragraphs:
			candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
			if len(candidate) <= self.chunk_size:
				current = candidate
				continue

			if current:
				chunks.append(current)

			if len(paragraph) <= self.chunk_size:
				current = paragraph
				continue

			start = 0
			while start < len(paragraph):
				end = min(start + self.chunk_size, len(paragraph))
				chunks.append(paragraph[start:end])
				if end == len(paragraph):
					break
				start = max(end - self.chunk_overlap, start + 1)
			current = ""

		if current:
			chunks.append(current)

		return chunks

	@staticmethod
	def _build_doc_id(source_url: str) -> str:
		return hashlib.sha1(source_url.encode("utf-8")).hexdigest()

	def process_single_regulation(self, law: RegulationSource) -> Optional[ProcessedDocument]:
		self._log(f"Processing regulation: {law.title}")
		pdf_bytes = self.download_pdf(law.link)
		if not pdf_bytes:
			self._log("Skip regulation because PDF download failed.")
			return None

		raw_text, extraction_method, extraction_score = self._extract_text_hybrid(pdf_bytes, law.title)
		cleaned_text = self._normalize_text(raw_text)
		if not cleaned_text:
			self._log("Skip regulation because extracted text is empty.")
			return None

		doc_id = self._build_doc_id(law.link)
		chunk_texts = self._chunk_text(cleaned_text)
		self._log(f"Chunked into {len(chunk_texts)} chunks.")

		chunks: List[ProcessedChunk] = []
		total_chunks = len(chunk_texts)
		for idx, chunk_text in enumerate(chunk_texts):
			chunk_id = f"{doc_id}:{idx}"
			chunks.append(
				ProcessedChunk(
					chunk_id=chunk_id,
					text=chunk_text,
					metadata={
						"doc_id": doc_id,
						"source_url": law.link,
						"title": law.title,
						"chunk_index": idx,
						"total_chunks": total_chunks,
						"extraction_method": extraction_method,
					},
				)
			)

		return ProcessedDocument(
			doc_id=doc_id,
			title=law.title,
			source_url=law.link,
			full_text=cleaned_text,
			chunks=chunks,
			metadata={
				"source": "hcmut-regulations",
				"crawl_url": self.regulation_url,
				"chunk_size": self.chunk_size,
				"chunk_overlap": self.chunk_overlap,
				"extraction_method": extraction_method,
				"extraction_score": extraction_score,
			},
		)

	def process_regulations(
		self,
		only_new: bool = True,
		max_documents: Optional[int] = None,
		mark_as_processed: bool = True,
	) -> List[ProcessedDocument]:
		regulations = self.crawl_regulations()
		targets = self.get_new_regulations(regulations) if only_new else regulations
		self._log(f"Selected {len(targets)} regulations for processing (only_new={only_new}).")

		if max_documents is not None:
			targets = targets[:max_documents]

		processed_docs: List[ProcessedDocument] = []
		processed_sources: List[RegulationSource] = []

		for law in targets:
			doc = self.process_single_regulation(law)
			if doc:
				processed_docs.append(doc)
				processed_sources.append(law)
				self._log(f"Processed {len(processed_docs)}/{len(targets)} regulations.")
			else:
				self._log(f"Failed processing: {law.title}")
			time.sleep(self.delay_between_requests)

		if mark_as_processed and processed_sources:
			self.mark_processed(processed_sources)
			self._log(f"Marked {len(processed_sources)} regulations as processed.")

		self._log(f"Pipeline complete: {len(processed_docs)} documents ready for vector store.")

		return processed_docs

	@staticmethod
	def to_vector_store_records(documents: List[ProcessedDocument]) -> List[Dict[str, Any]]:
		records: List[Dict[str, Any]] = []
		for doc in documents:
			for chunk in doc.chunks:
				records.append(
					{
						"id": chunk.chunk_id,
						"text": chunk.text,
						"metadata": chunk.metadata,
					}
				)
		return records


def process_documents_for_vector_store(
	only_new: bool = True,
	max_documents: Optional[int] = None,
) -> List[Dict[str, Any]]:
	"""Fetch vector-ready records with local cache to survive interrupted runs."""
	processor = DocumentProcessor()

	regulations = processor.crawl_regulations()
	targets = processor.get_new_regulations(regulations) if only_new else regulations
	unique_targets = processor._dedupe_regulations_by_link(targets)
	if len(unique_targets) != len(targets):
		processor._log(f"Removed {len(targets) - len(unique_targets)} duplicate links from crawl result.")
	targets = unique_targets
	if max_documents is not None:
		targets = targets[:max_documents]

	processor._log(
		f"Preparing vector records from {len(targets)} targets (cache_enabled={processor.cache_enabled})."
	)

	all_records: List[Dict[str, Any]] = []
	seen_record_ids = set()
	for idx, law in enumerate(targets, start=1):
		processor._log(f"[{idx}/{len(targets)}] Handling: {law.title}")
		cached_records = processor._get_cached_records(law.link)
		if cached_records:
			processor._log(f"Cache hit: reusing {len(cached_records)} records.")
			for record in cached_records:
				record_id = record.get("id")
				if record_id and record_id not in seen_record_ids:
					seen_record_ids.add(record_id)
					all_records.append(record)
			processor.mark_processed([law])
			continue

		doc = processor.process_single_regulation(law)
		if not doc:
			processor._log("Skipping due to processing failure.")
			continue

		records = processor.to_vector_store_records([doc])
		for record in records:
			record_id = record.get("id")
			if record_id and record_id not in seen_record_ids:
				seen_record_ids.add(record_id)
				all_records.append(record)
		processor._append_cache_entry(law.link, records)
		processor.mark_processed([law])

		if processor.delay_between_requests > 0:
			time.sleep(processor.delay_between_requests)

	processor._log(f"Prepared {len(all_records)} total records for vector store.")
	return all_records


def _run_cache_maintenance_command() -> None:
	parser = argparse.ArgumentParser(description="Document processor local cache maintenance")
	parser.add_argument("--cache-stats", action="store_true", help="Show cache file stats")
	parser.add_argument("--clear-cache", action="store_true", help="Delete local cache file")
	parser.add_argument("--dedupe-cache", action="store_true", help="Dedupe local cache by source URL")
	parser.add_argument(
		"--rebuild-cache-from-state",
		action="store_true",
		help="Backfill cache entries for links in state file that are missing from cache",
	)
	parser.add_argument(
		"--limit",
		type=int,
		default=None,
		help="Optional max number of state entries to process when rebuilding cache",
	)
	args = parser.parse_args()

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

	parser.print_help()


if __name__ == "__main__":
	_run_cache_maintenance_command()
