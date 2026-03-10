"""
Compare PyMuPDF extraction vs Gemini API extraction with same processing strategies.
"""

import json
import os
import re
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import requests
import tempfile

try:
    import pymupdf  # Use newer import name
except ImportError:
    try:
        import fitz as pymupdf  # Fallback to older import name
    except ImportError:
        pymupdf = None

from document_processor import DocumentProcessor


@dataclass
class ExtractionResult:
    """Result of text extraction and processing."""
    source_name: str  # "PyMuPDF" or "Gemini"
    raw_text: str
    normalized_text: str
    chunks: List[str]
    chunk_metadata: List[Dict[str, Any]]
    char_count: int
    chunk_count: int


class ExtractionComparator:
    """Compare PyMuPDF and Gemini extraction on same document with same strategies."""

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.processor = DocumentProcessor()

    @staticmethod
    def _normalize_text(raw_text: str) -> str:
        """Apply same normalization as DocumentProcessor."""
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _split_paragraphs(text: str) -> List[str]:
        """Split text into paragraphs (double newline delimited)."""
        paragraphs = [p.strip() for p in text.split("\n\n")]
        return [p for p in paragraphs if p]

    def _chunk_text(self, text: str) -> List[str]:
        """Apply same chunking strategy as DocumentProcessor."""
        paragraphs = self._split_paragraphs(text)
        if not paragraphs:
            return []

        chunks: List[str] = []
        current = ""

        for paragraph in paragraphs:
            candidate = (
                f"{current}\n\n{paragraph}".strip() if current else paragraph
            )
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

    def extract_with_pymupdf(self, pdf_bytes: bytes, title: str) -> ExtractionResult:
        """Extract text from PDF using PyMuPDF."""
        if pymupdf is None:
            raise ImportError(
                "PyMuPDF not installed. Install with: pip install PyMuPDF"
            )

        print(f"[PyMuPDF] Extracting text from: {title}")
        
        # Create temporary file for PyMuPDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(pdf_bytes)
            temp_path = f.name

        try:
            # Extract text using PyMuPDF
            pdf_doc = pymupdf.open(temp_path)
            raw_text = ""
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                raw_text += page.get_text()
            pdf_doc.close()

            print(f"[PyMuPDF] Extracted {len(raw_text)} characters")

            # Apply same normalization
            normalized = self._normalize_text(raw_text)
            print(f"[PyMuPDF] After normalization: {len(normalized)} characters")

            # Apply same chunking
            chunks = self._chunk_text(normalized)
            print(f"[PyMuPDF] Split into {len(chunks)} chunks")

            # Build chunk metadata
            doc_id = self._build_doc_id(title)
            chunk_metadata = []
            for idx, chunk in enumerate(chunks):
                chunk_metadata.append(
                    {
                        "chunk_id": f"{doc_id}:{idx}",
                        "doc_id": doc_id,
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "char_count": len(chunk),
                    }
                )

            return ExtractionResult(
                source_name="PyMuPDF",
                raw_text=raw_text,
                normalized_text=normalized,
                chunks=chunks,
                chunk_metadata=chunk_metadata,
                char_count=len(normalized),
                chunk_count=len(chunks),
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @staticmethod
    def _build_doc_id(source_id: str) -> str:
        """Build deterministic doc ID."""
        return hashlib.sha1(source_id.encode("utf-8")).hexdigest()

    def extract_from_cache(self, source_url: str) -> Optional[ExtractionResult]:
        """Extract Gemini results from cache."""
        cache_index = self.processor._load_cache_index()
        records = cache_index.get(source_url)
        
        if not records:
            print(f"[Gemini Cache] No cached records found for: {source_url}")
            return None

        print(f"[Gemini Cache] Found {len(records)} records")

        # Reconstruct full text from chunks
        full_text_parts = []
        chunks = []
        chunk_metadata = []

        # Sort by chunk_index to maintain order
        sorted_records = sorted(
            records, key=lambda r: r.get("metadata", {}).get("chunk_index", 0)
        )

        for record in sorted_records:
            text = record.get("text", "")
            chunks.append(text)
            full_text_parts.append(text)
            chunk_metadata.append(record.get("metadata", {}))

        full_text = "\n\n".join(full_text_parts)

        return ExtractionResult(
            source_name="Gemini API",
            raw_text=full_text,  # Already normalized by Gemini
            normalized_text=full_text,
            chunks=chunks,
            chunk_metadata=chunk_metadata,
            char_count=len(full_text),
            chunk_count=len(chunks),
        )

    @staticmethod
    def _jaccard_similarity(text1: str, text2: str) -> float:
        """Calculate Jaccard similarity based on words."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance (character-level)."""
        if len(s1) < len(s2):
            return ExtractionComparator._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    @staticmethod
    def _calculate_similarity(s1: str, s2: str) -> float:
        """Calculate normalized similarity (0-1) based on Levenshtein distance."""
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        
        distance = ExtractionComparator._levenshtein_distance(s1, s2)
        return 1.0 - (distance / max_len)

    def compare_extractions(
        self,
        pymupdf_result: ExtractionResult,
        gemini_result: ExtractionResult,
    ) -> Dict[str, Any]:
        """Compare two extraction results with multiple metrics."""
        
        print("\n" + "="*70)
        print("EXTRACTION COMPARISON REPORT")
        print("="*70)

        # 1. Basic statistics
        print("\n[1] BASIC STATISTICS")
        print(f"  PyMuPDF  - Characters: {pymupdf_result.char_count:,}, Chunks: {pymupdf_result.chunk_count}")
        print(f"  Gemini   - Characters: {gemini_result.char_count:,}, Chunks: {gemini_result.chunk_count}")
        
        char_diff = abs(pymupdf_result.char_count - gemini_result.char_count)
        char_diff_pct = (
            (char_diff / max(pymupdf_result.char_count, gemini_result.char_count)) * 100
            if max(pymupdf_result.char_count, gemini_result.char_count) > 0
            else 0
        )
        print(f"  Char difference: {char_diff:,} ({char_diff_pct:.2f}%)")

        chunk_diff = abs(pymupdf_result.chunk_count - gemini_result.chunk_count)
        print(f"  Chunk difference: {chunk_diff}")

        # 2. Text content comparison
        print("\n[2] TEXT CONTENT COMPARISON")
        full_text_similarity = self._calculate_similarity(
            pymupdf_result.normalized_text, gemini_result.normalized_text
        )
        print(f"  Full text similarity: {full_text_similarity:.4f} (Levenshtein-based)")

        jaccard_sim = self._jaccard_similarity(
            pymupdf_result.normalized_text, gemini_result.normalized_text
        )
        print(f"  Jaccard similarity (word-level): {jaccard_sim:.4f}")

        # 3. Chunk-level comparison
        print("\n[3] CHUNK-LEVEL COMPARISON")
        min_chunks = min(pymupdf_result.chunk_count, gemini_result.chunk_count)
        matching_chunks = 0
        similar_chunks = 0
        similarity_threshold = 0.8

        for i in range(min_chunks):
            py_chunk = pymupdf_result.chunks[i]
            gem_chunk = gemini_result.chunks[i]
            
            chunk_sim = self._calculate_similarity(py_chunk, gem_chunk)
            if chunk_sim == 1.0:
                matching_chunks += 1
            elif chunk_sim >= similarity_threshold:
                similar_chunks += 1

        print(f"  Perfect matches (index {0}-{min_chunks-1}): {matching_chunks}/{min_chunks}")
        print(f"  Similar chunks (>={similarity_threshold}): {similar_chunks}/{min_chunks}")
        match_pct = (matching_chunks / min_chunks * 100) if min_chunks > 0 else 0
        print(f"  Match percentage: {match_pct:.2f}%")

        # 4. Chunking strategy evaluation
        print("\n[4] CHUNKING STRATEGY EVALUATION")
        py_avg_chars = (
            pymupdf_result.char_count / pymupdf_result.chunk_count
            if pymupdf_result.chunk_count > 0
            else 0
        )
        gem_avg_chars = (
            gemini_result.char_count / gemini_result.chunk_count
            if gemini_result.chunk_count > 0
            else 0
        )
        print(f"  PyMuPDF avg chunk size: {py_avg_chars:.0f} chars")
        print(f"  Gemini avg chunk size: {gem_avg_chars:.0f} chars")
        print(f"  Target chunk size: {self.chunk_size} chars")

        # Chunk size distribution
        py_chunk_sizes = [len(c) for c in pymupdf_result.chunks]
        gem_chunk_sizes = [len(c) for c in gemini_result.chunks]

        if py_chunk_sizes:
            print(f"  PyMuPDF - Min: {min(py_chunk_sizes)}, Max: {max(py_chunk_sizes)}, Avg: {sum(py_chunk_sizes)/len(py_chunk_sizes):.0f}")
        if gem_chunk_sizes:
            print(f"  Gemini  - Min: {min(gem_chunk_sizes)}, Max: {max(gem_chunk_sizes)}, Avg: {sum(gem_chunk_sizes)/len(gem_chunk_sizes):.0f}")

        # 5. Quality metrics
        print("\n[5] QUALITY METRICS")
        
        # Check for common words
        py_words = set(pymupdf_result.normalized_text.lower().split())
        gem_words = set(gemini_result.normalized_text.lower().split())
        common_words = len(py_words & gem_words)
        print(f"  Common words: {common_words} (PyMuPDF: {len(py_words)}, Gemini: {len(gem_words)})")

        # Sample chunk comparison (first 3 chunks)
        print("\n[6] SAMPLE CHUNK COMPARISON (First 3 chunks)")
        for i in range(min(3, min_chunks)):
            py_chunk = pymupdf_result.chunks[i][:100]
            gem_chunk = gemini_result.chunks[i][:100]
            sim = self._calculate_similarity(
                pymupdf_result.chunks[i], gemini_result.chunks[i]
            )
            print(f"\n  Chunk {i} (similarity: {sim:.4f})")
            print(f"    PyMuPDF: {py_chunk}...")
            print(f"    Gemini:  {gem_chunk}...")

        # Build summary
        summary = {
            "timestamp": None,  # Could add current timestamp
            "statistics": {
                "pymupdf_chars": pymupdf_result.char_count,
                "gemini_chars": gemini_result.char_count,
                "char_diff": char_diff,
                "char_diff_pct": round(char_diff_pct, 2),
                "pymupdf_chunks": pymupdf_result.chunk_count,
                "gemini_chunks": gemini_result.chunk_count,
                "chunk_diff": chunk_diff,
            },
            "similarity_metrics": {
                "full_text_similarity": round(full_text_similarity, 4),
                "jaccard_similarity": round(jaccard_sim, 4),
            },
            "chunk_comparison": {
                "total_comparable": min_chunks,
                "perfect_matches": matching_chunks,
                "similar_matches": similar_chunks,
                "match_percentage": round(match_pct, 2),
            },
            "chunking": {
                "pymupdf_avg_size": round(py_avg_chars, 2),
                "gemini_avg_size": round(gem_avg_chars, 2),
                "target_size": self.chunk_size,
            },
        }

        return summary

    def save_comparison_report(
        self, summary: Dict[str, Any], output_file: str = "extraction_comparison.json"
    ) -> None:
        """Save comparison report to JSON file."""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nComparison report saved to: {output_file}")

    def save_detailed_comparison(
        self,
        pymupdf_result: ExtractionResult,
        gemini_result: ExtractionResult,
        output_file: str = "extraction_detailed.json",
    ) -> None:
        """Save detailed comparison including sample chunks."""
        detail = {
            "pymupdf": {
                "source": "PyMuPDF",
                "char_count": pymupdf_result.char_count,
                "chunk_count": pymupdf_result.chunk_count,
                "sample_raw": pymupdf_result.raw_text[:500],
                "sample_normalized": pymupdf_result.normalized_text[:500],
                "sample_chunks": pymupdf_result.chunks[:3],
            },
            "gemini": {
                "source": "Gemini API",
                "char_count": gemini_result.char_count,
                "chunk_count": gemini_result.chunk_count,
                "sample_text": gemini_result.normalized_text[:500],
                "sample_chunks": gemini_result.chunks[:3],
            },
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(detail, f, ensure_ascii=False, indent=2)
        print(f"Detailed comparison saved to: {output_file}")


def test_with_cache():
    """Test comparison using cached Gemini data."""
    print("\n" + "="*70)
    print("TESTING WITH CACHED DATA")
    print("="*70)

    comparator = ExtractionComparator()
    
    # Load cache index
    cache_index = comparator.processor._load_cache_index()
    
    if not cache_index:
        print("No cached data found. Run document processor first to build cache.")
        return

    print(f"Found {len(cache_index)} cached sources")

    # Test with first cached source
    source_url = list(cache_index.keys())[0]
    print(f"\nTesting with source: {source_url}")

    # Get Gemini results from cache
    gemini_result = comparator.extract_from_cache(source_url)
    if not gemini_result:
        print("Failed to extract cached Gemini results")
        return

    print(f"Gemini extraction: {gemini_result.chunk_count} chunks, {gemini_result.char_count} chars")

    # For PyMuPDF test, we need the original PDF
    print("\nNote: To test PyMuPDF extraction, we need the original PDF file.")
    print("Skipping PyMuPDF test (PDF file not available).")
    print("\nTo run full comparison:")
    print("1. Download a regulation PDF from Google Drive")
    print("2. Use compare_with_pdf_file() function")


def compare_with_pdf_file(pdf_path: str) -> None:
    """Compare extraction for a specific PDF file."""
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return

    print(f"Loading PDF: {pdf_path}")
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    comparator = ExtractionComparator()
    
    # Extract with PyMuPDF
    pymupdf_result = comparator.extract_with_pymupdf(pdf_bytes, os.path.basename(pdf_path))
    
    print("\n" + "="*70)
    print("Create Gemini extraction using DocumentProcessor if needed")
    print("For now, showing PyMuPDF extraction only:")
    print("="*70)
    print(f"\nPyMuPDF Results:")
    print(f"  Raw text: {len(pymupdf_result.raw_text)} characters")
    print(f"  Normalized: {len(pymupdf_result.normalized_text)} characters")
    print(f"  Chunks: {pymupdf_result.chunk_count}")
    print(f"  Avg chunk size: {pymupdf_result.char_count / pymupdf_result.chunk_count:.0f} chars")
    
    # Save results
    results = {
        "pymupdf_extraction": {
            "char_count": pymupdf_result.char_count,
            "chunk_count": pymupdf_result.chunk_count,
            "avg_chunk_size": pymupdf_result.char_count / pymupdf_result.chunk_count,
            "sample_raw": pymupdf_result.raw_text[:300],
            "sample_normalized": pymupdf_result.normalized_text[:300],
            "chunk_distribution": {
                "min": min([len(c) for c in pymupdf_result.chunks]),
                "max": max([len(c) for c in pymupdf_result.chunks]),
                "avg": pymupdf_result.char_count / pymupdf_result.chunk_count,
            }
        }
    }
    
    with open("pymupdf_extraction_sample.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\nResults saved to: pymupdf_extraction_sample.json")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Test with specific PDF file
        pdf_file = sys.argv[1]
        compare_with_pdf_file(pdf_file)
    else:
        # Test with cached data
        test_with_cache()
