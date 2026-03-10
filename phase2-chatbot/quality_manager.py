#!/usr/bin/env python3
"""
Practical demo: Using extraction comparison for quality assurance.
Shows hybrid approach, batch processing, and quality scoring.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from compare_extractions import ExtractionComparator, ExtractionResult
from document_processor import DocumentProcessor


class ExtractionQualityManager:
    """Manages extraction quality with hybrid approach and scoring."""

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200):
        self.comparator = ExtractionComparator(chunk_size, chunk_overlap)
        self.processor = DocumentProcessor()
        self.results_log: List[Dict] = []

    def score_extraction(
        self,
        result: ExtractionResult,
        expected_min_chars: int = 1000,
        expected_min_chunks: int = 2,
    ) -> Dict[str, float]:
        """Score extraction quality on multiple dimensions."""
        
        scores = {}
        
        # 1. Content completeness
        scores["content_score"] = min(
            1.0,
            result.char_count / expected_min_chars
        ) if expected_min_chars > 0 else 1.0
        
        # 2. Chunking appropriateness
        avg_chunk_size = (
            result.char_count / result.chunk_count
            if result.chunk_count > 0
            else 0
        )
        target_size = self.comparator.chunk_size
        size_diff_ratio = abs(avg_chunk_size - target_size) / target_size
        scores["chunking_score"] = max(0.0, 1.0 - size_diff_ratio)
        
        # 3. Structure preservation (based on chunk count)
        scores["structure_score"] = min(
            1.0,
            result.chunk_count / expected_min_chunks
        ) if expected_min_chunks > 0 else 1.0
        
        # Overall score (weighted average)
        scores["overall"] = (
            scores["content_score"] * 0.4 +
            scores["chunking_score"] * 0.3 +
            scores["structure_score"] * 0.3
        )
        
        return scores

    def hybrid_extract(
        self,
        pdf_bytes: bytes,
        title: str,
        pymupdf_threshold: float = 0.75,
    ) -> tuple[ExtractionResult, str]:
        """
        Hybrid extraction: Try PyMuPDF first, fallback to Gemini if needed.
        
        Returns:
            (result, method_used)
        """
        print(f"🔄 Extracting: {title}")
        
        # Try PyMuPDF (fast, free)
        try:
            pymupdf_result = self.comparator.extract_with_pymupdf(pdf_bytes, title)
            pymupdf_score = self.score_extraction(pymupdf_result)
            print(f"  PyMuPDF score: {pymupdf_score['overall']:.2%}")
            
            if pymupdf_score["overall"] >= pymupdf_threshold:
                print(f"  ✅ Using PyMuPDF (score: {pymupdf_score['overall']:.2%})")
                self.results_log.append({
                    "title": title,
                    "method": "PyMuPDF",
                    "score": pymupdf_score,
                })
                return pymupdf_result, "PyMuPDF"
            else:
                print(f"  ⚠️ PyMuPDF quality below threshold, falling back to Gemini...")
        
        except Exception as e:
            print(f"  ❌ PyMuPDF failed: {e}, using Gemini...")
        
        # Fallback: Gemini API
        # Note: In real scenario, you'd extract with Gemini here
        # For demo, we skip and just note the fallback would happen
        print(f"  Using Gemini API (would extract via API)")
        return None, "Gemini API (skipped in demo)"

    def batch_compare(self, pdf_directory: str) -> Dict:
        """Compare multiple PDFs to get quality statistics."""
        results = {
            "total_documents": 0,
            "pymupdf_only": 0,
            "gemini_fallback": 0,
            "avg_score": 0.0,
            "quality_distribution": {"excellent": 0, "good": 0, "fair": 0, "poor": 0},
            "details": []
        }
        
        pdf_files = list(Path(pdf_directory).glob("*.pdf"))
        if not pdf_files:
            print(f"No PDFs found in {pdf_directory}")
            return results
        
        print(f"\n📊 Processing {len(pdf_files)} documents...\n")
        
        for pdf_path in pdf_files:
            try:
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                
                result, method = self.hybrid_extract(pdf_bytes, pdf_path.name)
                
                if result:
                    scores = self.score_extraction(result)
                    results["details"].append({
                        "file": pdf_path.name,
                        "method": method,
                        "scores": scores,
                    })
                    results["total_documents"] += 1
                    
                    if method == "PyMuPDF":
                        results["pymupdf_only"] += 1
                    else:
                        results["gemini_fallback"] += 1
                    
                    # Categorize quality
                    overall = scores["overall"]
                    if overall >= 0.9:
                        results["quality_distribution"]["excellent"] += 1
                    elif overall >= 0.75:
                        results["quality_distribution"]["good"] += 1
                    elif overall >= 0.6:
                        results["quality_distribution"]["fair"] += 1
                    else:
                        results["quality_distribution"]["poor"] += 1
                        
            except Exception as e:
                print(f"  ❌ Failed to process {pdf_path.name}: {e}")
        
        # Calculate statistics
        if results["details"]:
            avg_score = sum(d["scores"]["overall"] for d in results["details"]) / len(results["details"])
            results["avg_score"] = round(avg_score, 4)
        
        return results

    def print_summary(self, results: Dict) -> None:
        """Print a nice summary of batch results."""
        print("\n" + "="*70)
        print("📈 QUALITY ASSESSMENT SUMMARY")
        print("="*70)
        print(f"Total documents processed: {results['total_documents']}")
        print(f"PyMuPDF only: {results['pymupdf_only']} ({results['pymupdf_only']/max(results['total_documents'], 1)*100:.1f}%)")
        print(f"Gemini fallback: {results['gemini_fallback']} ({results['gemini_fallback']/max(results['total_documents'], 1)*100:.1f}%)")
        print(f"Average quality score: {results['avg_score']:.2%}")
        
        print("\nQuality Distribution:")
        for quality, count in results["quality_distribution"].items():
            pct = count / max(results['total_documents'], 1) * 100
            print(f"  {quality.capitalize()}: {count} ({pct:.1f}%)")
        
        if results["details"]:
            print("\nDocument Details:")
            for detail in results["details"]:
                print(f"  - {detail['file']}")
                print(f"    Method: {detail['method']}")
                print(f"    Overall Score: {detail['scores']['overall']:.2%}")


def demo_single_pdf(pdf_path: str) -> None:
    """Demo: Analyze a single PDF."""
    print("\n" + "="*70)
    print("📄 SINGLE DOCUMENT ANALYSIS")
    print("="*70)
    
    manager = ExtractionQualityManager()
    
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    
    result, method = manager.hybrid_extract(pdf_bytes, Path(pdf_path).name)
    
    if result:
        scores = manager.score_extraction(result)
        
        print("\n" + "-"*70)
        print("DETAILED SCORES:")
        print("-"*70)
        print(f"Content Score: {scores['content_score']:.2%}")
        print(f"Chunking Score: {scores['chunking_score']:.2%}")
        print(f"Structure Score: {scores['structure_score']:.2%}")
        print(f"Overall Quality: {scores['overall']:.2%} ⭐")
        
        print("\nEXTRACTION STATISTICS:")
        print(f"Total characters: {result.char_count:,}")
        print(f"Number of chunks: {result.chunk_count}")
        print(f"Average chunk size: {result.char_count/result.chunk_count:.0f} chars")


def demo_batch_comparison() -> None:
    """Demo: Batch compare multiple PDFs (simulated)."""
    print("\n" + "="*70)
    print("🔄 BATCH COMPARISON DEMO")
    print("="*70)
    
    manager = ExtractionQualityManager()
    
    # Simulate batch results
    simulated_results = {
        "total_documents": 5,
        "pymupdf_only": 4,
        "gemini_fallback": 1,
        "avg_score": 0.82,
        "quality_distribution": {
            "excellent": 2,
            "good": 2,
            "fair": 1,
            "poor": 0
        },
        "details": [
            {"file": "regulation_001.pdf", "method": "PyMuPDF", "scores": {"overall": 0.85}},
            {"file": "regulation_002.pdf", "method": "PyMuPDF", "scores": {"overall": 0.88}},
            {"file": "regulation_003.pdf", "method": "Gemini API", "scores": {"overall": 0.92}},
            {"file": "regulation_004.pdf", "method": "PyMuPDF", "scores": {"overall": 0.79}},
            {"file": "regulation_005.pdf", "method": "PyMuPDF", "scores": {"overall": 0.80}},
        ]
    }
    
    manager.print_summary(simulated_results)


def demo_comparison_metrics() -> None:
    """Demo: Show comparison metrics between PyMuPDF and Gemini."""
    print("\n" + "="*70)
    print("📊 COMPARISON METRICS DEMONSTRATION")
    print("="*70)
    
    comparator = ExtractionComparator()
    
    # Example texts
    text1 = """
    TRƯỜNG ĐẠI HỌC BÁCH KHOA
    HỘI ĐỒNG HỌC VỤ
    Số: 280/TB-ĐHBK-ĐT
    CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM
    Độc lập - Tự do - Hạnh phúc
    """
    
    text2 = """
    TRƯỜNG ĐẠI HỌC BÁCH KHOA
    HỘI ĐỒNG HỌC VỤ
    CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM
    Độc lập - Tự do - Hạnh phúc
    """
    
    levenshtein_sim = comparator._calculate_similarity(text1, text2)
    jaccard_sim = comparator._jaccard_similarity(text1, text2)
    
    print(f"Text 1 length: {len(text1)} chars")
    print(f"Text 2 length: {len(text2)} chars")
    print(f"\nLevenshtein Similarity: {levenshtein_sim:.4f}")
    print(f"Jaccard Similarity: {jaccard_sim:.4f}")
    
    print("\n✅ Lower scores (even 85%+) expected due to formatting differences")


if __name__ == "__main__":
    import sys
    
    print("\n🎯 EXTRACTION QUALITY MANAGEMENT DEMO\n")
    
    if len(sys.argv) > 1:
        # Test with specific PDF
        pdf_path = sys.argv[1]
        if os.path.exists(pdf_path):
            demo_single_pdf(pdf_path)
        else:
            print(f"File not found: {pdf_path}")
    else:
        # Show all demos
        demo_comparison_metrics()
        demo_batch_comparison()
        
        print("\n" + "="*70)
        print("💡 HOW TO USE IN PRODUCTION:")
        print("="*70)
        print("""
1. Initialize QualityManager:
   manager = ExtractionQualityManager()
   
2. Extract with hybrid approach:
   result, method = manager.hybrid_extract(pdf_bytes, title)
   
3. Score the extraction:
   scores = manager.score_extraction(result)
   
4. Make decision based on score:
   if scores['overall'] >= 0.75:
       use_result(result)
   else:
       request_manual_review(pdf_path)

5. Batch process for reporting:
   results = manager.batch_compare('./pdfs/')
   manager.print_summary(results)
        """)
