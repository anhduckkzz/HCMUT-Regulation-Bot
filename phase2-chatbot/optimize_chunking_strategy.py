#!/usr/bin/env python3
"""
Optimize chunking strategy independently without re-crawling or re-indexing.

This script loads pre-cached regulation data and tests different chunking strategies
to find the optimal parameters for RAG retrieval and reranking.

Usage:
    python optimize_chunking_strategy.py [--chunk-sizes 1000,1200,1500] [--overlaps 100,200] [--output report.json]
"""

import argparse
import json
import os
import re
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from pathlib import Path


@dataclass
class ChunkingConfig:
    """Configuration for a chunking strategy."""
    chunk_size: int
    chunk_overlap: int
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"size_{self.chunk_size}_overlap_{self.chunk_overlap}"


@dataclass
class ChunkingResult:
    """Result metrics for a chunking strategy."""
    config: ChunkingConfig
    total_chunks: int
    total_text_length: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    median_chunk_size: float
    chunks_under_threshold: int  # chunks < overlap size
    chunks_over_limit: int  # chunks > chunk_size
    cumulative_redundancy_pct: float  # % of text repeated due to overlap
    processing_time_ms: float


@dataclass
class DocumentMetrics:
    """Metrics for a document after chunking."""
    doc_id: str
    title: str
    source_url: str
    original_text_length: int
    chunking_results: Dict[str, ChunkingResult]


class CachedDataLoader:
    """Load and parse cached regulation data."""
    
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.documents: Dict[str, Dict] = {}  # doc_id -> document data
        
    def load(self) -> Dict[str, Dict]:
        """Load cached records and reconstruct documents."""
        if not os.path.exists(self.cache_file):
            raise FileNotFoundError(f"Cache file not found: {self.cache_file}")
        
        print(f"Loading cache from: {self.cache_file}")
        
        documents_by_id = defaultdict(lambda: {
            "title": "",
            "source_url": "",
            "chunks": []
        })
        
        line_count = 0
        with open(self.cache_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    source_url = entry.get("source_url", "")
                    records = entry.get("records", [])
                    
                    for record in records:
                        chunk_id = record.get("id", "")
                        doc_id = chunk_id.split(":")[0] if ":" in chunk_id else ""
                        
                        metadata = record.get("metadata", {})
                        doc_data = documents_by_id[doc_id]
                        doc_data["title"] = metadata.get("title", "")
                        doc_data["source_url"] = metadata.get("source_url", source_url)
                        doc_data["chunks"].append(record)
                    
                    line_count += 1
                except json.JSONDecodeError as e:
                    print(f"Warning: failed to parse line {line_count + 1}: {e}")
                    continue
        
        # Sort chunks by index within each document
        for doc_id, doc_data in documents_by_id.items():
            doc_data["chunks"].sort(
                key=lambda c: c.get("metadata", {}).get("chunk_index", 0)
            )
        
        self.documents = dict(documents_by_id)
        print(f"Loaded {len(self.documents)} documents from {line_count} cache entries")
        return self.documents
    
    def reconstruct_full_text(self, doc_id: str) -> Optional[str]:
        """Reconstruct full text from chunks (without overlaps)."""
        if doc_id not in self.documents:
            return None
        
        doc_data = self.documents[doc_id]
        chunks = doc_data["chunks"]
        
        if not chunks:
            return None
        
        # Take the first chunk fully, then extract non-overlapping portions
        full_text = chunks[0].get("text", "")
        
        for i in range(1, len(chunks)):
            current_chunk = chunks[i].get("text", "")
            prev_chunk = chunks[i-1].get("text", "")
            
            # Find overlap between end of prev chunk and start of current chunk
            prev_end = prev_chunk[-500:] if len(prev_chunk) > 500 else prev_chunk
            overlap_start = 0
            
            for j in range(len(prev_end), 0, -1):
                if current_chunk.startswith(prev_end[-j:]):
                    overlap_start = j
                    break
            
            # Add non-overlapping portion
            if overlap_start > 0:
                full_text += current_chunk[overlap_start:]
            else:
                full_text += current_chunk
        
        return full_text


class ChunkingStrategyOptimizer:
    """Test different chunking strategies and collect metrics."""
    
    def __init__(self):
        self.results: Dict[str, List[ChunkingResult]] = {}  # doc_id -> results
        
    @staticmethod
    def _normalize_text(raw_text: str) -> str:
        """Normalize text."""
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    
    @staticmethod
    def _split_paragraphs(text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = [p.strip() for p in text.split("\n\n")]
        return [p for p in paragraphs if p]
    
    @staticmethod
    def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Apply chunking strategy."""
        paragraphs = ChunkingStrategyOptimizer._split_paragraphs(text)
        if not paragraphs:
            return []
        
        chunks: List[str] = []
        current = ""
        
        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if len(candidate) <= chunk_size:
                current = candidate
                continue
            
            if current:
                chunks.append(current)
            
            if len(paragraph) <= chunk_size:
                current = paragraph
                continue
            
            start = 0
            while start < len(paragraph):
                end = min(start + chunk_size, len(paragraph))
                chunks.append(paragraph[start:end])
                if end == len(paragraph):
                    break
                start = max(end - chunk_overlap, start + 1)
            current = ""
        
        if current:
            chunks.append(current)
        
        return chunks
    
    @staticmethod
    def _calculate_metrics(
        chunks: List[str],
        original_text_length: int,
        chunk_size: int,
        chunk_overlap: int,
        processing_time_ms: float
    ) -> Dict:
        """Calculate metrics for chunking result."""
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
                "median_chunk_size": 0,
                "chunks_under_threshold": 0,
                "chunks_over_limit": 0,
                "cumulative_redundancy_pct": 0,
            }
        
        chunk_sizes = [len(c) for c in chunks]
        chunk_sizes.sort()
        
        # Calculate redundancy from overlaps
        total_chunk_length = sum(chunk_sizes)
        overlap_redundancy = total_chunk_length - original_text_length
        cumulative_redundancy_pct = (overlap_redundancy / original_text_length * 100) if original_text_length > 0 else 0
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "median_chunk_size": chunk_sizes[len(chunk_sizes) // 2],
            "chunks_under_threshold": sum(1 for s in chunk_sizes if s < chunk_overlap),
            "chunks_over_limit": sum(1 for s in chunk_sizes if s > chunk_size),
            "cumulative_redundancy_pct": round(cumulative_redundancy_pct, 2),
        }
    
    def test_strategy(
        self,
        doc_data: Dict,
        full_text: str,
        config: ChunkingConfig
    ) -> ChunkingResult:
        """Test a single chunking strategy."""
        import time
        
        full_text = self._normalize_text(full_text)
        start_time = time.time() * 1000
        chunks = self._chunk_text(full_text, config.chunk_size, config.chunk_overlap)
        processing_time_ms = time.time() * 1000 - start_time
        
        metrics = self._calculate_metrics(
            chunks,
            len(full_text),
            config.chunk_size,
            config.chunk_overlap,
            processing_time_ms
        )
        
        return ChunkingResult(
            config=config,
            total_chunks=metrics["total_chunks"],
            total_text_length=len(full_text),
            avg_chunk_size=metrics["avg_chunk_size"],
            min_chunk_size=metrics["min_chunk_size"],
            max_chunk_size=metrics["max_chunk_size"],
            median_chunk_size=metrics["median_chunk_size"],
            chunks_under_threshold=metrics["chunks_under_threshold"],
            chunks_over_limit=metrics["chunks_over_limit"],
            cumulative_redundancy_pct=metrics["cumulative_redundancy_pct"],
            processing_time_ms=processing_time_ms
        )
    
    def optimize_all_documents(
        self,
        documents: Dict[str, Dict],
        full_texts: Dict[str, str],
        configs: List[ChunkingConfig]
    ) -> Dict[str, DocumentMetrics]:
        """Test all strategies on all documents."""
        results = {}
        
        for i, (doc_id, doc_data) in enumerate(documents.items(), 1):
            print(f"\n[{i}/{len(documents)}] Testing {doc_data['title'] or doc_id}")
            
            full_text = full_texts.get(doc_id)
            if not full_text:
                print(f"  Skipped: no full text")
                continue
            
            doc_metrics = DocumentMetrics(
                doc_id=doc_id,
                title=doc_data["title"],
                source_url=doc_data["source_url"],
                original_text_length=len(full_text),
                chunking_results={}
            )
            
            for config in configs:
                result = self.test_strategy(doc_data, full_text, config)
                doc_metrics.chunking_results[config.name] = result
                print(
                    f"  {config.name}: {result.total_chunks} chunks, "
                    f"avg={result.avg_chunk_size:.0f}, "
                    f"redundancy={result.cumulative_redundancy_pct:.1f}%"
                )
            
            results[doc_id] = doc_metrics
        
        return results


class VectorStoreRecordGenerator:
    """Generate vector-store records with optimized chunking."""
    
    @staticmethod
    def generate_records(
        doc_id: str,
        doc_data: Dict,
        full_text: str,
        config: ChunkingConfig
    ) -> List[Dict]:
        """Generate vector-store records for a document with given chunking config."""
        optimizer = ChunkingStrategyOptimizer()
        normalized_text = optimizer._normalize_text(full_text)
        chunks = optimizer._chunk_text(normalized_text, config.chunk_size, config.chunk_overlap)
        
        records = []
        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}:{idx}"
            records.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "doc_id": doc_id,
                    "source_url": doc_data.get("source_url", ""),
                    "title": doc_data.get("title", ""),
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "chunk_size": config.chunk_size,
                    "chunk_overlap": config.chunk_overlap,
                    "strategy": config.name,
                }
            })
        
        return records


class ReportGenerator:
    """Generate reports for chunking optimization."""
    
    @staticmethod
    def generate_summary_report(results: Dict[str, DocumentMetrics]) -> Dict:
        """Generate summary statistics across all documents."""
        if not results:
            return {}
        
        all_configs = set()
        for doc_metrics in results.values():
            all_configs.update(doc_metrics.chunking_results.keys())
        
        summary = {}
        for config_name in sorted(all_configs):
            config_results = []
            for doc_metrics in results.values():
                if config_name in doc_metrics.chunking_results:
                    config_results.append(doc_metrics.chunking_results[config_name])
            
            if config_results:
                chunk_counts = [r.total_chunks for r in config_results]
                avg_chunk_sizes = [r.avg_chunk_size for r in config_results]
                redundancies = [r.cumulative_redundancy_pct for r in config_results]
                
                summary[config_name] = {
                    "num_documents_tested": len(config_results),
                    "total_chunks_generated": sum(chunk_counts),
                    "avg_chunks_per_doc": sum(chunk_counts) / len(config_results),
                    "avg_chunk_size_across_docs": sum(avg_chunk_sizes) / len(avg_chunk_sizes),
                    "avg_redundancy_pct": sum(redundancies) / len(redundancies),
                    "min_chunks": min(chunk_counts),
                    "max_chunks": max(chunk_counts),
                }
        
        return summary
    
    @staticmethod
    def save_report(results: Dict[str, DocumentMetrics], output_file: str) -> None:
        """Save detailed report to JSON file."""
        report = {
            "metadata": {
                "total_documents": len(results),
                "timestamp": __import__("datetime").datetime.now().isoformat(),
            },
            "documents": {}
        }
        
        for doc_id, doc_metrics in results.items():
            doc_report = {
                "title": doc_metrics.title,
                "source_url": doc_metrics.source_url,
                "original_text_length": doc_metrics.original_text_length,
                "strategies": {}
            }
            
            for config_name, result in doc_metrics.chunking_results.items():
                doc_report["strategies"][config_name] = {
                    "chunk_size": result.config.chunk_size,
                    "chunk_overlap": result.config.chunk_overlap,
                    "total_chunks": result.total_chunks,
                    "avg_chunk_size": round(result.avg_chunk_size, 2),
                    "min_chunk_size": result.min_chunk_size,
                    "max_chunk_size": result.max_chunk_size,
                    "median_chunk_size": result.median_chunk_size,
                    "chunks_under_threshold": result.chunks_under_threshold,
                    "chunks_over_limit": result.chunks_over_limit,
                    "cumulative_redundancy_pct": result.cumulative_redundancy_pct,
                    "processing_time_ms": round(result.processing_time_ms, 2),
                }
            
            report["documents"][doc_id] = doc_report
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed report saved to: {output_file}")
    
    @staticmethod
    def print_summary(summary: Dict) -> None:
        """Print summary report to console."""
        print("\n" + "="*80)
        print("CHUNKING STRATEGY OPTIMIZATION SUMMARY")
        print("="*80)
        
        for config_name in sorted(summary.keys()):
            stats = summary[config_name]
            print(f"\n{config_name}:")
            print(f"  Documents tested:        {stats['num_documents_tested']}")
            print(f"  Total chunks generated:  {stats['total_chunks_generated']}")
            print(f"  Avg chunks per doc:      {stats['avg_chunks_per_doc']:.1f}")
            print(f"  Avg chunk size:          {stats['avg_chunk_size_across_docs']:.0f} chars")
            print(f"  Avg redundancy:          {stats['avg_redundancy_pct']:.2f}%")
            print(f"  Chunk count range:       {stats['min_chunks']}-{stats['max_chunks']}")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize chunking strategy without re-crawling or re-indexing"
    )
    parser.add_argument(
        "--cache-file",
        default="cache/processed_records.jsonl",
        help="Path to cached records file"
    )
    parser.add_argument(
        "--chunk-sizes",
        default="800,1000,1200,1500,2000",
        help="Comma-separated chunk sizes to test"
    )
    parser.add_argument(
        "--overlaps",
        default="100,200,300",
        help="Comma-separated overlap sizes to test"
    )
    parser.add_argument(
        "--output",
        default="chunking_optimization_report.json",
        help="Output report file path"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limit number of documents to test (for quick testing)"
    )
    parser.add_argument(
        "--export-records",
        type=str,
        default=None,
        help="Export optimized records to JSONL file for a specific strategy (e.g., 'size_1200_overlap_200')"
    )
    
    args = parser.parse_args()
    
    # Parse configurations
    chunk_sizes = [int(x.strip()) for x in args.chunk_sizes.split(",")]
    overlaps = [int(x.strip()) for x in args.overlaps.split(",")]
    configs = [
        ChunkingConfig(chunk_size=cs, chunk_overlap=ov)
        for cs in chunk_sizes for ov in overlaps
    ]
    
    print(f"Testing {len(configs)} chunking configurations:")
    for config in configs:
        print(f"  - {config.name}")
    
    # Load cached data
    loader = CachedDataLoader(args.cache_file)
    documents = loader.load()
    
    if args.max_docs:
        documents = dict(list(documents.items())[:args.max_docs])
    
    # Reconstruct full texts
    print(f"\nReconstructing full texts from {len(documents)} documents...")
    full_texts = {}
    for doc_id, doc_data in documents.items():
        full_text = loader.reconstruct_full_text(doc_id)
        if full_text:
            full_texts[doc_id] = full_text
    
    print(f"Successfully reconstructed {len(full_texts)} documents")
    
    # Optimize chunking strategies
    print(f"\nTesting chunking strategies on {len(full_texts)} documents...")
    optimizer = ChunkingStrategyOptimizer()
    results = optimizer.optimize_all_documents(documents, full_texts, configs)
    
    # Generate reports
    summary = ReportGenerator.generate_summary_report(results)
    ReportGenerator.print_summary(summary)
    ReportGenerator.save_report(results, args.output)
    
    # Export optimized records if requested
    if args.export_records:
        print(f"\nExporting optimized records for strategy: {args.export_records}")
        
        # Find the config matching the export_records name
        target_config = None
        for config in configs:
            if config.name == args.export_records:
                target_config = config
                break
        
        if not target_config:
            print(f"Error: strategy '{args.export_records}' not found in tested configs")
            return
        
        all_records = []
        for doc_id, doc_data in documents.items():
            if doc_id in full_texts:
                records = VectorStoreRecordGenerator.generate_records(
                    doc_id,
                    doc_data,
                    full_texts[doc_id],
                    target_config
                )
                all_records.extend(records)
        
        export_file = f"optimized_records_{args.export_records}.jsonl"
        with open(export_file, 'w', encoding='utf-8') as f:
            for record in all_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print(f"Exported {len(all_records)} optimized records to: {export_file}")
        print(f"\nTo re-index with these records:")
        print(f"  1. Run: python phase2-chatbot/rag_engine.py --clear-index")
        print(f"  2. Run: python phase2-chatbot/vector_store.py --ingest {export_file}")


if __name__ == "__main__":
    main()
