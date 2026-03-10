#!/usr/bin/env python3
"""Utility script to inspect ChromaDB indexed records."""

import argparse
import os
from typing import Optional

import chromadb
from dotenv import load_dotenv

load_dotenv()


def inspect_collection(
    max_records: Optional[int] = None,
    search_query: Optional[str] = None,
    show_embeddings: bool = False,
) -> None:
    """Inspect ChromaDB collection."""
    
    persist_directory = os.getenv(
        "VECTOR_DB_PATH",
        os.path.join(os.path.dirname(__file__), "..", "database", "vectors"),
    )
    collection_name = os.getenv("VECTOR_COLLECTION_NAME", "hcmut_regulations")
    
    print(f"[INFO] Connecting to ChromaDB at: {persist_directory}")
    print(f"[INFO] Collection name: {collection_name}\n")
    
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name=collection_name)
    
    # Get total count
    total_count = collection.count()
    print(f"📊 Total indexed records: {total_count}\n")
    
    if total_count == 0:
        print("⚠️  Collection is empty!")
        return
    
    if search_query:
        print(f"🔍 Searching for: '{search_query}'\n")
        results = collection.query(
            query_texts=[search_query],
            n_results=min(5, total_count),
            include=["documents", "metadatas", "distances", "embeddings" if show_embeddings else None]
        )
        
        if results["documents"] and results["documents"][0]:
            for i, (doc, meta, distance) in enumerate(
                zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
            ):
                print(f"Result {i+1} (distance: {distance:.4f}):")
                print(f"  Text: {doc[:200]}..." if len(doc) > 200 else f"  Text: {doc}")
                print(f"  Metadata: {meta}")
                if show_embeddings and results["embeddings"]:
                    print(f"  Embedding dim: {len(results['embeddings'][0][i])}")
                print()
        else:
            print("No results found.")
        return
    
    # Get all records (with limit)
    limit = max_records or 10
    results = collection.get(
        limit=limit,
        include=["documents", "metadatas"]
    )
    
    print(f"📋 Showing first {min(limit, total_count)} records:\n")
    
    for i, (doc_id, document, metadata) in enumerate(
        zip(results["ids"], results["documents"], results["metadatas"])
    ):
        print(f"Record {i+1}:")
        print(f"  ID: {doc_id}")
        print(f"  Text: {document[:150]}..." if len(document) > 150 else f"  Text: {document}")
        print(f"  Metadata: {metadata}")
        if show_embeddings and results["embeddings"]:
            embedding = results["embeddings"][i]
            print(f"  Embedding: {len(embedding)} dimensions, first 5 values: {embedding[:5]}")
        print()


def list_collections() -> None:
    """List all available collections."""
    persist_directory = os.getenv(
        "VECTOR_DB_PATH",
        os.path.join(os.path.dirname(__file__), "..", "database", "vectors"),
    )
    
    client = chromadb.PersistentClient(path=persist_directory)
    collections = client.list_collections()
    
    if not collections:
        print("No collections found.")
        return
    
    print("📦 Available collections:\n")
    for col in collections:
        print(f"  - {col.name}: {col.count()} records")


def delete_collection(collection_name: str) -> None:
    """Delete a collection."""
    persist_directory = os.getenv(
        "VECTOR_DB_PATH",
        os.path.join(os.path.dirname(__file__), "..", "database", "vectors"),
    )
    
    client = chromadb.PersistentClient(path=persist_directory)
    client.delete_collection(name=collection_name)
    print(f"✅ Collection '{collection_name}' deleted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect ChromaDB indexed records")
    parser.add_argument(
        "--list-collections",
        action="store_true",
        help="List all available collections",
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Search query to find similar documents",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=10,
        help="Max records to display (default: 10)",
    )
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="Include embedding vectors in output",
    )
    parser.add_argument(
        "--delete-collection",
        type=str,
        help="Delete a collection by name",
    )
    
    args = parser.parse_args()
    
    if args.list_collections:
        list_collections()
    elif args.delete_collection:
        delete_collection(args.delete_collection)
    else:
        inspect_collection(
            max_records=args.max_records,
            search_query=args.search,
            show_embeddings=args.with_embeddings,
        )
