"""Validation script for checking embedding dimensions and Milvus setup.

This script validates:
1. Gemini embedding model dimension (should be 768)
2. Milvus connection and availability
3. Embedding dimension consistency
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

def validate_gemini_embedding_dimension():
    """Validate Gemini embedding dimension matches expected (3072)."""
    import google.generativeai as genai

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in .env")
        return False

    genai.configure(api_key=api_key)
    embedding_model = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

    print(f"\n--- Validating Gemini Embedding Dimension ---")
    print(f"Embedding Model: {embedding_model}")

    try:
        test_text = "Testing embedding dimension for HCMUT regulations"
        response = genai.embed_content(
            model=embedding_model,
            content=test_text,
            task_type="retrieval_document",
        )
        embedding = response.get("embedding", [])
        dimension = len(embedding)

        print(f"✓ Successfully generated embedding")
        print(f"  Dimension: {dimension}")

        if dimension == 3072:
            print(f"✓ Dimension matches expected (3072)")
            return True
        else:
            print(f"❌ Dimension mismatch! Expected 3072, got {dimension}")
            return False

    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return False


def validate_milvus_connection():
    """Validate connection to Milvus server."""
    load_dotenv()
    milvus_host = os.getenv("MILVUS_HOST", "localhost")
    milvus_port = int(os.getenv("MILVUS_PORT", "19530"))

    print(f"\n--- Validating Milvus Connection ---")
    print(f"Host: {milvus_host}")
    print(f"Port: {milvus_port}")

    try:
        from pymilvus import connections
        
        connections.connect(
            alias="default",
            host=milvus_host,
            port=milvus_port,
            timeout=10,
        )
        print(f"✓ Successfully connected to Milvus")
        return True

    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")
        print(f"   Make sure Milvus is running (docker-compose up)")
        return False


def validate_milvus_collection_schema():
    """Validate Milvus collection schema has correct embedding dimension (3072)."""
    load_dotenv()
    milvus_host = os.getenv("MILVUS_HOST", "localhost")
    milvus_port = int(os.getenv("MILVUS_PORT", "19530"))

    print(f"\n--- Validating Milvus Collection Schema ---")

    try:
        from pymilvus import connections, Collection, utility

        connections.connect(
            alias="default",
            host=milvus_host,
            port=milvus_port,
            timeout=10,
        )

        collection_name = "hcmut_regulations"
        
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            schema = collection.schema
            
            print(f"✓ Collection '{collection_name}' exists")
            print(f"  Fields:")
            
            embedding_dim = None
            for field in schema.fields:
                print(f"    - {field.name}: {field.dtype}")
                if field.name == "embedding":
                    embedding_dim = field.params.get("dim", 0)
                    print(f"      Dimension: {embedding_dim}")
            
            if embedding_dim == 3072:
                print(f"✓ Embedding dimension is correct (3072)")
                return True
            elif embedding_dim is None:
                print(f"⚠️  Embedding field not found, collection needs to be initialized")
                return True  # Not an error, just not yet initialized
            else:
                print(f"❌ Embedding dimension mismatch! Expected 3072, got {embedding_dim}")
                return False
        else:
            print(f"⚠️  Collection '{collection_name}' does not exist yet")
            print(f"   It will be created when you run milvus_injection.py")
            return True

    except Exception as e:
        print(f"❌ Error validating collection: {e}")
        return False


def validate_environment_setup():
    """Validate all environment variables are set."""
    load_dotenv()
    
    print(f"\n--- Validating Environment Setup ---")
    
    required_vars = [
        "GEMINI_API_KEY",
        "EMBEDDING_MODEL",
        "EMBEDDING_TASK_TYPE",
    ]
    
    optional_vars = [
        "MILVUS_HOST",
        "MILVUS_PORT",
        "CACHE_DIR",
        "CRAWLED_CACHE_FILE",
        "VECTOR_DB_PATH",
    ]
    
    all_good = True
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✓ {var}: {'*' * (len(value) - 4) + value[-4:] if var.endswith('KEY') else value}")
        else:
            print(f"❌ {var}: NOT SET")
            all_good = False
    
    print(f"\n  Optional variables:")
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✓ {var}: {value}")
        else:
            default = {
                "MILVUS_HOST": "localhost",
                "MILVUS_PORT": "19530",
                "CACHE_DIR": "cache",
                "VECTOR_DB_PATH": "database/vectors",
            }.get(var, "(no default)")
            print(f"  ⚠️  {var}: using default ({default})")
    
    return all_good


def main():
    parser = argparse.ArgumentParser(
        description="Validate Milvus and Gemini setup for HCMUT regulations"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all validation checks",
    )
    parser.add_argument(
        "--gemini",
        action="store_true",
        help="Validate Gemini embedding dimension only",
    )
    parser.add_argument(
        "--milvus",
        action="store_true",
        help="Validate Milvus connection only",
    )
    parser.add_argument(
        "--schema",
        action="store_true",
        help="Validate Milvus collection schema only",
    )
    parser.add_argument(
        "--env",
        action="store_true",
        help="Validate environment variables only",
    )

    args = parser.parse_args()

    # Default to full validation if no specific check requested
    if not any([args.full, args.gemini, args.milvus, args.schema, args.env]):
        args.full = True

    print("=" * 60)
    print("HCMUT Regulations - Embedding & Milvus Validation")
    print("=" * 60)

    results = {}

    if args.full or args.env:
        results["env"] = validate_environment_setup()

    if args.full or args.gemini:
        results["gemini"] = validate_gemini_embedding_dimension()

    if args.full or args.milvus:
        results["milvus"] = validate_milvus_connection()

    if args.full or args.schema:
        results["schema"] = validate_milvus_collection_schema()

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for check, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL" if result is False else "⚠️  WARN"
        print(f"{check.upper()}: {status}")

    print(f"\nTotal: {passed}/{total} checks passed")

    if all(results.values()):
        print("\n✓ All validation checks passed!")
        print("You can now run: python milvus_injection.py")
        return 0
    else:
        print("\n❌ Some validation checks failed.")
        print("Please fix the issues before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
