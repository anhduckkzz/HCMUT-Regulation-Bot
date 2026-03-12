import chromadb
import google.generativeai as genai
import os
import argparse
import time
from typing import Optional
from dotenv import load_dotenv

from cache_config import CacheConfig

# 1. Cấu hình môi trường (Shared config)
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class RAGEngine:
    def __init__(self, enable_delay: bool = False, vector_db_path: Optional[str] = None):
        """Initialize RAG Engine.
        
        Args:
            enable_delay: Enable 20-second delay on quota limit
            vector_db_path: Optional path to vector DB (env: VECTOR_DB_PATH)
        """
        # Initialize cache configuration to get vector DB path
        cache_config = CacheConfig(vector_db_path=vector_db_path)
        cache_config.ensure_directories_exist()
        
        # Kết nối tới local ChromaDB (đã được khởi tạo và chứa dữ liệu quy định)
        self.client = chromadb.PersistentClient(path=cache_config.vector_db_path)
        self.collection = self.client.get_collection(name=cache_config.collection_name)

        # Khởi tạo model Gemini
        self.model = genai.GenerativeModel(os.getenv("MAIN_MODEL", "gemini-2.0-flash"))
        
        # Enable delay on quota limit
        self.enable_delay = enable_delay
        self.delay_seconds = 20

    def _get_embedding(self, text):
        """Biến câu hỏi của sinh viên thành Vector số"""
        while True:
            try:
                result = genai.embed_content(
                    model=os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001"),
                    content=text,
                    task_type=os.getenv("EMBEDDING_TASK_TYPE", "retrieval_query")
                )
                return result["embedding"]
            except Exception as exc:
                error_msg = str(exc).lower()
                # Detect quota limit errors
                if "quota" in error_msg or "rate_limit" in error_msg or "429" in error_msg:
                    if self.enable_delay:
                        print(f"⚠️ Quota limit detected. Waiting {self.delay_seconds} seconds before retry...")
                        time.sleep(self.delay_seconds)
                        continue  # Retry after delay
                    else:
                        raise
                else:
                    raise

    def retrieve(self, query_text, n_results=3):
        """Bước R (Retrieval)"""
        query_vector = self._get_embedding(query_text)

        # Truy vấn vào ChromaDB
        results = self.collection.query(
            query_embeddings=[query_vector], n_results=int(os.getenv("RETRIEVAL_K", 5))
        )
        return results

    def build_prompt(self, query, context_chunks):
        """Bước A (Augmentation)"""
        context_str = "\n---\n".join(context_chunks)

        prompt = f"""
Bạn là Trợ lý ảo thông minh của Đại học Bách Khoa TP.HCM (HCMUT). 
Nhiệm vụ của bạn là giải đáp thắc mắc về quy chế học vụ dựa trên dữ liệu được cung cấp dưới đây.

DỮ LIỆU QUY ĐỊNH:
{context_str}

CÂU HỎI CỦA SINH VIÊN: 
{query}

HƯỚNG DẪN TRẢ LỜI:
1. Chỉ trả lời dựa trên dữ liệu được cung cấp. 
2. Nếu không có thông tin trong dữ liệu, hãy nói: "Xin lỗi, mình không tìm thấy quy định này trong cơ sở dữ liệu hiện tại."
3. Câu trả lời cần ngắn gọn, rõ ràng, dùng ngôi 'mình' và 'bạn'.
"""
        return prompt

    def generate_response(self, user_query):
        """Bước G (Generation)"""
        # 1. Tìm tài liệu liên quan
        search_results = self.retrieve(user_query)
        chunks = search_results["documents"][0]
        metadatas = search_results["metadatas"][0]

        # 2. Xây dựng Prompt
        full_prompt = self.build_prompt(user_query, chunks)

        # 3. Gọi Gemini with quota limit handling
        while True:
            try:
                response = self.model.generate_content(full_prompt)
                break  # Success, exit retry loop
            except Exception as exc:
                error_msg = str(exc).lower()
                # Detect quota limit errors
                if "quota" in error_msg or "rate_limit" in error_msg or "429" in error_msg:
                    if self.enable_delay:
                        print(f"⚠️ Quota limit detected. Waiting {self.delay_seconds} seconds before retry...")
                        time.sleep(self.delay_seconds)
                        continue  # Retry after delay
                    else:
                        raise
                else:
                    raise

        # 4. Trình bày kèm Metadata (Nguồn tham khảo)
        final_answer = response.text + "\n\n**Nguồn tham khảo:**\n"
        sources = set([m["source_url"] for m in metadatas])  # Lọc trùng nguồn
        for s in sources:
            final_answer += f"- {s}\n"

        return final_answer

    def experiment_generate_response(self, user_query):
        """Experiment mode: retrieve and build augmented prompt without API call"""
        # 1. Tìm tài liệu liên quan
        search_results = self.retrieve(user_query)
        chunks = search_results["documents"][0]
        metadatas = search_results["metadatas"][0]

        # 2. Xây dựng Prompt
        full_prompt = self.build_prompt(user_query, chunks)

        return full_prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Engine for HCMUT Regulations")
    parser.add_argument("--delay", action="store_true", help="Enable 20-second delay on quota limit")
    parser.add_argument(
        "--vector-db-path",
        type=str,
        default=None,
        help="Path to vector DB (env: VECTOR_DB_PATH, default: ../database/vectors/)",
    )
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="Experiment mode: print augmented prompt without API call (no quota usage)",
    )
    args = parser.parse_args()
    
    engine = RAGEngine(enable_delay=args.delay, vector_db_path=args.vector_db_path)
    
    if args.experiment:
        print(engine.experiment_generate_response("Sinh viên đạo văn hoặc làm bài hộ bị xử lý như thế nào?"))
    else:
        print(engine.generate_response("Sinh viên đạo văn hoặc làm bài hộ bị xử lý như thế nào?"))
    