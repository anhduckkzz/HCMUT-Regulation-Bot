import google.generativeai as genai
import os
import argparse
import time
from dotenv import load_dotenv
from pymilvus import Collection, connections

# 1. Cấu hình môi trường (Shared config)
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class RAGEngine:
    def __init__(self, enable_delay: bool = False):
        """Initialize RAG Engine.
        
        Args:
            enable_delay: Enable 20-second delay on quota limit
        """
        
        # Get Milvus connection parameters from environment
        self.milvus_host = os.getenv("MILVUS_HOST", "localhost")
        self.milvus_port = int(os.getenv("MILVUS_PORT", "19530"))
        self.collection_name = os.getenv("MILVUS_COLLECTION_NAME", "hcmut_regulations")
        
        # Connect to Milvus
        connections.connect(
            alias="default",
            host=self.milvus_host,
            port=self.milvus_port,
            timeout=10,
        )
        
        # Get the collection
        self.collection = Collection(self.collection_name)
        self.collection.load()

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

    def retrieve(self, query_text):
        """Bước R (Retrieval) - using Milvus"""
        query_vector = self._get_embedding(query_text)
        
        # Number of results to retrieve from Milvus
        k = int(os.getenv("RETRIEVAL_K", 5))

        # Search in Milvus
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["id", "text", "source_url", "doc_title"],
        )

        # Normalize Milvus results to the internal response structure.
        formatted_results = {
            "documents": [[]],
            "metadatas": [[]],
            "ids": [],
            "distances": [],
        }
        
        if results and len(results) > 0:
            for hit in results[0]:
                formatted_results["documents"][0].append(hit.entity.get("text", ""))
                formatted_results["metadatas"][0].append({
                    "source_url": hit.entity.get("source_url", ""),
                    "doc_title": hit.entity.get("doc_title", ""),
                })
                formatted_results["ids"].append(hit.id)
                formatted_results["distances"].append(float(hit.distance))
        
        return formatted_results

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
4. Không trả lời dưới form khác ngoài văn bản (không dùng markdown, không dùng HTML, không dùng code block).
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
    parser = argparse.ArgumentParser(description="RAG Engine for HCMUT Regulations (using Milvus)")
    parser.add_argument("--delay", action="store_true", help="Enable 20-second delay on quota limit")
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="Experiment mode: print augmented prompt without API call (no quota usage)",
    )
    parser.add_argument(
        "--log-txt",
        type=str,
        nargs='?',
        const="rag_engine.log",
        default=None,
        help="Path to write logs to a text file (default: rag_engine.log)",
    )
    args = parser.parse_args()
    
    engine = RAGEngine(enable_delay=args.delay)
    
    query = input("Nhập câu hỏi của bạn về quy chế học vụ HCMUT: ")
    
    if args.experiment:
        response = engine.experiment_generate_response(query)
        print(response)
        if args.log_txt:
            with open(args.log_txt, 'w') as f:
                f.write(response)
    else:
        response = engine.generate_response(query)
        print(response)
        if args.log_txt:
            with open(args.log_txt, 'w') as f:
                f.write(response)
    