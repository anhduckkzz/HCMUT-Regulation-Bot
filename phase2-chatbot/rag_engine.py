import chromadb
import google.generativeai as genai
import os
from dotenv import load_dotenv

# 1. Cấu hình môi trường (Shared config)
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class RAGEngine:
    def __init__(self):
        # Kết nối tới local ChromaDB (đã được khởi tạo và chứa dữ liệu quy định)
        self.client = chromadb.PersistentClient(path="../database/vectors")
        self.collection = self.client.get_collection(name="hcmut_regulations")

        # Khởi tạo model Gemini
        self.model = genai.GenerativeModel(os.getenv("MAIN_MODEL"))

    def _get_embedding(self, text):
        """Biến câu hỏi của sinh viên thành Vector số"""
        result = genai.embed_content(
            model="models/gemini-embedding-001", content=text, task_type="retrieval_query"
        )
        return result["embedding"]

    def retrieve(self, query_text, n_results=3):
        """Bước R (Retrieval)"""
        query_vector = self._get_embedding(query_text)

        # Truy vấn vào ChromaDB
        results = self.collection.query(
            query_embeddings=[query_vector], n_results=n_results
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

        # 3. Gọi Gemini
        response = self.model.generate_content(full_prompt)

        # 4. Trình bày kèm Metadata (Nguồn tham khảo)
        final_answer = response.text + "\n\n**Nguồn tham khảo:**\n"
        sources = set([m["source_url"] for m in metadatas])  # Lọc trùng nguồn
        for s in sources:
            final_answer += f"- {s}\n"

        return final_answer

if __name__ == "__main__":
    engine = RAGEngine()
    print(engine.generate_response("Kết luận hội đồng học vụ HK251 có những điểm đáng chú ý nào?"))
    