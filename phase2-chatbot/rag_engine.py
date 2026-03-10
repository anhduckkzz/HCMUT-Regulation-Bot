import chromadb
import os
from dotenv import load_dotenv
from mistralai import Mistral
from sentence_transformers import SentenceTransformer

# 1. Cấu hình môi trường (Shared config)
load_dotenv()


class RAGEngine:
    def __init__(self):
        # Kết nối tới local ChromaDB (đã được khởi tạo và chứa dữ liệu quy định)
        self.client = chromadb.PersistentClient(path="../database/vectors")
        self.collection = self.client.get_collection(name="hcmut_regulations")

        # Khởi tạo Mistral AI client (using mistral-medium for better quality)
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if not self.mistral_api_key:
            raise ValueError("Missing MISTRAL_API_KEY in .env")
        self.client_llm = Mistral(api_key=self.mistral_api_key)
        self.model_name = os.getenv("MAIN_MODEL", "mistral-medium-latest")

        # Khởi tạo HuggingFace embedding model
        embedding_model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-mpnet-base-v2")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Retrieval parameters
        self.retrieval_k = int(os.getenv("RETRIEVAL_K", "5"))

    def _get_embedding(self, text):
        """Biến câu hỏi của sinh viên thành Vector số"""
        embedding = self.embedding_model.encode(text, convert_to_tensor=False)
        return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)

    def retrieve(self, query_text):
        """Bước R (Retrieval) - retrieve top K documents"""
        query_vector = self._get_embedding(query_text)

        # Truy vấn vào ChromaDB với số lượng chunks được cấu hình
        results = self.collection.query(
            query_embeddings=[query_vector], n_results=self.retrieval_k
        )
        return results

    def build_prompt(self, query, context_chunks):
        """Bước A (Augmentation) - Build augmented prompt with context"""
        context_str = "\n---\n".join(context_chunks)

        prompt = f"""You are a helpful assistant for HCMUT (Ho Chi Minh University of Technology).
Your role is to answer student questions about academic regulations accurately and concisely.

IMPORTANT RULES:
1. Answer ONLY based on provided regulations - do NOT use external knowledge
2. If information is not found, respond: "Xin lỗi, mình không tìm thấy quy định này trong cơ sở dữ liệu hiện tại."
3. Be concise, clear, and use Vietnamese naturally
4. Always be helpful and courteous

PROVIDED REGULATIONS:
{context_str}

STUDENT QUESTION: {query}

RESPONSE:"""
        return prompt

    def generate_response(self, user_query):
        """Bước G (Generation)"""
        # 1. Tìm tài liệu liên quan
        search_results = self.retrieve(user_query)
        chunks = search_results["documents"][0]
        metadatas = search_results["metadatas"][0]

        # 2. Xây dựng Prompt
        full_prompt = self.build_prompt(user_query, chunks)

        # 3. Gọi Mistral AI
        response = self.client_llm.chat.complete(
            model=self.model_name,
            messages=[{"role": "user", "content": full_prompt}]
        )
        response_text = response.choices[0].message.content

        # 4. Trình bày kèm Metadata (Nguồn tham khảo)
        final_answer = response_text + "\n\n**Nguồn tham khảo:**\n"
        sources = set([m["source_url"] for m in metadatas])  # Lọc trùng nguồn
        for s in sources:
            final_answer += f"- {s}\n"

        return final_answer

if __name__ == "__main__":
    engine = RAGEngine()
    print(engine.generate_response("Cho tôi coi về danh sách người hướng dẫn và đề tài nghiên cứu của đào tạo sau đại học nhé?"))
    