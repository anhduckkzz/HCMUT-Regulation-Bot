import google.generativeai as genai
import os
import argparse
import time
from typing import Any, List, Tuple
from dotenv import load_dotenv
from pymilvus import Collection, connections
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

# 1. Cấu hình môi trường (Shared config)
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# if not os.environ.get("GOOGLE_API_KEY"):
#     os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")


class MilvusVectorStore:
    """LangChain-compatible wrapper for Milvus vector database."""
    
    def __init__(
        self,
        embeddings: GoogleGenerativeAIEmbeddings,
        debug_title: bool = False,
    ):
        """Initialize Milvus Vector Store wrapper.
        
        Args:
            embeddings: LangChain embeddings instance
            debug_title: Enable debug output for title field retrieval
        """
        self.embeddings = embeddings
        self.debug_title = debug_title
        milvus_host = os.getenv("MILVUS_HOST", "localhost")
        milvus_port = int(os.getenv("MILVUS_PORT", "19530"))
        collection_name = os.getenv("MILVUS_COLLECTION_NAME", "hcmut_regulations")
        connections.connect(
            alias="default",
            host=milvus_host,
            port=milvus_port,
            timeout=10,
        )
        self.collection = Collection(collection_name)
        self.collection.load()
    
    def similarity_search_with_score(
        self, query_text: str, k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents with scores.
        
        Args:
            query_text: Query text to search for
            k: Number of results to retrieve
            
        Returns:
            List of (Document, distance_score) tuples
        """
        query_vector = self.embeddings.embed_query(query_text)
        
        search_params = {
            "metric_type": "L2", # euclidean distance
            "params": {"nprobe": 10}, # # of clusters used during query
        }
        
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["id", "text", "source_url", "doc_title"],
        )
        
        # Convert Milvus results to LangChain Document format
        docs_with_scores = []
        if results and len(results) > 0:
            for hit_idx, hit in enumerate(results[0]):
                # Safely extract field values from Milvus entity
                # hit.entity may be a dict-like object or need bracket notation
                entity = hit.entity
                if entity is None:
                    if self.debug_title:
                        print(f"[DEBUG] Hit {hit_idx}: entity is None")
                    continue
                
                if self.debug_title:
                    print(f"\n[DEBUG-TITLE] Processing hit {hit_idx}:")
                    print(f"  Entity type: {type(entity)}")
                    print(f"  Entity class: {entity.__class__.__name__}")
                    print(f"  Has 'get' method: {hasattr(entity, 'get')}")
                    print(f"  Is dict: {isinstance(entity, dict)}")
                    print(f"  Dir(entity): {[x for x in dir(entity) if not x.startswith('_')]}")
                    
                    # Try to access as dict
                    if isinstance(entity, dict):
                        print(f"  Dict keys: {list(entity.keys())}")
                        for key in entity.keys():
                            print(f"    {key}: {entity[key][:50] if isinstance(entity[key], str) and len(entity[key]) > 50 else entity[key]}")
                    
                    # Try bracket notation
                    try:
                        print(f"  Bracket access ['doc_title']: {entity['doc_title']}")
                    except (KeyError, TypeError) as e:
                        print(f"  Bracket access ['doc_title'] failed: {type(e).__name__}: {e}")
                    
                    # Try attribute access
                    try:
                        print(f"  Attribute access .doc_title: {entity.doc_title}")
                    except AttributeError as e:
                        print(f"  Attribute access .doc_title failed: {e}")
                    
                    # Try .get() method
                    try:
                        result = entity.get('doc_title', 'NOT_FOUND')
                        print(f"  .get('doc_title') result: {result}")
                    except Exception as e:
                        print(f"  .get('doc_title') failed: {type(e).__name__}: {e}")
                
                # Try to extract fields - handle both dict and object access patterns
                def get_field(field_name, default=""):
                    try:
                        # Try dictionary-style access first
                        if hasattr(entity, 'get'):
                            return entity.get(field_name, default)
                        # Try bracket notation
                        elif isinstance(entity, dict):
                            return entity[field_name] if field_name in entity else default
                        # Try attribute access
                        else:
                            return getattr(entity, field_name, default)
                    except (KeyError, AttributeError, TypeError):
                        return default
                
                text_val = get_field("text", "")
                source_url_val = get_field("source_url", "")
                doc_title_val = get_field("doc_title", "")
                
                if self.debug_title:
                    print(f"  Extracted text: {text_val[:50] if len(text_val) > 50 else text_val}")
                    print(f"  Extracted source_url: {source_url_val}")
                    print(f"  Extracted doc_title: {doc_title_val}")
                
                doc = Document(
                    page_content=text_val,
                    metadata={
                        "source_url": source_url_val,
                        "doc_title": doc_title_val,
                        "id": hit.id,
                    },
                )
                docs_with_scores.append((doc, float(hit.distance)))
        
        return docs_with_scores


class RAGEngine:
    def __init__(self, enable_delay: bool = False, debug_title: bool = False):
        """Initialize RAG Engine.
        
        Args:
            enable_delay: Enable 20-second delay on quota limit
            debug_title: Enable debug output for title field retrieval
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
        
        # Initialize LangChain embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            task_type=os.getenv("EMBEDDING_TASK_TYPE", "retrieval_query"),
        )
        
        # Initialize Milvus vector store wrapper
        self.vector_store = MilvusVectorStore(
            embeddings=self.embeddings,
            debug_title=debug_title,
        )

        # Initialize LangChain LLM
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("MAIN_MODEL", "gemini-2.0-flash"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            max_output_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "3072")),
        )
        
        # Enable delay on quota limit
        self.enable_delay = enable_delay
        self.delay_seconds = 20

    def _get_embedding_with_retry(self, text: str) -> List[float]:
        """Get embedding with quota-aware retry logic.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        while True:
            try:
                return self.embeddings.embed_query(text)
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

    def retrieve(self, query_text: str) -> List[Tuple[Document, float]]:
        """Bước R (Retrieval) - Retrieve documents using Milvus.
        
        Args:
            query_text: User query string
        Returns:
            List of (Document, score) tuples with metadata
        """
        # Number of results to retrieve from Milvus
        k = int(os.getenv("RETRIEVAL_K", 5))
        
        # Use LangChain-compatible vector store for retrieval
        return self.vector_store.similarity_search_with_score(query_text, k=k)

    def _build_messages(self, query: str, context_chunks: List[str]) -> List:
        """Bước A (Augmentation) - Build LangChain messages with Vietnamese system prompt.
        
        Args:
            query: User query
            context_chunks: List of retrieved context chunks
            
        Returns:
            List of LangChain message objects (SystemMessage + HumanMessage)
        """
        context_str = "\n---\n".join(context_chunks)
        
        # Keep prompt structure consistent with rag_engine.py and enforce richer output.
        user_content = f"""
    Bạn là Trợ lý ảo thông minh của Đại học Bách Khoa TP.HCM (HCMUT). 
    Nhiệm vụ của bạn là giải đáp thắc mắc về quy chế học vụ dựa trên dữ liệu được cung cấp dưới đây.

    DỮ LIỆU QUY ĐỊNH:
{context_str}

CÂU HỎI CỦA SINH VIÊN: 
{query}"""

        system_prompt = """HƯỚNG DẪN TRẢ LỜI:
    1. Chỉ trả lời dựa trên dữ liệu được cung cấp. 
    2. Nếu không có thông tin trong dữ liệu, hãy nói: "Xin lỗi, mình không tìm thấy quy định này trong cơ sở dữ liệu hiện tại."
    3. Câu trả lời cần rõ ràng, đầy đủ ý quan trọng, dùng ngôi 'mình' và 'bạn'.
    4. Không trả lời dưới form khác ngoài văn bản (không dùng markdown, không dùng HTML, không dùng code block)."""

        comparison_hint = (
            "\nYÊU CẦU TRÌNH BÀY: Nếu câu hỏi mang tính so sánh giữa quy định cũ và mới, "
            "hãy trình bày tối thiểu 4 ý theo thứ tự: (1) Đối tượng áp dụng, "
            "(2) Khác biệt cách tính/quy đổi, (3) Mốc thời gian hiệu lực/chuyển tiếp, "
            "(4) Kết luận ngắn gọn cho sinh viên."
        )

        if any(keyword in query.lower() for keyword in ["khác", "so sánh", "cũ", "mới", "thay đổi", "gpa"]):
            system_prompt += comparison_hint

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]
        
        return messages

    def _generate_with_retry(self, messages: List) -> Any:
        """Generate response with quota-aware retry logic.
        
        Args:
            messages: List of LangChain message objects
            
        Returns:
            Raw LangChain response object
        """
        while True:
            try:
                return self.llm.invoke(messages)
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
    
    def _extract_response_text(self, response_obj: Any) -> str:
        """Extract plain text from LangChain/Google response payloads."""
        content = getattr(response_obj, "content", response_obj)

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    item_type = str(item.get("type", "")).lower()
                    if item_type in {"text", "output_text", "text_part"} and item.get("text"):
                        text_parts.append(str(item.get("text")))
                elif isinstance(item, str):
                    text_parts.append(item)

            if text_parts:
                return "\n".join(part.strip() for part in text_parts if part and part.strip())

        # Last resort fallback.
        return str(content)

    def _is_response_truncated(self, response_obj: Any, response_text: str) -> bool:
        """Detect whether model output likely stopped due to token limits."""
        metadata = getattr(response_obj, "response_metadata", {}) or {}
        finish_reason = str(metadata.get("finish_reason", "")).upper()

        if finish_reason in {"MAX_TOKENS", "LENGTH"}:
            return True

        # Fallback heuristic: long response ending without sentence punctuation.
        stripped = response_text.rstrip()
        if len(stripped) > 300 and stripped[-1:] not in {".", "!", "?", ":", ";", '"', "'", ")"}:
            return True

        return False

    def _continue_response_if_needed(self, messages: List, response_obj: Any, response_text: str) -> str:
        """If response is truncated, ask the model to continue once or twice."""
        merged_text = response_text.strip()
        prior_response_obj = response_obj

        for _ in range(2):
            if not self._is_response_truncated(prior_response_obj, merged_text):
                break

            continuation_messages = list(messages)
            if isinstance(prior_response_obj, AIMessage):
                continuation_messages.append(prior_response_obj)
            else:
                continuation_messages.append(AIMessage(content=merged_text))

            continuation_messages.append(
                HumanMessage(
                    content=(
                        "Câu trả lời vừa rồi bị cắt giữa chừng. "
                        "Hãy tiếp tục đúng phần còn thiếu, không lặp lại các ý đã nêu."
                    )
                )
            )

            prior_response_obj = self._generate_with_retry(continuation_messages)
            continuation_text = self._extract_response_text(prior_response_obj).strip()
            if not continuation_text:
                break

            merged_text = f"{merged_text}\n\n{continuation_text}".strip()

        return merged_text

    def generate_response(self, user_query: str) -> str:
        """Bước G (Generation) - Complete RAG pipeline using LangChain.
        
        Args:
            user_query: User query string
            
        Returns:
            Generated response with source attribution
        """
        # 1. Tìm tài liệu liên quan (Retrieval)
        docs_with_scores = self.retrieve(user_query)
        context_chunks = [doc.page_content for doc, _ in docs_with_scores]
        
        # 2. Xây dựng Prompt (Augmentation)
        messages = self._build_messages(user_query, context_chunks)
        
        # 3. Gọi Gemini with quota limit handling (Generation)
        response_obj = self._generate_with_retry(messages)
        response_text = self._extract_response_text(response_obj)
        response_text = self._continue_response_if_needed(messages, response_obj, response_text)
        
        # 4. Trình bày kèm Metadata (Nguồn tham khảo)
        final_answer = response_text + "\n\n**Nguồn tham khảo:**\n"

        # Keep title + URL so downstream UI can show more informative references.
        source_pairs = []
        seen_pairs = set()
        for doc, _ in docs_with_scores:
            source_url = str(doc.metadata.get("source_url", "")).strip()
            if not source_url:
                continue

            doc_title = str(doc.metadata.get("doc_title", "")).strip()
            if not doc_title:
                doc_title = "Tài liệu không rõ tiêu đề"

            pair_key = (doc_title, source_url)
            if pair_key in seen_pairs:
                continue

            seen_pairs.add(pair_key)
            source_pairs.append(pair_key)

        source_pairs.sort(key=lambda x: (x[0].lower(), x[1]))
        for doc_title, source_url in source_pairs:
            final_answer += f"- {doc_title}: {source_url}\n"
        
        return final_answer

    def experiment_generate_response(self, user_query: str) -> str:
        """Experiment mode: retrieve and build augmented prompt without API call.
        
        Args:
            user_query: User query string
            
        Returns:
            Full augmented prompt (no generation to save quota)
        """
        # 1. Tìm tài liệu liên quan
        docs_with_scores = self.retrieve(user_query)
        context_chunks = [doc.page_content for doc, _ in docs_with_scores]
        
        # 2. Xây dựng Prompt
        messages = self._build_messages(user_query, context_chunks)
        
        # Return formatted prompt for inspection
        system_msg = messages[0].content
        user_msg = messages[1].content
        return f"=== SYSTEM PROMPT ===\n{system_msg}\n\n=== USER MESSAGE ===\n{user_msg}"


def generate_response(user_query: str) -> str:
    """Export-friendly function to generate RAG response.
    
    Args:
        user_query: User query string
        
    Returns:
        Generated response with source attribution
    """
    engine = RAGEngine(enable_delay=False)
    return engine.generate_response(user_query)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Engine for HCMUT Regulations (using Milvus)")
    parser.add_argument("--delay", action="store_true", help="Enable 20-second delay on quota limit")
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="Experiment mode: print augmented prompt without API call (no quota usage)",
    )
    parser.add_argument(
        "--debug-title",
        action="store_true",
        help="Debug mode: print detailed info about document title field extraction from Milvus",
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
    
    engine = RAGEngine(
        enable_delay=args.delay,
        debug_title=args.debug_title,
    )
    
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
    