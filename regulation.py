import time
import json
import os
import re
import tempfile
import requests
import google.generativeai as genai
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv

# ==========================================
# TẢI CẤU HÌNH TỪ FILE .ENV
# ==========================================
load_dotenv()

# ==========================================
# CẤU HÌNH HỆ THỐNG
# ==========================================
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
STATE_FILE = os.getenv("STATE_FILE", "seen_laws.json")
MAIN_MODEL_NAME = os.getenv("MAIN_MODEL", "gemini-3-flash-preview")
FALLBACK_MODEL_NAME = os.getenv("FALLBACK_MODEL", "gemini-2.5-flash")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "")
HEADLESS_MODE = os.getenv("HEADLESS_MODE", "true").lower() == "true"
WAIT_TIME = int(os.getenv("WAIT_TIME", "8"))
DELAY_BETWEEN_REQUESTS = int(os.getenv("DELAY_BETWEEN_REQUESTS", "5"))
PDF_DOWNLOAD_TIMEOUT = int(os.getenv("PDF_DOWNLOAD_TIMEOUT", "60"))
DISCORD_REQUEST_TIMEOUT = int(os.getenv("DISCORD_REQUEST_TIMEOUT", "15"))

# Kiểm tra các biến bắt buộc
if not DISCORD_WEBHOOK_URL or not GEMINI_API_KEY:
    raise ValueError("Vui lòng cấu hình DISCORD_WEBHOOK_URL và GEMINI_API_KEY trong file .env")

# ==========================================
# KHỞI TẠO AI MODELS (CONTEXT 1 TRIỆU TOKENS)
# ==========================================
genai.configure(api_key=GEMINI_API_KEY)
main_model = genai.GenerativeModel(MAIN_MODEL_NAME)
fallback_model = genai.GenerativeModel(FALLBACK_MODEL_NAME)

# ==========================================
# QUẢN LÝ TRẠNG THÁI (TRÁNH GỬI THÔNG BÁO TRÙNG LẶP)
# ==========================================
def load_seen_links():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_seen_links(links):
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(links, f, ensure_ascii=False, indent=4)

# ==========================================
# XỬ LÝ FILE PDF TỪ GOOGLE DRIVE
# ==========================================
def extract_drive_id(url):
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    return match.group(1) if match else None

def download_full_pdf(drive_url):
    """ Tải TOÀN BỘ file PDF không cắt xén để AI đọc cả Phụ lục """
    file_id = extract_drive_id(drive_url)
    if not file_id: return None

    print(f"   -> Đang tải toàn bộ PDF (ID: {file_id[:8]}...)...")
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    try:
        # Sử dụng timeout từ cấu hình
        response = requests.get(download_url, timeout=PDF_DOWNLOAD_TIMEOUT)
        if response.status_code != 200: return None
        return response.content
    except Exception as e:
        print(f"   -> Lỗi khi tải PDF: {e}")
        return None

def process_and_summarize_pdf(pdf_bytes, title):
    """ Upload PDF lên server Google, gọi LLM đọc toàn văn, sau đó dọn rác. """
    print("   -> Đang đẩy tài liệu lên Gemini Server (Native PDF)...")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_bytes)
        temp_path = temp_file.name

    try:
        uploaded_doc = genai.upload_file(path=temp_path, mime_type="application/pdf")
        
        # Sử dụng system prompt từ .env hoặc mặc định
        if SYSTEM_PROMPT:
            instruction = f"{SYSTEM_PROMPT}\nTài liệu: \"{title}\"."
        
        print("   -> Đang chờ model suy luận và tóm tắt toàn văn (Có thể mất 10-30s)...")
        try:
            response = main_model.generate_content([instruction, uploaded_doc])
            summary = response.text.strip()
        except Exception as e_main:
            print(f"      [Cảnh báo] Gemini 3 Flash lỗi: {e_main}. Đang kích hoạt Fallback 2.5...")
            response = fallback_model.generate_content([instruction, uploaded_doc])
            summary = response.text.strip()

        # Xóa file trên server Google để tiết kiệm dung lượng
        genai.delete_file(uploaded_doc.name)
        return summary
        
    except Exception as e:
        print(f"   -> Lỗi khi xử lý qua API: {e}")
        return "⚠️ Lỗi khi nhờ AI đọc file PDF."
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ==========================================
# DISCORD EMBED ALERTS (MÀU XANH BÁCH KHOA)
# ==========================================
def send_discord_alert(law_title, law_link, summary):
    print(f"--> Đang đẩy báo cáo lên Discord bằng thẻ Embed...")
    
    # Giới hạn an toàn của thẻ Description trong Embed là 4096 ký tự
    safe_summary = summary if len(summary) < 4000 else summary[:4000] + "\n\n*(Nội dung quá dài, đã được cắt bớt)*"
    
    embed = {
        "title": f"🚨 LUẬT HCMUT MỚI: {law_title}",
        "url": law_link,
        "description": f"**🤖 AI TÓM TẮT TOÀN VĂN:**\n\n{safe_summary}",
        "color": 3447003 # Mã màu thập phân của màu Light Blue (Xanh Bách Khoa)
    }
    
    payload = {
        "username": "Phòng Đào Tạo Bot", 
        "avatar_url": "https://upload.wikimedia.org/wikipedia/commons/d/de/HC_Polytechnic_Univ_Logo.png",
        "embeds": [embed] # Truyền Embed vào payload
    }
    
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=DISCORD_REQUEST_TIMEOUT)
        # Bắt lỗi rõ ràng nếu Discord từ chối nhận tin
        if response.status_code in [200, 204]:
            print("--> ✅ Đã đẩy lên Discord thành công!")
        else:
            print(f"--> ❌ Lỗi từ Discord (Mã {response.status_code}): {response.text}")
    except Exception as e:
        print(f"--> ❌ Lỗi mạng khi gửi Discord: {e}")

# ==========================================
# LUỒNG CHẠY CHÍNH (PIPELINE)
# ==========================================
def run_pipeline():
    print("=== BẮT ĐẦU CÀO DỮ LIỆU HCMUT ===")
    options = Options()
    if HEADLESS_MODE:
        options.add_argument('--headless')
    options.add_argument('--log-level=3')
    options.add_argument('--disable-gpu')
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    try:
        driver.get("https://hcmut.edu.vn/dao-tao/quy-che-quy-dinh")
        time.sleep(WAIT_TIME) # Chờ cho Javascript render DOM ổn định
        
        elements = driver.find_elements(By.CSS_SELECTOR, 'div.sub-link a')
        current_laws = []
        for el in elements:
            href = el.get_attribute('href')
            text = el.text.strip() or el.get_attribute('textContent').strip()
            if href and ("drive.google.com" in href or "docs.google.com" in href):
                current_laws.append({"title": text, "link": href})
        
        seen_links = load_seen_links()
        seen_urls = [law['link'] for law in seen_links]
        
        new_laws = [law for law in current_laws if law['link'] not in seen_urls]
        
        if new_laws:
            print(f"Phát hiện {len(new_laws)} quy định mới. Bắt đầu xử lý...\n")
            
            for law in new_laws:
                print(f"📌 Đang xử lý: {law['title']}")
                pdf_bytes = download_full_pdf(law['link'])
                
                if pdf_bytes:
                    summary = process_and_summarize_pdf(pdf_bytes, law['title'])
                else:
                    summary = "⚠️ Tool không tải được file PDF này (link hỏng hoặc bị khóa quyền tải)."
                
                send_discord_alert(law['title'], law['link'], summary)
                time.sleep(DELAY_BETWEEN_REQUESTS) # Delay giữa mỗi đợt quét để tránh rate limit
            
            seen_links.extend(new_laws)
            save_seen_links(seen_links)
            print("\n=== HOÀN TẤT CẬP NHẬT MỚI ===")
        else:
            print(f"Đã quét {len(current_laws)} tài liệu. Chưa có gì mới.")
            print("=== HOÀN TẤT ===")
            
    except Exception as e:
        print(f"Lỗi hệ thống: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    run_pipeline()