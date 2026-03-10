import time
import json
import os
import re
import tempfile
import requests
import fitz
from mistralai import Mistral
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
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
STATE_FILE = os.getenv("STATE_FILE", "seen_laws.json")
MAIN_MODEL_NAME = os.getenv("MAIN_MODEL", "mistral-small-latest")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "")
HEADLESS_MODE = os.getenv("HEADLESS_MODE", "true").lower() == "true"
WAIT_TIME = int(os.getenv("WAIT_TIME", "8"))
DELAY_BETWEEN_REQUESTS = int(os.getenv("DELAY_BETWEEN_REQUESTS", "5"))
PDF_DOWNLOAD_TIMEOUT = int(os.getenv("PDF_DOWNLOAD_TIMEOUT", "60"))
DISCORD_REQUEST_TIMEOUT = int(os.getenv("DISCORD_REQUEST_TIMEOUT", "15"))

# Kiểm tra các biến bắt buộc
if not DISCORD_WEBHOOK_URL or not MISTRAL_API_KEY:
    raise ValueError("Vui lòng cấu hình DISCORD_WEBHOOK_URL và MISTRAL_API_KEY trong file .env")

# ==========================================
# KHỞI TẠO AI MODELS
# ==========================================
client_llm = Mistral(api_key=MISTRAL_API_KEY)

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
    """ Extract PDF text with PyMuPDF, then summarize with Mistral AI. """
    import fitz
    
    print("   -> Đang trích xuất text từ PDF với PyMuPDF...")
    try:
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pdf_text = "\n".join([page.get_text("text") for page in pdf_doc])
        pdf_doc.close()
        
        if not pdf_text or len(pdf_text) < 100:
            return "⚠️ PDF không có nội dung hoặc trích xuất bị lỗi."
    except Exception as e:
        print(f"   -> Lỗi khi trích xuất PDF: {e}")
        return "⚠️ Lỗi khi trích xuất text từ PDF."
    
    # Sử dụng system prompt từ .env hoặc mặc định
    if SYSTEM_PROMPT:
        instruction = f"{SYSTEM_PROMPT}\n\nTài liệu: \"{title}\"\n\nNội dung PDF:\n{pdf_text}"
    else:
        instruction = f"Hãy tóm tắt nội dung chính của tài liệu sau:\n\nTài liệu: \"{title}\"\n\nNội dung:\n{pdf_text}"
    
    print("   -> Đang gọi Mistral AI để tóm tắt (Có thể mất 10-30s)...")
    try:
        response = client_llm.chat.complete(
            model=MAIN_MODEL_NAME,
            messages=[{"role": "user", "content": instruction}]
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        print(f"   -> Lỗi khi gọi Mistral AI: {e}")
        return "⚠️ Lỗi khi nhờ AI tóm tắt file PDF."

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