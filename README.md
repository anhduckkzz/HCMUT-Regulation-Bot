# HCMUT Smart Assistant

**Multi-Phase AI Project:** Intelligent assistant for Ho Chi Minh City University of Technology (HCMUT) students

## Project Roadmap

### Phase 1: Auto Regulation Monitor (Completed)
Automated tool that monitors and notifies new HCMUT regulations via Discord.

### Phase 2: RAG Chatbot (Coming Soon)
Intelligent chatbot with RAG (Retrieval-Augmented Generation) integration:
- Answer questions about HCMUT regulations 
- Search information in document database
- Vector database for regulation documents
- Context-aware responses with Gemini AI

## Phase 1 - Current Features

- Automated web scraping from HCMUT regulation page
- Download and analyze PDF files from Google Drive  
- Uses Gemini AI to read and summarize content
- Send automatic notifications to Discord with beautiful formatting
- Avoid duplicate notifications with tracking system
- Supports fallback model when main model fails

## Installation

### 1. Clone repository
```bash
git clone <repository-url>
cd tool
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure .env file
Copy `.env.example` to `.env` and fill in the information according to the guide below.

## API Keys & Webhooks Setup Guide

### Discord Webhook URL

#### Step 1: Create Webhook in Discord Server
1. Open Discord and go to your desired server
2. Right-click on the text channel you want notifications in
3. Select **"Edit Channel"**
4. Switch to **"Integrations"** tab
5. Click **"Create Webhook"**

#### Step 2: Configure Webhook
1. Set bot name: `HCMUT Regulation Bot`
2. Copy **Webhook URL**
3. Paste into `.env` file at `DISCORD_WEBHOOK_URL`

**Example:**
```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/123456789/abcdef...
```

### Google Gemini API Key

#### Step 1: Access Google AI Studio
1. Go to: https://aistudio.google.com/
2. Sign in with your Google account

#### Step 2: Create API Key
1. Click **"Get API key"** in the top menu
2. Select **"Create API key"**
3. Choose Google Cloud Project (or create new)
4. Copy the generated API key

#### Step 3: Enable service (if needed)
- Visit: https://console.cloud.google.com/
- Search for **"Generative Language API"** 
- Click **"Enable"** to activate

**Example:**
```
GEMINI_API_KEY=
```

### Security Notes
- **DO NOT** commit `.env` file to Git
- **DO NOT** share API keys or webhook URLs
- Add `.env` to your `.gitignore` file

## Configuration

### AI Models
```env
MAIN_MODEL=gemini-3-flash-preview      # Main model (latest, fast)
FALLBACK_MODEL=gemini-2.5-flash        # Backup model (stable)
```

### System Prompt
You can customize the prompt for AI analysis:
```env
SYSTEM_PROMPT=Your custom prompt here...
```

### Other Settings
```env
STATE_FILE=seen_laws.json              # State tracking file   
HEADLESS_MODE=true                     # Run browser headless
WAIT_TIME=8                            # Page load wait time (seconds)
DELAY_BETWEEN_REQUESTS=5               # Delay between requests (seconds)
PDF_DOWNLOAD_TIMEOUT=60                # PDF download timeout (seconds)
DISCORD_REQUEST_TIMEOUT=15             # Discord request timeout (seconds)
```

## Usage

### Manual Run
```bash
python regulation.py
```

### Automated with Task Scheduler (Windows)
1. Open **Task Scheduler**
2. Create **Basic Task**
3. Set schedule (e.g., every 6 hours)
4. Action: Start Program
   - Program: `python.exe`
   - Arguments: `regulation.py` 
   - Start in: `d:\tool`

### Automated with Cron job (Linux/Mac)
```bash
# Add to crontab (run 4 times per day)
0 */6 * * * cd /path/to/tool && python regulation.py
```

## Phase 2 Development Plan

### RAG Chatbot Architecture (Upcoming)
```
hcmut-smart-assistant/
├── phase1-monitor/          # Current regulation monitor
│   ├── regulation.py
│   ├── .env
│   └── seen_laws.json
├── phase2-chatbot/          # RAG chatbot system  
│   ├── rag_engine.py        # RAG core logic
│   ├── vector_store.py      # Vector database management
│   ├── chat_interface.py    # Discord bot interface
│   └── document_processor.py # PDF to vector conversion
├── database/
│   ├── regulations.db       # SQLite for metadata
│   └── vectors/             # Embeddings storage
└── shared/
    ├── .env                 # Shared config
    └── utils.py            # Common utilities
```

### Phase 2 Features
- **Seamless Integration:** Monitor + Chatbot in one system
- **Vector Search:** Semantic search in regulation documents
- **Discord Bot:** Direct reply to regulation questions
- **Smart Context:** RAG for accurate answers from reliable sources
- **Auto Learning:** Automatically update knowledge base from new regulations

## Project Structure

```
hcmut-smart-assistant/
├── regulation.py          # Main code
├── .env                   # Configuration (do not commit)
├── .env.example          # Configuration template
├── seen_laws.json        # State file (auto-created)
├── test.py              # Test file (if any)
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This documentation
```

## Sample Output

When new regulations are detected, the bot sends Discord embeds like this:

```
NEW HCMUT REGULATION: Decision 123/QD-DHBK-DT

AI FULL TEXT SUMMARY:

• Updated attendance policy: From 80% to 85% 
• Added online assignment submission via LMS
• Updated grading scale from 10-point to 4.0 scale
• Details in Appendix A: New grade conversion table
...
```

## Troubleshooting

### Common Issues

**1. Selenium can't find Chrome driver**
```bash
pip install --upgrade webdriver-manager
```

**2. Gemini API quota exceeded**
- Check quota at: https://console.cloud.google.com/
- Wait for quota reset or upgrade plan

**3. Discord webhook 429 (rate limit)**
- Increase `DELAY_BETWEEN_REQUESTS` in `.env`
- Reduce script execution frequency

**4. PDF download fails**
- Check Google Drive link permissions
- Increase `PDF_DOWNLOAD_TIMEOUT`

### Logs and Debug
Script prints detailed logs to console. For debugging:

```bash
python regulation.py > logs.txt 2>&1
```

## Requirements.txt

```txt
requests>=2.31.0
google-generativeai>=0.3.2
selenium>=4.15.0
webdriver-manager>=4.0.1
python-dotenv>=1.0.0
```

## Tech Stack

### Phase 1 (Current)
- **Language:** Python 3.8+
- **Web Scraping:** Selenium WebDriver
- **AI Processing:** Google Gemini AI
- **Notifications:** Discord Webhooks
- **File Processing:** PDF handling via Gemini

### Phase 2 (Planned)
- **Vector Database:** ChromaDB or Pinecone
- **RAG Framework:** LangChain or LlamaIndex
- **Bot Framework:** Discord.py
- **Embeddings:** Google Text Embeddings API
- **Search Engine:** Semantic similarity search

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -m 'Add new feature'`
4. Push branch: `git push origin feature/new-feature`
5. Create Pull Request

## License

MIT License - Free to use for educational and research purposes.

## Credits

- **HCMUT**: Data source for regulations
- **Google Gemini**: AI content analysis
- **Discord**: Notification platform
- **Selenium**: Web scraping tool

---
**Made with love for HCMUT students**