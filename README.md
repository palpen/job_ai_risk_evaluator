# AI Job Automation Likelihood Classifier

Is your job at risk of being replaced by an AI? Upload your resume to find out.

The resume data is sent to an OpenAI LLM model to do the analysis. No data is stored or shared with anyone else.

![Demo](./demo.gif)

## Requirements
- Python 3.9+
- OpenAI API key (`OPENAI_API_KEY`)

## Setup
```bash
# From the project root
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Optional (if you enable pyresparser in your environment)
python -m spacy download en_core_web_sm
python -m nltk.downloader words
```

## Run
```bash
# Required
export OPENAI_API_KEY="sk-..."

# Optional Configuration
export OPENAI_MODEL="gpt-4o-mini"           # Default model (set to model available to your account)

# Security & Rate Limiting (optional)
export RATE_LIMIT_MAX_REQUESTS=3            # Max requests per time window (default: 3)
export RATE_LIMIT_WINDOW_SECONDS=60         # Time window in seconds (default: 60)

# Debug Mode (optional) - Enables advanced features
export DEBUG_LLM=1                          # Enable debug logging and cache statistics

# Start the app
streamlit run app.py
```

Then open the provided local URL in your browser and upload a resume (PDF, DOCX, or TXT supported).

## Features

### 🔒 **Security & Privacy**
- **File validation**: MIME type verification, size limits (10MB max), PDF signature checking
- **In-memory processing**: No temporary files created, enhanced security
- **PII protection**: Resume content never logged, only metadata for debugging
- **Rate limiting**: Per-session API call limits to prevent abuse

### ⚡ **Performance & Caching**
- **Intelligent caching**: Identical resumes cached for 24 hours to save API costs
- **LRU eviction**: Automatic cache management (100-entry limit per session)
- **Text optimization**: Smart truncation and page limits to control processing costs

### 📊 **Debug Mode** (when `DEBUG_LLM=1`)
- **Cache statistics**: Hit rates, cache usage, performance metrics
- **Debug sidebar**: Real-time cache management and statistics
- **Cache controls**: Manual cache clearing for testing
- **Detailed metrics**: API usage tracking without PII exposure

### 📁 **File Support**
- **PDF**: Advanced text extraction with encryption detection and error handling
- **TXT**: Plain text file support with encoding detection
- **DOCX**: Currently disabled for security (can be enabled if needed)

## Notes
- This app is for informational purposes. Classifications are speculative LLM outputs, not definitive advice.
- Resume data is sent to OpenAI for analysis but not stored locally or shared with third parties.
- The app includes comprehensive security measures and works entirely in-memory for enhanced safety.

