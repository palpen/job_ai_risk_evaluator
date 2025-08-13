# AI Job Automation Likelihood Classifier

A minimal Streamlit app that analyzes a resume and classifies the likelihood that the job(s) described could be automated by current AI: Low, Moderate, or High. The app extracts relevant resume details (job titles, recent responsibilities/bullets, education/training), emphasizes recent experience, and asks an OpenAI model for a classification with a brief rationale.

## Features
- Upload resume (PDF or DOCX)
- Heuristic parsing with recency emphasis (shows extracted title, roles, bullets, education)
- LLM-based classification and short explanation
- Token usage printed to the terminal for cost tracking (optional per-1K cost estimate)
- Configurable model via `OPENAI_MODEL`

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

# Optional (default is gpt-4o-mini; set to a model available to your account)
export OPENAI_MODEL="gpt-4o-mini"

# Optional: debug logs and token cost estimates in terminal
export DEBUG_LLM=1
export OPENAI_PROMPT_RATE=0.005       # USD per 1K input tokens (example)
export OPENAI_COMPLETION_RATE=0.015   # USD per 1K output tokens (example)

# Start the app
streamlit run app.py
```

Then open the provided local URL in your browser, upload a resume (prefer PDF for best extraction), review the extracted info, and see the model’s classification and explanation.

## Notes
- This app is for informational purposes. Classifications are speculative LLM outputs, not definitive advice.
- PDFs generally extract better than scanned images. If using scanned PDFs, OCR first.
- The app prints token usage to the terminal. If you set the per-1K token rates, it will also print a simple cost estimate per request.
- Do not commit real resumes to Git; `*.pdf` is ignored by default in `.gitignore`.


