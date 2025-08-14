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
- The app prints token usage to the terminal. If you set the per-1K token rates, it will also print a simple cost estimate per request.

