import os
import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List

import streamlit as st
from openai import OpenAI

# Note: We intentionally do not use rule-based or third-party resume parsers.

# Fallback parsers
try:
    import PyPDF2  # type: ignore
except Exception:
    PyPDF2 = None  # type: ignore

docx = None  # python-docx unavailable on this index


ALLOWED_CLASSIFICATIONS = {"Very Low", "Low", "Moderate", "High", "Very High"}
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = 1.0


def _hydrate_env_from_streamlit_secrets() -> None:
    """Load required secrets into environment when running on Streamlit Cloud.

    This lets existing getenv-based code paths continue to work without refactor.
    """
    try:
        # Only set if not already present in the environment
        if "OPENAI_API_KEY" in st.secrets and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = str(st.secrets["OPENAI_API_KEY"]).strip()
        if "OPENAI_MODEL" in st.secrets and not os.getenv("OPENAI_MODEL"):
            os.environ["OPENAI_MODEL"] = str(st.secrets["OPENAI_MODEL"]).strip()
    except Exception:
        # st.secrets may not be available locally without a secrets.toml
        pass


_hydrate_env_from_streamlit_secrets()

def save_uploaded_file_to_temp(uploaded_file) -> str:
    file_suffix = Path(uploaded_file.name).suffix.lower() or ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name


def _extract_text_from_pdf(file_path: str) -> str:
    if PyPDF2 is None:
        return ""
    try:
        text_parts: List[str] = []
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in getattr(reader, "pages", []) or []:
                try:
                    text_parts.append(page.extract_text() or "")
                except Exception:
                    continue
        return "\n".join([t for t in text_parts if t]).strip()
    except Exception:
        return ""


def _extract_text_from_docx(file_path: str) -> str:
    if docx is None:
        return ""
    try:
        document = docx.Document(file_path)
        paragraphs = [p.text for p in document.paragraphs if p.text]
        return "\n".join(paragraphs).strip()
    except Exception:
        return ""


def extract_raw_text(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return _extract_text_from_pdf(file_path)
    if suffix == ".docx":
        return _extract_text_from_docx(file_path)
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def analyze_resume_with_llm(client: OpenAI, raw_text: str, model: str) -> Dict[str, Any]:
    system_prompt = (
        "You are an expert career analyst system. Your purpose is to parse raw resume text, "
        "extract key professional information, and classify the role's automation risk based on a defined framework."
    )
    user_prompt = f"""--- Analysis Framework & Definitions ---

AI Capability Definition: For this analysis, "AI" refers to current models capable of advanced text/code generation, data analysis, pattern recognition, and process automation. It does not include physical robotics or Artificial General Intelligence (AGI).

Classification Rubric: You must use the following five-level scale for your classification.

Very High: The role's core functions are almost entirely digital, repetitive, and follow predictable patterns that can be fully automated with current AI. (e.g., Data Entry, Transcription, Basic Customer Service Chat).

High: The majority of core tasks are automatable (e.g., content generation, data analysis, report summarization), but the role may include minor tasks requiring human oversight or simple judgment. (e.g., Financial Analyst, Digital Marketer, Copywriter).

Moderate: The role contains a significant mix of automatable tasks and responsibilities that require human-centric skills like strategic planning, complex problem-solving, or nuanced interpersonal communication. (e.g., Product Manager, HR Manager, Graphic Designer).

Low: The majority of core tasks depend on high-stakes human judgment, strategic leadership, client relationship management, or creative work that is not easily replicated. AI can act as a tool but cannot replace the core function. (e.g., Engineering Manager, Lawyer, Senior Sales Executive).

Very Low: The role is fundamentally grounded in physical interaction, high-level strategic direction for an entire organization, or deep emotional intelligence and empathy. (e.g., Surgeon, CEO, Licensed Therapist, Skilled Tradesperson).

--- Few-Shot Examples ---

# Example 1: High Automation Risk

Resume Text:

Alex Chen - Content Specialist

A data-driven content creator with 4 years of experience in the tech industry. Proven ability to grow organic traffic through targeted SEO strategies and compelling blog content.

Experience:
Content Marketing Specialist, DataCorp (2021-Present)
- Write and publish 4-5 SEO-optimized blog posts per week using Clearscope and SurferSEO.
- Manage the corporate social media calendar across Twitter and LinkedIn, scheduling posts with Buffer.
- Analyze content performance using Google Analytics and create monthly traffic reports.
- Increased organic blog traffic by 150% in 18 months.

Education:
B.A. in Communications, State University

Output:

{{
  "job_title": "Content Marketing Specialist",
  "skills": ["Content Creation", "SEO", "Google Analytics", "Social Media Management", "Buffer", "Clearscope", "Data Analysis", "Reporting"],
  "recent_experience": [
    "Writes and publishes 4-5 SEO-optimized blog posts weekly.",
    "Manages social media calendar and post scheduling.",
    "Analyzes content performance and creates monthly reports using Google Analytics."
  ],
  "classification": "High",
  "rationale": "The core responsibilities, such as writing SEO-optimized content, managing social media schedules, and generating performance reports, are all tasks that can be significantly automated by current AI content generation and analytics tools."
}}

# Example 2: Low Automation Risk

Resume Text:

Samantha Rivera

Senior Engineering Manager with over 12 years of experience leading cross-functional software development teams in agile environments. Expert in strategic planning, talent development, and stakeholder management for large-scale SaaS products.

Experience:
Senior Engineering Manager, Innovate Inc. (2018-Present)
- Lead a team of 15 software engineers responsible for the flagship SaaS platform.
- Collaborate with product and design leadership to define the long-term technical roadmap and strategy.
- Conduct performance reviews, mentor junior engineers, and manage hiring and team growth.
- Mediate technical disagreements and align engineering teams on architectural decisions.
- Successfully delivered three major product releases ahead of schedule.

Lead Software Engineer, Tech Solutions LLC (2014-2018)

Output:

{{
  "job_title": "Senior Engineering Manager",
  "skills": ["Team Leadership", "Strategic Planning", "Talent Development", "Mentoring", "Hiring", "Stakeholder Management", "Agile Methodologies", "SaaS", "Software Development"],
  "recent_experience": [
    "Leads a team of 15 software engineers.",
    "Defines long-term technical roadmap and strategy in collaboration with product and design.",
    "Manages team performance, mentorship, and hiring.",
    "Mediates technical disagreements and aligns teams on architectural decisions."
  ],
  "classification": "Low",
  "rationale": "The role's primary functions revolve around strategic leadership, team management, mentorship, and complex stakeholder negotiation. These tasks require a high degree of human judgment and interpersonal skill that cannot be automated by current AI."
}}

--- Your Task ---

Now, process the following resume text. Perform the following steps:

Extract Information:

job_title: Extract the most relevant and recent job title.

skills: Extract up to 20 key skills as a list of short phrases.

recent_experience: Summarize the most recent experience into 3-8 impactful bullet points.

Classify and Justify:

classification: Assign an automation likelihood classification using the rubric provided.

rationale: Write a 2-4 sentence explanation for your classification.

Your output must be a single, compact JSON object, following the exact structure of the examples.

Resume Text:

{raw_text}

Output:
Your response MUST be a single, compact JSON object and nothing else.
"""

    response = client.chat.completions.create(
        model=model,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    # Token usage logging for cost debugging
    try:
        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
            print(
                f"[LLM Usage] model={model} prompt_tokens={prompt_tokens} "
                f"completion_tokens={completion_tokens} total_tokens={total_tokens}"
            )
    except Exception:
        pass

    content = (response.choices[0].message.content or "").strip() if response and response.choices else ""
    try:
        data = json.loads(content)
    except Exception:
        data = {}

    # Normalize classification
    data["classification"] = normalize_classification(str(data.get("classification", "")))
    return data


def parse_resume(file_path: str) -> Optional[Dict[str, Any]]:
    # Simplified: only return raw_text; no rule-based extraction
    text = extract_raw_text(file_path)
    if not text.strip():
        return None
    return {"raw_text": text}


def build_openai_client() -> Optional[OpenAI]:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    return OpenAI()


# Removed older 3-class classifier; we rely on analyze_resume_with_llm only.


def normalize_classification(raw_text: str) -> str:
    if not raw_text:
        return "Uncertain"
    cleaned = (
        raw_text.strip().strip(".").strip().strip('"').strip("'").title()
    )
    if cleaned in ALLOWED_CLASSIFICATIONS:
        return cleaned
    lowered = cleaned.lower()
    if lowered.startswith("very low"):
        return "Very Low"
    if lowered.startswith("very high"):
        return "Very High"
    if lowered.startswith("low"):
        return "Low"
    if lowered.startswith("mod"):
        return "Moderate"
    if lowered.startswith("high"):
        return "High"
    return "Uncertain"


# No longer needed: rule-based skills formatting


def main() -> None:
    st.set_page_config(page_title="Will AI Take My Job?", page_icon="🤖")
    st.title("Will AI Take My Job?")
    st.write(
        "This tool classifies the likelihood that a role's tasks could be automated by AI. "
        "Results are speculative, generated by an LLM, and should not be treated as definitive."
        "\n\nThe resume data is sent to an OpenAI LLM model to do the analysis. No data is stored or shared with anyone else."
    )

    uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "docx"], key="resume_uploader")

    if uploaded_file is None:
        return

    with st.spinner("Parsing your resume…"):
        temp_path = save_uploaded_file_to_temp(uploaded_file)
        try:
            extracted = parse_resume(temp_path)
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass

    if not extracted:
        st.error("Could not parse resume. Please check the file format.")
        return

    raw_text: str = str(extracted.get("raw_text", ""))
    if not raw_text.strip():
        st.error("Could not read text from resume. Please upload a text-based PDF.")
        return

    client = build_openai_client()
    if client is None:
        st.error(
            "OPENAI_API_KEY is not set. Please set the environment variable and restart the app."
        )
        return

    with st.spinner("Evaluating likelihood of your job being automated by AI…"):
        try:
            if os.getenv("DEBUG_LLM") == "1":
                print("=== LLM Inputs (raw resume) ===")
                print(raw_text[:2000])
            parsed = analyze_resume_with_llm(client, raw_text, MODEL_NAME)
            job_title = parsed.get("job_title")
            skills_list = parsed.get("skills") or []
            recent_experience = parsed.get("recent_experience") or []
            classification = parsed.get("classification", "Uncertain")
            rationale = parsed.get("rationale", "")
        except Exception as e:
            st.error("There was an error contacting the AI service. Please try again later.")
            st.caption("Debug info (model and error message):")
            st.code(f"model={MODEL_NAME}\nerror={str(e)}")
            return

    # Show parsed info from the model
    st.subheader("Extracted from resume (AI)")
    if job_title:
        st.markdown(f"- **Primary Job Title**: {job_title}")
    if skills_list:
        st.markdown("- **Key Skills**:")
        for s in skills_list[:15]:
            st.markdown(f"  - {s}")
        if len(skills_list) > 15:
            st.caption(f"(+{len(skills_list)-15} more)")
    if recent_experience:
        st.markdown("- **Recent Experience**:")
        for b in recent_experience[:10]:
            st.markdown(f"  - {b}")

    st.subheader("AI Evaluation")
    st.markdown(f"**How likely is your job to be automated by AI?** {classification}")
    if rationale:
        st.markdown("**Explanation:**")
        st.write(rationale)


if __name__ == "__main__":
    main()

