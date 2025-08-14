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
        "You are an expert career analyst. Parse the resume text and extract key information, "
        "then assess AI automation risk."
    )
    user_prompt = f"""Resume Text:
{raw_text}

Tasks:
1) Extract a concise job_title for the candidate (best current/most recent fit).
2) Extract up to 20 key skills as a list of short phrases.
3) Summarize recent experience as 3-8 bullet points highlighting impactful responsibilities.
4) Provide a classification of AI automation likelihood using ONLY one of: Very Low, Low, Moderate, High, Very High.
5) Provide a brief rationale (2-4 sentences) referencing recent responsibilities or tools.

Output strictly as compact JSON with the following keys:
{{
  "job_title": "<string>",
  "skills": ["..."],
  "recent_experience": ["..."],
  "classification": "Very Low|Low|Moderate|High|Very High",
  "rationale": "<short explanation>"
}}
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

