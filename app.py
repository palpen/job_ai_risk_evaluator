import os
import re
import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List

import streamlit as st
from openai import OpenAI

# Optional dependency: pyresparser. Use if available; otherwise, fall back to simple parsing.
try:
    from pyresparser import ResumeParser  # type: ignore
    HAS_PYRESPARSER = True
except Exception:
    ResumeParser = None  # type: ignore
    HAS_PYRESPARSER = False

# Fallback parsers
try:
    import PyPDF2  # type: ignore
except Exception:
    PyPDF2 = None  # type: ignore

docx = None  # python-docx unavailable on this index


ALLOWED_CLASSIFICATIONS = {"Low", "Moderate", "High"}
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = 1.0


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


def _extract_bullet_lines(text: str) -> List[str]:
    bullets: List[str] = []
    bullet_prefixes = ("- ", "• ", "* ", "– ", "— ", "· ")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(bullet_prefixes):
            bullets.append(line.lstrip("-•*–—· ").strip())
    return bullets


def _heuristic_title_and_skills_from_text(text: str) -> Dict[str, Any]:
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]

    # Heuristic job title: first short line that looks like a title
    title_keywords = [
        "engineer",
        "developer",
        "manager",
        "analyst",
        "scientist",
        "designer",
        "consultant",
        "specialist",
        "architect",
        "administrator",
        "coordinator",
        "director",
        "lead",
        "intern",
        "officer",
        "technician",
        "writer",
        "marketer",
        "accountant",
        "lawyer",
        "doctor",
        "nurse",
        "teacher",
    ]

    job_title: Optional[str] = None
    for ln in lines[:25]:
        if len(ln) <= 70:
            lower_ln = ln.lower()
            if any(tok in lower_ln for tok in title_keywords):
                job_title = ln
                break
    if job_title is None and lines:
        job_title = lines[0][:100]

    # Heuristic skills: capture lines after a 'Skills' heading until blank/next heading
    skills: List[str] = []
    skills_section_indices: List[int] = [
        idx for idx, ln in enumerate(lines)
        if ln.lower().strip().startswith("skills") or "technical skills" in ln.lower()
    ]
    if skills_section_indices:
        start = skills_section_indices[0] + 1
        for ln in lines[start: start + 100]:
            if not ln:
                break
            if (ln.isupper() and len(ln.split()) <= 6) or ln.endswith(":"):
                break
            if "," in ln:
                parts = [p.strip() for p in ln.split(",") if p.strip()]
                skills.extend(parts)
            else:
                skills.append(ln)

    if not skills:
        skills = _extract_bullet_lines(text)

    if not skills:
        candidate_lines: List[str] = []
        for ln in lines[:120]:
            if any(x in ln.lower() for x in ["email", "phone", "address", "linkedin", "github"]):
                continue
            if len(ln) < 3:
                continue
            candidate_lines.append(ln)
            if len(candidate_lines) >= 12:
                break
        skills = candidate_lines

    skills = list(dict.fromkeys([s.strip() for s in skills if s.strip()]))

    return {"designation": job_title, "skills": skills, "raw_text": text}


# -------------------- Extended parsing: Experience / Education --------------------
MONTHS_PATTERN = r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
DATE_RANGE_REGEX = re.compile(
    rf"(?i)\b((?:{MONTHS_PATTERN})?\s?\d{{4}}|\d{{4}})\s*(?:–|-|to|—)\s*(Present|(?:{MONTHS_PATTERN})?\s?\d{{4}}|\d{{4}})\b"
)
YEAR_REGEX = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _parse_year_from_fragment(text: str) -> Optional[int]:
    match = YEAR_REGEX.search(text)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            return None
    return None


def _parse_end_year(date_range_text: str) -> int:
    if re.search(r"(?i)present", date_range_text):
        return 9999
    year = _parse_year_from_fragment(date_range_text) or 0
    return year


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return (stripped.isupper() and len(stripped.split()) <= 6) or stripped.endswith(":")


def extract_experience_and_education(raw_text: str) -> Dict[str, Any]:
    lines = [ln.rstrip() for ln in raw_text.splitlines()]
    lines = [ln for ln in lines if ln.strip()]

    exp_section_idx = None
    for i, ln in enumerate(lines):
        low = ln.lower()
        if any(h in low for h in ["experience", "work experience", "professional experience", "employment"]):
            exp_section_idx = i
            break

    education_section_idx = None
    for i, ln in enumerate(lines):
        low = ln.lower()
        if any(h in low for h in ["education", "training", "certifications", "courses"]):
            education_section_idx = i
            break

    experience_entries: List[Dict[str, Any]] = []
    search_start = exp_section_idx + 1 if exp_section_idx is not None else 0
    i = search_start
    current_entry: Optional[Dict[str, Any]] = None
    while i < len(lines):
        ln = lines[i]
        if education_section_idx is not None and i >= education_section_idx:
            break

        if DATE_RANGE_REGEX.search(ln) or (YEAR_REGEX.search(ln) and len(ln) <= 72):
            if current_entry:
                experience_entries.append(current_entry)
            current_entry = {
                "header": ln.strip(),
                "bullets": [],
                "end_year": _parse_end_year(ln),
            }
        elif ln.strip().startswith(("- ", "• ", "* ", "– ", "— ", "· ")) and current_entry:
            current_entry["bullets"].append(ln.lstrip("-•*–—· ").strip())
        elif _looks_like_heading(ln) and current_entry:
            experience_entries.append(current_entry)
            current_entry = None
        else:
            if current_entry and len(ln) <= 120:
                if "title_company" not in current_entry:
                    current_entry["title_company"] = ln.strip()
                else:
                    if not any(x in ln.lower() for x in ["email", "phone", "linkedin", "github", "www."]):
                        current_entry.setdefault("context", []).append(ln.strip())
        i += 1
    if current_entry:
        experience_entries.append(current_entry)

    experience_entries.sort(key=lambda e: e.get("end_year", 0), reverse=True)

    education_items: List[str] = []
    if education_section_idx is not None:
        for ln in lines[education_section_idx + 1 : education_section_idx + 60]:
            if _looks_like_heading(ln):
                break
            if any(x in ln.lower() for x in ["email", "phone", "linkedin", "github", "www."]):
                continue
            education_items.append(ln.strip())

    title_candidates: List[str] = []
    for e in experience_entries[:10]:
        header = e.get("title_company") or e.get("header") or ""
        if header:
            title_candidates.append(header)
    seen = set()
    job_titles = []
    for t in title_candidates:
        if t not in seen:
            job_titles.append(t)
            seen.add(t)

    return {
        "experience": experience_entries,
        "education": education_items,
        "job_titles": job_titles,
    }


def build_recency_weighted_summary(extracted: Dict[str, Any]) -> str:
    job_title = extracted.get("designation") or ""
    raw_text = extracted.get("raw_text") or ""
    sections = extract_experience_and_education(str(raw_text))

    lines: List[str] = []
    if job_title:
        lines.append(f"Primary job title (extracted): {job_title}")

    if sections.get("job_titles"):
        lines.append("Job titles and headers found:")
        for t in sections["job_titles"][:6]:
            lines.append(f" - {t}")

    if sections.get("experience"):
        lines.append("\nExperience (most recent first):")
        for role in sections["experience"][:3]:
            header = role.get("title_company") or role.get("header") or ""
            end_year = role.get("end_year")
            end_label = "Present" if end_year == 9999 else str(end_year)
            lines.append(f" - {header} (thru {end_label})")
            bullets = role.get("bullets", [])
            for b in bullets[:5]:
                lines.append(f"    • {b}")
            ctx = role.get("context", [])
            for c in ctx[:2]:
                lines.append(f"    · {c}")

    if sections.get("education"):
        lines.append("\nEducation and training:")
        for ed in sections["education"][:8]:
            lines.append(f" - {ed}")

    lines.append(
        "\nNote: Place more weight on the most recent experience when judging AI automation risk."
    )

    return "\n".join(lines)


def parse_resume(file_path: str) -> Optional[Dict[str, Any]]:
    if HAS_PYRESPARSER and ResumeParser is not None:
        try:
            parser = ResumeParser(file_path)
            extracted = parser.get_extracted_data()
            if not extracted or not isinstance(extracted, dict):
                return None
            return extracted
        except Exception:
            pass

    suffix = Path(file_path).suffix.lower()
    text = ""
    if suffix == ".pdf":
        text = _extract_text_from_pdf(file_path)
    elif suffix == ".docx":
        # DOCX parsing disabled (python-docx not installed). Convert to PDF instead.
        text = ""
    else:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            text = ""

    if not text.strip():
        return None
    return _heuristic_title_and_skills_from_text(text)


def build_openai_client() -> Optional[OpenAI]:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    return OpenAI()


def classify_role_with_llm(
    client: OpenAI,
    job_title: str,
    skills_text: str,
    enriched_context: Optional[str] = None,
    model: str = MODEL_NAME,
) -> Dict[str, str]:
    system_prompt = (
        "You are an expert career analyst. Your task is to evaluate the provided job "
        "title, responsibilities, experience, and education to classify the likelihood "
        "of the role's tasks being automated by current AI technology."
    )
    context_blob = enriched_context or (
        f"Job Title: {job_title}\nResponsibilities/Skills: {skills_text}"
    )
    user_prompt = (
        f"""Analyze the following resume-derived context and weigh more recent roles more heavily:
{context_blob}

Classify the likelihood that this person's role(s) could be automated by current AI using ONLY one of: Low, Moderate, High.
Then explain your reasoning succinctly in 2-4 sentences, referencing recent responsibilities, tooling, and education/training where relevant.

Output STRICTLY as compact JSON with these keys:
{{"classification": "Low|Moderate|High", "rationale": "<short explanation>"}}
"""
    )

    debug = os.getenv("DEBUG_LLM") == "1"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if debug:
        print("=== LLM Request ===")
        print(f"model: {model}")
        print("system_prompt:\n" + system_prompt)
        print("user_prompt:\n" + user_prompt)

    response = client.chat.completions.create(
        model=model,
        temperature=TEMPERATURE,
        messages=messages,
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
            prompt_rate = os.getenv("OPENAI_PROMPT_RATE")
            completion_rate = os.getenv("OPENAI_COMPLETION_RATE")
            if (
                prompt_rate
                and completion_rate
                and isinstance(prompt_tokens, int)
                and isinstance(completion_tokens, int)
            ):
                try:
                    pr = float(prompt_rate)
                    cr = float(completion_rate)
                    cost = (prompt_tokens / 1000.0) * pr + (completion_tokens / 1000.0) * cr
                    print(f"[LLM Usage] estimated_cost_usd={cost:.6f} (rates: prompt={pr}/1K, completion={cr}/1K)")
                except Exception as _cost_err:
                    print(f"[LLM Usage] rate parsing failed: {_cost_err}")
    except Exception as _usage_err:
        print(f"[LLM Usage] usage read failed: {_usage_err}")

    if not response or not response.choices:
        return {"classification": "Uncertain", "rationale": "No response from the model."}

    content = (response.choices[0].message.content or "").strip()

    if debug:
        print("=== LLM Raw Response ===")
        print(content)

    try:
        data = json.loads(content)
        classification_raw = str(data.get("classification", "")).strip()
        rationale = str(data.get("rationale", "")).strip() or "No explanation provided."
        classification = normalize_classification(classification_raw)
        return {"classification": classification, "rationale": rationale}
    except Exception:
        pass

    classification = normalize_classification(content)
    if classification in ALLOWED_CLASSIFICATIONS:
        return {"classification": classification, "rationale": "Model returned only a label."}

    tokens = content.split()
    if tokens:
        guessed = normalize_classification(tokens[0])
        remaining = content[len(tokens[0]):].strip()
        if guessed in ALLOWED_CLASSIFICATIONS:
            return {"classification": guessed, "rationale": remaining or "Model returned a label and minimal text."}

    return {"classification": "Uncertain", "rationale": "Unable to parse the model output reliably."}


def normalize_classification(raw_text: str) -> str:
    if not raw_text:
        return "Uncertain"
    cleaned = (
        raw_text.strip().strip(".").strip().strip('"').strip("'").title()
    )
    if cleaned in ALLOWED_CLASSIFICATIONS:
        return cleaned
    lowered = cleaned.lower()
    if lowered.startswith("low"):
        return "Low"
    if lowered.startswith("mod"):
        return "Moderate"
    if lowered.startswith("high"):
        return "High"
    return "Uncertain"


def format_skills(skills_value: Any) -> str:
    if skills_value is None:
        return ""
    if isinstance(skills_value, list):
        return ", ".join([str(s) for s in skills_value if s])
    return str(skills_value)


def main() -> None:
    st.set_page_config(page_title="AI Job Automation Likelihood Classifier", page_icon="🤖")
    st.title("AI Job Automation Likelihood Classifier")
    st.write(
        "This tool classifies the likelihood that a role's tasks could be automated by AI. "
        "Results are speculative, generated by an LLM, and should not be treated as definitive."
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

    job_title: Optional[str] = (
        extracted.get("designation")
        or extracted.get("job_title")
        or extracted.get("occupation")
    )
    skills_text: str = format_skills(extracted.get("skills"))
    raw_text: str = str(extracted.get("raw_text", ""))

    if not job_title and not skills_text.strip():
        st.error(
            "Could not extract job title or responsibilities from the resume. Please upload a clearer PDF."
        )
        return

    st.subheader("Extracted from resume")
    if job_title:
        st.markdown(f"- **Primary Job Title**: {job_title}")
    else:
        st.markdown("- **Primary Job Title**: (not found)")

    responsibilities_list: List[str] = [s.strip() for s in (extracted.get("skills") or []) if str(s).strip()] if isinstance(extracted.get("skills"), list) else []
    if responsibilities_list:
        to_show = responsibilities_list[:12]
        st.markdown("- **Responsibilities/Skills (sample)**:")
        for item in to_show:
            st.markdown(f"  - {item}")
        if len(responsibilities_list) > len(to_show):
            st.caption(f"(+{len(responsibilities_list) - len(to_show)} more)")
    else:
        st.markdown("- **Responsibilities/Skills**: (not found)")

    sections = extract_experience_and_education(raw_text)
    if sections.get("experience"):
        st.markdown("- **Experience (most recent first)**:")
        for role in sections["experience"][:3]:
            header = role.get("title_company") or role.get("header") or "(role)"
            end_year = role.get("end_year")
            end_label = "Present" if end_year == 9999 else (str(end_year) if end_year else "")
            st.markdown(f"  - {header} {f'(thru {end_label})' if end_label else ''}")
            for b in (role.get("bullets") or [])[:4]:
                st.markdown(f"    • {b}")
    if sections.get("education"):
        st.markdown("- **Education/Training**:")
        for ed in sections["education"][:5]:
            st.markdown(f"  - {ed}")

    client = build_openai_client()
    if client is None:
        st.error(
            "OPENAI_API_KEY is not set. Please set the environment variable and restart the app."
        )
        return

    with st.spinner("Evaluating role with AI…"):
        try:
            enriched = build_recency_weighted_summary({
                "designation": job_title or "",
                "skills": extracted.get("skills"),
                "raw_text": raw_text,
            })
            if os.getenv("DEBUG_LLM") == "1":
                print("=== LLM Inputs ===")
                print(f"job_title: {job_title or ''}")
                print(f"skills_text: {skills_text}")
                print("enriched_context:\n" + enriched)
            result = classify_role_with_llm(
                client,
                job_title or "",
                skills_text,
                enriched_context=enriched,
                model=MODEL_NAME,
            )
            classification = result.get("classification", "Uncertain")
            rationale = result.get("rationale", "")
        except Exception as e:
            st.error("There was an error contacting the AI service. Please try again later.")
            st.caption("Debug info (model and error message):")
            st.code(f"model={MODEL_NAME}\nerror={str(e)}")
            return

    st.subheader("AI Evaluation")
    st.markdown(f"**Classification:** {classification}")
    if rationale:
        st.markdown("**Explanation:**")
        st.write(rationale)


if __name__ == "__main__":
    main()

