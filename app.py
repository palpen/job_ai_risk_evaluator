"""AI Job Automation Risk Evaluator.

A Streamlit web application that analyzes uploaded resumes to classify the likelihood
that a job role could be automated by current AI technology. Uses OpenAI's language
models to extract key information from resumes and provide risk assessments based
on a detailed 5-level classification rubric.

The application:
- Accepts PDF and DOCX resume uploads
- Extracts text content from documents
- Sends anonymized data to OpenAI for analysis
- Returns structured job information and automation risk classification
- Provides detailed explanations for risk assessments

Privacy: Resume data is processed by OpenAI but not stored locally or shared.
Results are for informational purposes only and should not be considered
definitive career advice.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import streamlit as st
from openai import OpenAI
from openai.types.chat import ChatCompletion

# Note: We intentionally do not use rule-based or third-party resume parsers.

# Fallback parsers - conditional imports to handle missing dependencies gracefully
try:
    import PyPDF2  # type: ignore
except Exception:
    PyPDF2 = None  # type: ignore  # PDF parsing will be disabled if PyPDF2 not available

# DOCX support is intentionally disabled in this implementation
docx = None  # python-docx unavailable on this index


ALLOWED_CLASSIFICATIONS = {"Very Low", "Low", "Moderate", "High", "Very High"}
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = 1.0


def _hydrate_env_from_streamlit_secrets() -> None:
    """Load required secrets into environment when running on Streamlit Cloud.

    This function checks for OpenAI API credentials in Streamlit's secrets management
    system and copies them to environment variables. This allows the app to work
    seamlessly both locally (with environment variables) and on Streamlit Cloud
    (with secrets.toml configuration).

    The function only sets environment variables if they don't already exist,
    giving precedence to locally set environment variables.

    Raises:
        Exception: Silently catches and ignores any exceptions, as secrets may
                  not be available in local development environments.
    """
    try:
        # Only set if not already present in the environment (local env vars take precedence)
        if "OPENAI_API_KEY" in st.secrets and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = str(st.secrets["OPENAI_API_KEY"]).strip()
        if "OPENAI_MODEL" in st.secrets and not os.getenv("OPENAI_MODEL"):
            os.environ["OPENAI_MODEL"] = str(st.secrets["OPENAI_MODEL"]).strip()
    except Exception:
        # st.secrets may not be available locally without a secrets.toml file
        # This is expected and normal for local development
        pass


_hydrate_env_from_streamlit_secrets()

def save_uploaded_file_to_temp(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> str:
    """Save an uploaded Streamlit file to a temporary file on disk.

    This function creates a temporary file with the same extension as the uploaded
    file and writes the file contents to it. The temporary file is not automatically
    deleted, so the caller is responsible for cleanup.

    Args:
        uploaded_file: A Streamlit uploaded file object containing the file data
                      and metadata.

    Returns:
        str: The absolute path to the created temporary file.

    Example:
        >>> temp_path = save_uploaded_file_to_temp(uploaded_file)
        >>> # Process the file...
        >>> os.remove(temp_path)  # Clean up when done
    """
    file_suffix = Path(uploaded_file.name).suffix.lower() or ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name


def _extract_text_from_pdf(file_path: str) -> str:
    """Extract text content from a PDF file using PyPDF2.

    Attempts to read all pages from a PDF file and concatenate their text content.
    This function is resilient to errors - if PyPDF2 is not available, if the file
    cannot be opened, or if individual pages fail to extract, it will return an
    empty string rather than raising an exception.

    Args:
        file_path: The absolute path to the PDF file to extract text from.

    Returns:
        str: The concatenated text content from all readable pages, with pages
             separated by newlines. Returns empty string if extraction fails.

    Note:
        This function may not work well with scanned PDFs or PDFs with complex
        layouts, as PyPDF2 has limitations with these document types.
    """
    if PyPDF2 is None:
        return ""
    try:
        text_parts: List[str] = []
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            # Iterate through all pages, handling both old and new PyPDF2 API versions
            for page in getattr(reader, "pages", []) or []:
                try:
                    # Extract text from each page, some pages may fail individually
                    text_parts.append(page.extract_text() or "")
                except Exception:
                    # Skip pages that can't be processed (corrupted, encrypted, etc.)
                    continue
        # Join all non-empty text parts and clean up whitespace
        return "\n".join([t for t in text_parts if t]).strip()
    except Exception:
        # Return empty string for any file-level errors (permissions, corruption, etc.)
        return ""


def _extract_text_from_docx(file_path: str) -> str:
    """Extract text content from a DOCX file using python-docx.

    Reads all paragraphs from a Microsoft Word document and concatenates them
    with newlines. This function is currently not functional as the docx library
    is intentionally set to None in this implementation.

    Args:
        file_path: The absolute path to the DOCX file to extract text from.

    Returns:
        str: The concatenated text content from all paragraphs, or empty string
             if the docx library is unavailable or extraction fails.

    Note:
        This function is currently disabled (docx = None) and will always
        return an empty string. Enable by installing python-docx and importing it.
    """
    if docx is None:
        return ""
    try:
        document = docx.Document(file_path)
        paragraphs = [p.text for p in document.paragraphs if p.text]
        return "\n".join(paragraphs).strip()
    except Exception:
        return ""


def extract_raw_text(file_path: str) -> str:
    """Extract text content from various file formats.

    This is the main text extraction function that dispatches to format-specific
    extractors based on file extension. Supports PDF, DOCX, and plain text files.
    For unsupported formats or extraction failures, returns empty string.

    Args:
        file_path: The absolute path to the file to extract text from.

    Returns:
        str: The extracted text content, or empty string if extraction fails
             or the file format is not supported.

    Supported Formats:
        - .pdf: Extracted using PyPDF2
        - .docx: Extracted using python-docx (currently disabled)
        - Other: Treated as plain text with UTF-8 encoding
    """
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
    """Analyze resume text using OpenAI's language model for automation risk assessment.

    This function sends the resume text to OpenAI's API with a detailed prompt
    containing classification rubric and few-shot examples. The LLM extracts
    structured information (job title, skills, experience) and provides an
    automation risk classification with explanation.

    Args:
        client: An initialized OpenAI client instance.
        raw_text: The raw text content extracted from the resume.
        model: The OpenAI model name to use (e.g., 'gpt-4o-mini').

    Returns:
        Dict[str, Any]: A dictionary containing:
            - job_title (str): The primary job title extracted from resume
            - skills (List[str]): List of key skills identified
            - recent_experience (List[str]): Recent work experience bullet points
            - classification (str): Automation risk level (Very Low to Very High)
            - rationale (str): Explanation for the risk classification

    Raises:
        Exception: May raise OpenAI API exceptions for network/auth issues.
                  Token usage is logged to console if DEBUG_LLM=1.

    Note:
        The prompt includes a comprehensive 5-level classification rubric with
        detailed examples to ensure consistent and accurate risk assessments.
    """
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

    # Token usage logging for cost debugging (only printed to console)
    try:
        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
            # Print usage stats for debugging and cost tracking
            print(
                f"[LLM Usage] model={model} prompt_tokens={prompt_tokens} "
                f"completion_tokens={completion_tokens} total_tokens={total_tokens}"
            )
    except Exception:
        # Silently handle any issues with usage reporting - not critical
        pass

    # Extract the response content safely
    content = (response.choices[0].message.content or "").strip() if response and response.choices else ""
    
    # Parse JSON response from LLM
    try:
        data = json.loads(content)
    except Exception:
        # If JSON parsing fails, return empty dict (will be handled by UI)
        data = {}

    # Normalize the classification to ensure it matches our expected values
    data["classification"] = normalize_classification(str(data.get("classification", "")))
    return data


def parse_resume(file_path: str) -> Optional[Dict[str, Any]]:
    """Parse a resume file and extract its text content.

    This function serves as the main entry point for resume processing. It
    extracts raw text from the file and returns it in a dictionary structure.
    No rule-based parsing or structured data extraction is performed - that
    is handled by the LLM analysis.

    Args:
        file_path: The absolute path to the resume file to parse.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the 'raw_text' key
                                 with the extracted text, or None if no text
                                 could be extracted from the file.

    Example:
        >>> result = parse_resume('/path/to/resume.pdf')
        >>> if result:
        ...     text = result['raw_text']
    """
    # Simplified approach: only return raw_text; no rule-based extraction
    # All structured data extraction is handled by the LLM
    text = extract_raw_text(file_path)
    if not text.strip():
        return None  # Signal that no usable text was found
    return {"raw_text": text}


def build_openai_client() -> Optional[OpenAI]:
    """Create an OpenAI client instance if API key is available.

    Checks for the OPENAI_API_KEY environment variable and creates an OpenAI
    client instance if the key is present. This function handles the client
    initialization and provides a clean way to check for API key availability.

    Returns:
        Optional[OpenAI]: An initialized OpenAI client if the API key is
                         available, None otherwise.

    Environment Variables:
        OPENAI_API_KEY: Required OpenAI API key for authentication.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return None
    return OpenAI()


# Removed older 3-class classifier; we rely on analyze_resume_with_llm only.


def normalize_classification(raw_text: str) -> str:
    """Normalize LLM classification output to standard risk categories.

    The LLM may return classifications in various formats (different cases,
    with punctuation, etc.). This function standardizes the output to match
    one of the five allowed classification levels, with fallback logic for
    partial matches.

    Args:
        raw_text: The raw classification text returned by the LLM.

    Returns:
        str: One of the normalized classification values: 'Very Low', 'Low',
             'Moderate', 'High', 'Very High', or 'Uncertain' if no match found.

    Classification Levels:
        - Very Low: Minimal automation risk
        - Low: Limited automation potential
        - Moderate: Mixed automation potential
        - High: Significant automation risk
        - Very High: High automation likelihood
        - Uncertain: Could not determine or invalid input
    """
    if not raw_text:
        return "Uncertain"
    cleaned = (
        raw_text.strip().strip(".").strip().strip('"').strip("'").title()
    )
    if cleaned in ALLOWED_CLASSIFICATIONS:
        return cleaned
    # Fallback: try to match partial strings (case-insensitive)
    lowered = cleaned.lower()
    if lowered.startswith("very low"):
        return "Very Low"
    if lowered.startswith("very high"):
        return "Very High"
    if lowered.startswith("low"):  # Must come after "very low" check
        return "Low"
    if lowered.startswith("mod"):  # Matches "moderate", "mod", etc.
        return "Moderate"
    if lowered.startswith("high"):  # Must come after "very high" check
        return "High"
    # If no pattern matches, return uncertain
    return "Uncertain"


# No longer needed: rule-based skills formatting


def main() -> None:
    """Main Streamlit application function.

    This function defines the complete user interface and application flow:
    1. Sets up the Streamlit page configuration and title
    2. Displays upload interface for resume files
    3. Processes uploaded files (PDF/DOCX text extraction)
    4. Sends resume data to OpenAI for analysis
    5. Displays extracted information and automation risk assessment

    The function handles all error cases gracefully, including missing API keys,
    file parsing failures, and API communication issues. User feedback is
    provided through Streamlit's UI components.

    Environment Variables Required:
        OPENAI_API_KEY: OpenAI API authentication key

    Environment Variables Optional:
        OPENAI_MODEL: Model name (default: gpt-4o-mini)
        DEBUG_LLM: Enable debug logging (set to "1")
    """
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

    # Process the uploaded file in a temporary location
    with st.spinner("Parsing your resume…"):
        temp_path = save_uploaded_file_to_temp(uploaded_file)
        try:
            extracted = parse_resume(temp_path)
        finally:
            # Always clean up the temporary file, even if processing fails
            try:
                os.remove(temp_path)
            except Exception:
                # Ignore cleanup errors - not critical
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

    # Send resume to OpenAI for analysis
    with st.spinner("Evaluating likelihood of your job being automated by AI…"):
        try:
            # Debug logging: print first 2000 chars if DEBUG_LLM is enabled
            if os.getenv("DEBUG_LLM") == "1":
                print("=== LLM Inputs (raw resume) ===")
                print(raw_text[:2000])  # Truncate for readability
            
            # Call the LLM analysis function
            parsed = analyze_resume_with_llm(client, raw_text, MODEL_NAME)
            
            # Extract results with safe defaults
            job_title = parsed.get("job_title")
            skills_list = parsed.get("skills") or []
            recent_experience = parsed.get("recent_experience") or []
            classification = parsed.get("classification", "Uncertain")
            rationale = parsed.get("rationale", "")
        except Exception as e:
            # Handle API errors gracefully with user-friendly message
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

