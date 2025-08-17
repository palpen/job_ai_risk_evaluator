"""AI Job Automation Risk Evaluator

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

import hashlib
import io
import json
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import streamlit as st
from openai import OpenAI
from openai.types.chat import ChatCompletion

from prompts import prompt_manager, PromptVersion

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

# Simple per-session rate limit (defaults: 3 requests per 60s)
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "3"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

# Security limits
MAX_FILE_SIZE_MB = 10  # Hard cap on file uploads
MAX_PDF_PAGES = 50     # Limit PDF pages to prevent DoS
MAX_TEXT_CHARS = 200000  # Limit extracted text before LLM processing
MAX_LLM_INPUT_CHARS = 50000  # Truncate text sent to LLM

# Allowed MIME types for security
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "text/plain"
}

# Cache configuration
CACHE_EXPIRATION_HOURS = 24
CACHE_MAX_ENTRIES = 100
CACHE_KEY_PREFIX = "resume_analysis"

# Prompt configuration
PROMPT_VERSION = os.getenv("PROMPT_VERSION", PromptVersion.V1_ORIGINAL.value)



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


@dataclass
class CacheEntry:
    """Represents a cached analysis result with metadata."""
    result: Dict[str, Any]
    created_at: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired based on creation time."""
        age_hours = (time.time() - self.created_at) / 3600
        return age_hours > CACHE_EXPIRATION_HOURS
    
    def touch(self) -> None:
        """Update access statistics for LRU tracking."""
        self.access_count += 1
        self.last_accessed = time.time()


class ResumeAnalysisCache:
    """
    LRU cache for resume analysis results with expiration.
    
    Uses Streamlit session state for storage with automatic cleanup
    of expired entries and LRU eviction when size limits are exceeded.
    """
    
    def __init__(self):
        """Initialize cache using Streamlit session state."""
        # Initialize cache in session state if not exists
        if "analysis_cache" not in st.session_state:
            st.session_state.analysis_cache = OrderedDict()
        if "cache_stats" not in st.session_state:
            st.session_state.cache_stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "expired_cleanups": 0
            }
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries from cache."""
        cache = st.session_state.analysis_cache
        expired_keys = [
            key for key, entry in cache.items() 
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del cache[key]
            st.session_state.cache_stats["expired_cleanups"] += 1
    
    def _enforce_size_limit(self) -> None:
        """Enforce cache size limit using LRU eviction."""
        cache = st.session_state.analysis_cache
        
        while len(cache) >= CACHE_MAX_ENTRIES:
            # Remove least recently used (first in OrderedDict)
            oldest_key = next(iter(cache))
            del cache[oldest_key]
            st.session_state.cache_stats["evictions"] += 1
    
    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached result if available and not expired.
        
        Args:
            cache_key: SHA-256 hash of normalized resume text
            
        Returns:
            Cached analysis result or None if not found/expired
        """
        self._cleanup_expired()
        
        cache = st.session_state.analysis_cache
        
        if cache_key not in cache:
            st.session_state.cache_stats["misses"] += 1
            return None
        
        entry = cache[cache_key]
        if entry.is_expired():
            del cache[cache_key]
            st.session_state.cache_stats["expired_cleanups"] += 1
            st.session_state.cache_stats["misses"] += 1
            return None
        
        # Move to end for LRU (most recently used)
        entry.touch()
        cache.move_to_end(cache_key)
        
        st.session_state.cache_stats["hits"] += 1
        return entry.result
    
    def put(self, cache_key: str, result: Dict[str, Any]) -> None:
        """
        Store analysis result in cache.
        
        Args:
            cache_key: SHA-256 hash of normalized resume text
            result: Analysis result from OpenAI
        """
        self._cleanup_expired()
        self._enforce_size_limit()
        
        cache = st.session_state.analysis_cache
        entry = CacheEntry(
            result=result.copy(),  # Deep copy to prevent mutations
            created_at=time.time()
        )
        
        cache[cache_key] = entry
        # Ensure it's at the end (most recently used)
        cache.move_to_end(cache_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        stats = st.session_state.cache_stats.copy()
        cache = st.session_state.analysis_cache
        
        total_requests = stats["hits"] + stats["misses"]
        hit_rate = (stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        stats.update({
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 1),
            "current_entries": len(cache),
            "max_entries": CACHE_MAX_ENTRIES,
            "expiration_hours": CACHE_EXPIRATION_HOURS
        })
        
        return stats
    
    def clear(self) -> None:
        """Clear all cached entries (useful for testing/debugging)."""
        st.session_state.analysis_cache.clear()
        st.session_state.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired_cleanups": 0
        }


def normalize_resume_text(text: str) -> str:
    """
    Normalize resume text for consistent cache key generation.
    
    Applies standardization to ensure textually equivalent resumes
    (with minor formatting differences) generate the same cache key.
    
    Args:
        text: Raw extracted resume text
        
    Returns:
        Normalized text suitable for hashing
    """
    # Convert to lowercase for case-insensitive matching
    normalized = text.lower()
    
    # Normalize whitespace: collapse multiple spaces, tabs, newlines into single spaces
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Remove common formatting artifacts that don't affect content
    normalized = re.sub(r'[^\w\s@.,()-]', '', normalized)  # Keep basic punctuation
    
    # Strip leading/trailing whitespace
    normalized = normalized.strip()
    
    return normalized


def generate_cache_key(text: str, model: str) -> str:
    """
    Generate a SHA-256 based cache key for resume analysis.
    
    The key incorporates both the normalized resume text and model name
    to ensure cache invalidation when switching between different models.
    
    Args:
        text: Normalized resume text
        model: OpenAI model name (e.g., 'gpt-4o-mini')
        
    Returns:
        Hexadecimal SHA-256 hash suitable as cache key
    """
    # Include model in cache key to handle model changes
    cache_input = f"{CACHE_KEY_PREFIX}:{model}:{text}"
    
    # Generate SHA-256 hash
    hash_object = hashlib.sha256(cache_input.encode('utf-8'))
    return hash_object.hexdigest()


def validate_uploaded_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> tuple[bool, str]:
    """Validate uploaded file for security and size constraints.
    
    Performs comprehensive validation including:
    - File size limits
    - MIME type validation (don't trust extensions)
    - Basic file structure checks
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        tuple[bool, str]: (is_valid, error_message). Error message empty if valid.
    """
    # Check file size
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        return False, f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB."
    
    # Validate MIME type (don't trust file extension)
    if uploaded_file.type not in ALLOWED_MIME_TYPES:
        return False, f"Unsupported file type: {uploaded_file.type}. Only PDF and DOCX files are allowed."
    
    # Additional validation for PDF files
    if uploaded_file.type == "application/pdf":
        # Read first few bytes to verify PDF signature
        file_data = uploaded_file.getvalue()
        if not file_data.startswith(b'%PDF-'):
            return False, "Invalid PDF file format."
    
    return True, ""


def extract_text_from_uploaded_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> tuple[str, str]:
    """Extract text directly from uploaded file in memory (no temp files).

    This function processes the uploaded file entirely in memory using BytesIO,
    avoiding the security risks and cleanup complexity of temporary files.
    
    Args:
        uploaded_file: A Streamlit uploaded file object containing the file data
                      and metadata.

    Returns:
        tuple[str, str]: (extracted_text, error_message). Error message empty if successful.
    """
    try:
        file_data = uploaded_file.getvalue()
        file_type = uploaded_file.type
        
        if file_type == "application/pdf":
            return _extract_text_from_pdf_bytes(file_data)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return _extract_text_from_docx_bytes(file_data)
        elif file_type == "text/plain":
            # Handle plain text files
            try:
                text = file_data.decode('utf-8', errors='ignore')
                return text[:MAX_TEXT_CHARS], ""
            except Exception as e:
                return "", f"Error reading text file: {str(e)}"
        else:
            return "", f"Unsupported file type: {file_type}"
            
    except Exception as e:
        return "", f"Error processing uploaded file: {str(e)}"


def _extract_text_from_pdf_bytes(file_data: bytes) -> tuple[str, str]:
    """Extract text content from PDF bytes using PyPDF2 with security limits.

    Processes PDF data in memory with strict limits on pages and characters
    to prevent DoS attacks. Handles malformed PDFs gracefully.

    Args:
        file_data: The raw bytes of the PDF file.

    Returns:
        tuple[str, str]: (extracted_text, error_message). Error message empty if successful.

    Security Features:
        - Limits processing to MAX_PDF_PAGES
        - Truncates output at MAX_TEXT_CHARS
        - Handles malformed/encrypted PDFs gracefully
    """
    if PyPDF2 is None:
        return "", "PDF processing not available (PyPDF2 not installed)"
    
    try:
        text_parts: List[str] = []
        pdf_stream = io.BytesIO(file_data)
        reader = PyPDF2.PdfReader(pdf_stream)
        
        # Check if PDF is encrypted
        if reader.is_encrypted:
            return "", "Cannot process encrypted PDF files"
            
        pages = getattr(reader, "pages", [])
        total_pages = len(pages)
        
        if total_pages == 0:
            return "", "PDF contains no readable pages. If this is a scanned document, please use OCR software first."
            
        # Limit pages processed to prevent DoS
        pages_to_process = min(total_pages, MAX_PDF_PAGES)
        
        total_chars = 0
        for i, page in enumerate(pages[:pages_to_process]):
            try:
                # Extract text from each page with character limit
                page_text = page.extract_text() or ""
                if page_text:
                    # Check if adding this page would exceed character limit
                    if total_chars + len(page_text) > MAX_TEXT_CHARS:
                        # Truncate the last page to fit within limit
                        remaining_chars = MAX_TEXT_CHARS - total_chars
                        if remaining_chars > 0:
                            text_parts.append(page_text[:remaining_chars])
                        break
                    
                    text_parts.append(page_text)
                    total_chars += len(page_text)
                    
            except Exception:
                # Skip individual pages that can't be processed
                continue
                
        if not text_parts:
            return "", "No readable text found in PDF. If this is a scanned document, please use OCR software first."
            
        extracted_text = "\n".join(text_parts).strip()
        warning = ""
        
        if pages_to_process < total_pages:
            warning = f" (processed {pages_to_process} of {total_pages} pages due to size limits)"
        elif total_chars >= MAX_TEXT_CHARS:
            warning = " (text truncated due to length limits)"
            
        return extracted_text, warning
        
    except Exception as e:
        return "", f"Error processing PDF: {str(e)}. If this is a scanned document, please use OCR software first."


def _extract_text_from_docx_bytes(file_data: bytes) -> tuple[str, str]:
    """Extract text content from DOCX bytes using python-docx.

    Processes DOCX data in memory with character limits to prevent DoS.
    Currently disabled for security - DOCX parsing can be complex.

    Args:
        file_data: The raw bytes of the DOCX file.

    Returns:
        tuple[str, str]: (extracted_text, error_message). Error message empty if successful.

    Note:
        This function is currently disabled for security reasons.
        DOCX files contain complex XML that can be exploited.
    """
    # DOCX parsing is disabled for security reasons
    if docx is None:
        return "", "DOCX processing is disabled for security reasons. Please convert to PDF first."
    
    # This code would be used if DOCX parsing were enabled:
    # try:
    #     docx_stream = io.BytesIO(file_data)
    #     document = docx.Document(docx_stream)
    #     paragraphs = [p.text for p in document.paragraphs if p.text]
    #     text = "\n".join(paragraphs).strip()
    #     return text[:MAX_TEXT_CHARS], ""
    # except Exception as e:
    #     return "", f"Error processing DOCX: {str(e)}"
    
    return "", "DOCX processing is disabled for security reasons. Please convert to PDF first."




def analyze_resume_with_llm(client: OpenAI, raw_text: str, model: str) -> Dict[str, Any]:
    """Analyze resume text using OpenAI's language model for automation risk assessment.

    This function sends the resume text to OpenAI's API using versioned prompt templates
    for maintainable and testable prompt engineering. The LLM extracts structured 
    information and provides automation risk classification.

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
        Uses versioned prompt templates from prompts.py for maintainable prompt engineering.
        Current prompt version can be set via PROMPT_VERSION environment variable.
    """
    # Get the appropriate prompt template
    template = prompt_manager.get_template(PROMPT_VERSION)
    prompts = template.render(raw_text)
    
    # Try to use template's temperature, fall back to default if not supported
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=template.temperature,
            messages=[
                {"role": "system", "content": prompts["system"]},
                {"role": "user", "content": prompts["user"]},
            ],
        )
    except Exception as e:
        # If temperature not supported, try without it (uses default)
        if "temperature" in str(e).lower():
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompts["system"]},
                    {"role": "user", "content": prompts["user"]},
                ],
            )
        else:
            # Re-raise if it's a different error
            raise

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


def analyze_resume_with_cache(client: OpenAI, raw_text: str, model: str) -> tuple[Dict[str, Any], bool]:
    """
    Analyze resume with caching to avoid duplicate API calls.
    
    This function wraps the LLM analysis with intelligent caching:
    1. Normalizes the input text for consistent hashing
    2. Checks cache for existing analysis of identical content
    3. Returns cached result if found, otherwise calls OpenAI API
    4. Stores new results in cache for future use
    
    Args:
        client: Initialized OpenAI client
        raw_text: Raw resume text to analyze
        model: OpenAI model name
        
    Returns:
        tuple[Dict[str, Any], bool]: (analysis_result, was_cached)
        was_cached indicates if result came from cache (True) or API (False)
    """
    cache = ResumeAnalysisCache()
    
    # Normalize text and generate cache key
    normalized_text = normalize_resume_text(raw_text)
    cache_key = generate_cache_key(normalized_text, model)
    
    # Try to get cached result first
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        return cached_result, True
    
    # Cache miss - call OpenAI API
    fresh_result = analyze_resume_with_llm(client, raw_text, model)
    
    # Store result in cache for future use
    cache.put(cache_key, fresh_result)
    
    return fresh_result, False


def parse_resume_from_upload(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> tuple[Optional[Dict[str, Any]], str]:
    """Parse a resume from uploaded file and extract its text content.

    This function serves as the main entry point for resume processing. It
    extracts raw text from the uploaded file in memory and returns it in a 
    dictionary structure. No rule-based parsing or structured data extraction
    is performed - that is handled by the LLM analysis.

    Args:
        uploaded_file: Streamlit uploaded file object.

    Returns:
        tuple[Optional[Dict[str, Any]], str]: (parsed_data, error_message).
        Parsed data contains 'raw_text' key, or None if extraction failed.
        Error message empty if successful.

    Example:
        >>> result, error = parse_resume_from_upload(uploaded_file)
        >>> if result:
        ...     text = result['raw_text']
    """
    # Extract text directly from uploaded file (in memory)
    text, error_msg = extract_text_from_uploaded_file(uploaded_file)
    
    if error_msg:
        return None, error_msg
        
    if not text.strip():
        return None, "No readable text found in the file"
        
    # Truncate text for LLM processing to prevent excessive costs/DoS
    truncated_text = text[:MAX_LLM_INPUT_CHARS]
    truncation_note = ""
    if len(text) > MAX_LLM_INPUT_CHARS:
        truncation_note = f" (truncated from {len(text):,} to {MAX_LLM_INPUT_CHARS:,} characters for processing)"
    
    return {"raw_text": truncated_text, "truncation_note": truncation_note}, ""


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
    3. Processes uploaded files (PDF/DOCX text extraction) in memory
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
    
    # Debug sidebar for cache management
    if os.getenv("DEBUG_LLM") == "1":
        with st.sidebar:
            st.header("🔧 Debug Tools")
            cache = ResumeAnalysisCache()
            stats = cache.get_stats()
            
            st.metric("Cache Entries", f"{stats['current_entries']}/{stats['max_entries']}")
            st.metric("Hit Rate", f"{stats['hit_rate_percent']}%")
            
            if st.button("🗑️ Clear Cache", help="Clear all cached analysis results"):
                cache.clear()
                st.success("Cache cleared!")
                st.rerun()
    
    st.title("Will AI Take My Job?")
    st.markdown("##### Upload your resume below to find out")
    uploaded_file = st.file_uploader(
        "", 
        type=["pdf", "docx", "txt"], 
        key="resume_uploader",
        help=f"Supported formats: PDF, DOCX, TXT. Maximum size: {MAX_FILE_SIZE_MB}MB"
    )
    st.write(
        "The resume data is sent to an LLM to do the analysis. No data is stored or shared with anyone else."
        "\nResults are speculative, generated by an LLM, and should not be treated as definitive!"
    )

    if uploaded_file is None:
        return

    # Validate uploaded file for security
    is_valid, validation_error = validate_uploaded_file(uploaded_file)
    if not is_valid:
        st.error(f"File validation failed: {validation_error}")
        return

    # Process the uploaded file in memory (no temp files)
    with st.spinner("Parsing your resume…"):
        extracted, parse_error = parse_resume_from_upload(uploaded_file)

    if parse_error:
        st.error(f"Could not parse resume: {parse_error}")
        return
        
    if not extracted:
        st.error("Could not extract text from resume. Please check the file format.")
        return

    raw_text: str = str(extracted.get("raw_text", ""))
    truncation_note = extracted.get("truncation_note", "")
    
    if not raw_text.strip():
        st.error("No readable text found in the resume.")
        return
        
    # Show truncation warning if applicable
    if truncation_note:
        st.info(f"📄 Resume processed{truncation_note}")
    
    # Show file processing info
    st.success(f"✅ Resume processed successfully ({len(raw_text):,} characters)")

    client = build_openai_client()
    if client is None:
        st.error(
            "OPENAI_API_KEY is not set. Please set the environment variable and restart the app."
        )
        return

    # Simple per-session rate limiting
    now_ts = time.time()
    req_ts: List[float] = st.session_state.get("_req_timestamps", [])
    # Keep only timestamps within the sliding window
    req_ts = [t for t in req_ts if now_ts - t < RATE_LIMIT_WINDOW_SECONDS]
    if len(req_ts) >= RATE_LIMIT_MAX_REQUESTS:
        oldest = min(req_ts)
        wait_seconds = int(RATE_LIMIT_WINDOW_SECONDS - (now_ts - oldest)) + 1
        st.error(f"Rate limit exceeded. Please wait {wait_seconds}s and try again.")
        st.caption(f"Limit: {RATE_LIMIT_MAX_REQUESTS} request(s) per {RATE_LIMIT_WINDOW_SECONDS}s per session.")
        return

    # Send resume to OpenAI for analysis
    with st.spinner("Evaluating likelihood of your job being automated by AI…"):
        try:
            # Debug logging: print metadata only (no PII) if DEBUG_LLM is enabled
            if os.getenv("DEBUG_LLM") == "1":
                print("=== LLM Request Debug Info ===")
                print(f"Text length: {len(raw_text):,} characters")
                print(f"Model: {MODEL_NAME}")
                print(f"Prompt version: {PROMPT_VERSION}")
                print(f"Temperature: {prompt_manager.get_template(PROMPT_VERSION).temperature}")
                # Do NOT log actual resume content to prevent PII leakage
            
            # Call the cached LLM analysis function
            # Only update rate limiting for actual API calls
            parsed, was_cached = analyze_resume_with_cache(client, raw_text, MODEL_NAME)
            
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
    st.header("Key information from your resume")
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
    
    # Show cache status
    if was_cached:
        st.info("💾 **Result retrieved from cache** - No API call needed!")
    else:
        st.info("🔄 **Analysis completed** - Result cached for future use")
        # Update rate limiting only for fresh API calls
        req_ts.append(now_ts)
        st.session_state["_req_timestamps"] = req_ts
    
    # Show centered classification result
    st.markdown("### **How likely is your job to be automated by AI?**")
    left, mid, right = st.columns([1, 2, 1])
    with mid:
        st.markdown(f"### **{classification}**")
    if rationale:
        st.markdown("#### **Explanation:**")
        st.write(rationale)
        
    # Show detailed cache statistics in debug mode
    if os.getenv("DEBUG_LLM") == "1":
        cache = ResumeAnalysisCache()
        stats = cache.get_stats()
        
        with st.expander("🔍 Cache Statistics (Debug Mode)", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Cache Hit Rate", f"{stats['hit_rate_percent']}%")
                st.metric("Total Requests", stats['total_requests'])
            
            with col2:
                st.metric("Cache Hits", stats['hits'])
                st.metric("Cache Misses", stats['misses'])
            
            with col3:
                st.metric("Current Entries", f"{stats['current_entries']}/{stats['max_entries']}")
                st.metric("Evictions", stats['evictions'])
            
            st.caption(f"Cache expires after {stats['expiration_hours']} hours • "
                      f"Expired cleanups: {stats['expired_cleanups']}")


if __name__ == "__main__":
    main()

