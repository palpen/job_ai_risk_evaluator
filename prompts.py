"""Prompt management system for AI job automation analysis.

This module provides versioned, testable, and maintainable prompt templates
for the resume analysis workflow.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class PromptVersion(Enum):
    """Available prompt versions."""
    V1_ORIGINAL = "v1.0"
    V2_ENHANCED = "v2.0"
    V3_EXPERIMENTAL = "v3.0"


@dataclass
class PromptTemplate:
    """Template for AI analysis prompts."""
    version: str
    system_prompt: str
    user_template: str
    examples: List[Dict[str, str]]
    temperature: float = 0.1  # Lower for consistent structured output
    
    def render(self, resume_text: str) -> Dict[str, str]:
        """Render the complete prompt with resume text."""
        return {
            "system": self.system_prompt,
            "user": self.user_template.format(resume_text=resume_text)
        }


class PromptManager:
    """Manages prompt templates with versioning and A/B testing."""
    
    def __init__(self):
        self._templates: Dict[str, PromptTemplate] = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load all available prompt templates."""
        # V1 - Original prompt (current production)
        self._templates[PromptVersion.V1_ORIGINAL.value] = self._create_v1_template()
        
        # V2 - Enhanced with better examples (future version)
        self._templates[PromptVersion.V2_ENHANCED.value] = self._create_v2_template()
    
    def get_template(self, version: str = PromptVersion.V1_ORIGINAL.value) -> PromptTemplate:
        """Get prompt template by version."""
        if version not in self._templates:
            raise ValueError(f"Unknown prompt version: {version}. Available: {list(self._templates.keys())}")
        return self._templates[version]
    
    def list_versions(self) -> List[str]:
        """List all available prompt versions."""
        return list(self._templates.keys())
    
    def _create_v1_template(self) -> PromptTemplate:
        """Create V1 prompt template (current production version)."""
        system_prompt = (
            "You are an expert career analyst system. Your purpose is to parse raw resume text, "
            "extract key professional information, and classify the role's automation risk based on a defined framework."
        )
        
        user_template = """--- Analysis Framework & Definitions ---

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

{resume_text}

Output:
Your response MUST be a single, compact JSON object and nothing else."""
        
        return PromptTemplate(
            version=PromptVersion.V1_ORIGINAL.value,
            system_prompt=system_prompt,
            user_template=user_template,
            examples=[],  # Examples embedded in template for V1
            temperature=0.1
        )
    
    def _create_v2_template(self) -> PromptTemplate:
        """Create V2 prompt template with improvements."""
        system_prompt = (
            "You are an expert AI automation analyst. Analyze resumes to assess job automation risk "
            "using current AI capabilities. Focus on task-level analysis rather than job titles. "
            "Be conservative in your assessments and consider human-AI collaboration scenarios."
        )
        
        # V2 could have a more concise template with external examples
        user_template = """Analyze this resume for AI automation risk using the 5-level scale:
Very High, High, Moderate, Low, Very Low

Consider:
- Task complexity and repeatability
- Need for human judgment and creativity  
- Physical vs digital work requirements
- Interpersonal and leadership responsibilities

Resume: {resume_text}

Respond with valid JSON:
{{
  "job_title": "extracted title",
  "skills": ["skill1", "skill2"],
  "recent_experience": ["bullet1", "bullet2"],
  "classification": "risk level",
  "rationale": "explanation"
}}"""
        
        return PromptTemplate(
            version=PromptVersion.V2_ENHANCED.value,
            system_prompt=system_prompt,
            user_template=user_template,
            examples=[],  # Could load from external file
            temperature=0.1
        )


# Global instance for easy access
prompt_manager = PromptManager()