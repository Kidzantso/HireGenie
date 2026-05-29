import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from backend.pdf_utils import extract_pdf_text
from backend.schemas import (
    BehavioralInterviewQuestions,
    CVScreeningOutput,
    HiringRecommendationOutput,
    InterviewQuestionOutput,
    TechnicalInterviewQuestions,
)


load_dotenv()


def _model_name() -> str:
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def _groq_api_key() -> str | None:
    return os.getenv("GROQ_KEY") or os.getenv("GROQ_API_KEY")


def build_llm() -> ChatGroq:
    """Create the Groq chat model used by both agents."""
    groq_api_key = _groq_api_key()
    if not groq_api_key:
        raise RuntimeError("Missing GROQ_KEY or GROQ_API_KEY. Add it to your environment or a local .env file.")

    return ChatGroq(
        model=_model_name(),
        groq_api_key=groq_api_key,
        temperature=0.2,
        max_retries=2,
        timeout=90,
    )


CV_SCREENING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an elite Technical Recruiter and Applicant Tracking System optimization expert. "
                "Analyze the candidate CV against the job description and return only structured data that "
                "matches the provided schema. Use evidence from the CV. If a value is missing, use null, "
                "an empty list, or 'Unknown' as appropriate. Do not add conversational preambles."
            ),
        ),
        (
            "human",
            (
                "Job Description:\n{job_description}\n\n"
                "Candidate CV Text:\n{cv_text}\n\n"
                "Responsibilities:\n"
                "1. Extract contact info, education, and tech stack/skills.\n"
                "2. Calculate total and relevant years of experience.\n"
                "3. Extract project highlights mapped to JD requirements.\n"
                "4. Calculate an ATS compatibility score from 0 to 100.\n"
                "5. List missing technical or soft requirements from the JD."
            ),
        ),
    ]
)


HIRING_RECOMMENDATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a seasoned Chief Human Resources Officer. Use the screening data and the original "
                "job description to make a definitive recommendation. Return only structured data matching "
                "the provided schema. Draft both candidate email variants even though only one may be sent."
            ),
        ),
        (
            "human",
            (
                "Job Description:\n{job_description}\n\n"
                "Structured CV Screening Data:\n{cv_screening_output}\n\n"
                "Make a final recommendation of exactly Hire, Maybe, or Reject. Provide an executive summary, "
                "strengths, weaknesses, HR justification, an acceptance/next-steps email, and a rejection email."
            ),
        ),
    ]
)


TECHNICAL_INTERVIEW_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a senior technical interviewer. Generate realistic, industry-level interview "
                "questions that adapt to the candidate's experience and the target role. Return only "
                "structured data matching the provided schema."
            ),
        ),
        (
            "human",
            (
                "Job Description:\n{job_description}\n\n"
                "Candidate Summary:\n{candidate_summary}\n\n"
                "Candidate Projects:\n{candidate_projects}\n\n"
                "Generate exactly 10 technical questions:\n"
                "- 5 technical questions related to the required job skills.\n"
                "- 3 project-based questions related to the candidate projects.\n"
                "- 2 deep follow-up questions that test real understanding.\n\n"
                "Focus on problem solving, implementation details, architecture, challenges, technologies used, "
                "model choices, deployment, optimization, debugging, and the candidate's real contribution. "
                "Avoid generic questions."
            ),
        ),
    ]
)


BEHAVIORAL_INTERVIEW_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an HR interviewer. Generate behavioral questions tailored to the role and the "
                "candidate summary. Return only structured data matching the provided schema."
            ),
        ),
        (
            "human",
            (
                "Job Description:\n{job_description}\n\n"
                "Candidate Summary:\n{candidate_summary}\n\n"
                "Generate exactly 5 behavioral interview questions focused on teamwork, leadership, "
                "communication, handling pressure, and conflict resolution."
            ),
        ),
    ]
)


def build_chains() -> tuple[Any, Any, Any, Any]:
    """Build the sequential structured-output chains."""
    llm = build_llm()
    cv_screening_chain = CV_SCREENING_PROMPT | llm.with_structured_output(CVScreeningOutput)
    hiring_chain = HIRING_RECOMMENDATION_PROMPT | llm.with_structured_output(HiringRecommendationOutput)
    technical_interview_chain = TECHNICAL_INTERVIEW_PROMPT | llm.with_structured_output(TechnicalInterviewQuestions)
    behavioral_interview_chain = BEHAVIORAL_INTERVIEW_PROMPT | llm.with_structured_output(BehavioralInterviewQuestions)
    return cv_screening_chain, hiring_chain, technical_interview_chain, behavioral_interview_chain


def _build_candidate_summary(cv_screening: CVScreeningOutput, hiring_recommendation: HiringRecommendationOutput) -> str:
    contact = cv_screening.contact_info
    experience = cv_screening.experience
    skills = ", ".join(cv_screening.tech_stack_skills) or "Unknown"
    strengths = "; ".join(hiring_recommendation.strengths) or "Unknown"
    weaknesses = "; ".join(hiring_recommendation.weaknesses) or "Unknown"
    gaps = "; ".join(f"{item.requirement}: {item.gap}" for item in cv_screening.gap_analysis) or "No major gaps found."

    return (
        f"Candidate: {contact.name}\n"
        f"Experience: {experience.total_years} total years, {experience.relevant_years} relevant years. "
        f"{experience.reasoning}\n"
        f"Skills: {skills}\n"
        f"ATS compatibility score: {cv_screening.ats_compatibility_score}/100\n"
        f"Strengths: {strengths}\n"
        f"Weaknesses: {weaknesses}\n"
        f"Gaps: {gaps}"
    )


def _build_candidate_projects(cv_screening: CVScreeningOutput) -> str:
    if not cv_screening.project_highlights:
        return "No project highlights were extracted from the CV."

    project_lines = []
    for index, project in enumerate(cv_screening.project_highlights, start=1):
        technologies = ", ".join(project.technologies) or "Unknown technologies"
        project_lines.append(
            f"{index}. {project.project_name}\n"
            f"   JD match: {project.jd_requirement_matched}\n"
            f"   Evidence: {project.evidence_from_cv}\n"
            f"   Technologies: {technologies}"
        )
    return "\n".join(project_lines)


def _compile_interview_report(technical_questions: list[str], behavioral_questions: list[str]) -> str:
    technical = "\n".join(f"{index}. {question}" for index, question in enumerate(technical_questions, start=1))
    behavioral = "\n".join(f"{index}. {question}" for index, question in enumerate(behavioral_questions, start=1))

    return (
        "# AI Interview Question Report\n\n"
        "## Technical Questions\n\n"
        f"{technical}\n\n"
        "## Behavioral Questions\n\n"
        f"{behavioral}"
    )


def run_hr_pipeline(pdf_path: str | Path, jd_text: str) -> dict[str, Any]:
    """Parse a CV PDF, run both Grok-backed agents, and return an HR review payload."""
    if not jd_text or not jd_text.strip():
        raise ValueError("Job description text is required.")

    cv_text = extract_pdf_text(pdf_path)
    cv_screening_chain, hiring_chain, technical_interview_chain, behavioral_interview_chain = build_chains()

    cv_screening: CVScreeningOutput = cv_screening_chain.invoke(
        {
            "job_description": jd_text.strip(),
            "cv_text": cv_text,
        }
    )

    hiring_recommendation: HiringRecommendationOutput = hiring_chain.invoke(
        {
            "job_description": jd_text.strip(),
            "cv_screening_output": cv_screening.model_dump_json(indent=2),
        }
    )

    candidate_summary = _build_candidate_summary(cv_screening, hiring_recommendation)
    candidate_projects = _build_candidate_projects(cv_screening)
    technical_result: TechnicalInterviewQuestions = technical_interview_chain.invoke(
        {
            "job_description": jd_text.strip(),
            "candidate_summary": candidate_summary,
            "candidate_projects": candidate_projects,
        }
    )
    behavioral_result: BehavioralInterviewQuestions = behavioral_interview_chain.invoke(
        {
            "job_description": jd_text.strip(),
            "candidate_summary": candidate_summary,
        }
    )
    technical_questions = technical_result.technical_questions
    behavioral_questions = behavioral_result.behavioral_questions
    interview_questions = InterviewQuestionOutput(
        technical_questions=technical_questions,
        behavioral_questions=behavioral_questions,
        final_report=_compile_interview_report(technical_questions, behavioral_questions),
    )

    return {
        "model": _model_name(),
        "job_description": jd_text.strip(),
        "cv_text_preview": cv_text[:1500],
        "cv_screening_results": cv_screening.model_dump(),
        "hiring_recommendation_results": hiring_recommendation.model_dump(),
        "interview_question_results": interview_questions.model_dump(),
    }


def run_hr_pipeline_json(pdf_path: str | Path, jd_text: str) -> str:
    """Convenience wrapper for notebook-style use."""
    return json.dumps(run_hr_pipeline(pdf_path, jd_text), indent=2, ensure_ascii=False)
