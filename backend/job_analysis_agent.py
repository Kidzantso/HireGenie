import json
from typing import Any

from backend.schemas import JobAnalysisOutput


JOB_ANALYSIS_PROMPT_MESSAGES = [
    (
        "system",
        (
            "You are Agent #2: Job Analysis Agent for an HR resume screening pipeline. "
            "Analyze job descriptions and extract clean hiring requirements. Return only structured data "
            "matching the schema. If a field is not present, use 'Unknown' for strings and an empty list for lists. "
            "Do not invent requirements that are not supported by the job description."
        ),
    ),
    (
        "human",
        (
            "Job Description:\n{job_description}\n\n"
            "Extract:\n"
            "1. Job title.\n"
            "2. Experience level as exactly one of: Intern, Junior, Mid, Senior, Lead, Unknown.\n"
            "3. Years of experience requirement.\n"
            "4. Required technical skills and hard business skills as must_have_skills.\n"
            "5. Preferred, optional, bonus, or plus skills as nice_to_have_skills.\n"
            "6. Soft and business collaboration skills as soft_skills.\n"
            "7. Responsibilities.\n"
            "8. Education, certification, industry/domain, location, remote/on-site, employment type, and language requirements.\n\n"
            "Keep must_have_skills separate from nice_to_have_skills. Keep soft_skills separate from technical skills. "
            "Return concise, normalized values that can be passed to a CV Screening Agent."
        ),
    ),
]


def _normalize_job_description(job_description: str | None) -> str:
    if not job_description or not job_description.strip():
        raise ValueError("Job description text is required.")
    return job_description.strip()


def _as_job_analysis_output(output: Any) -> JobAnalysisOutput:
    if isinstance(output, JobAnalysisOutput):
        return output
    if isinstance(output, dict):
        return JobAnalysisOutput.model_validate(output)
    if hasattr(output, "model_dump"):
        return JobAnalysisOutput.model_validate(output.model_dump())
    raise TypeError("Job analysis chain returned an unsupported output type.")


def build_job_analysis_chain(llm: Any | None = None) -> Any:
    """Build the structured-output chain for Agent #2."""
    from langchain_core.prompts import ChatPromptTemplate

    if llm is None:
        from backend.pipeline import build_llm

        llm = build_llm()

    prompt = ChatPromptTemplate.from_messages(JOB_ANALYSIS_PROMPT_MESSAGES)
    return prompt | llm.with_structured_output(JobAnalysisOutput)


class JobAnalysisAgent:
    """Callable wrapper for the Job Analysis Agent."""

    def __init__(self, chain: Any | None = None) -> None:
        self.chain = chain

    def analyze(self, job_description: str | None) -> dict[str, Any]:
        cleaned_description = _normalize_job_description(job_description)
        if self.chain is None:
            self.chain = build_job_analysis_chain()
        output = self.chain.invoke({"job_description": cleaned_description})
        return _as_job_analysis_output(output).model_dump()


def analyze_job_description(job_description: str | None, chain: Any | None = None) -> dict[str, Any]:
    """Analyze a job description and return structured hiring requirements."""
    return JobAnalysisAgent(chain=chain).analyze(job_description)


def analyze_job_description_json(job_description: str | None, chain: Any | None = None) -> str:
    """Analyze a job description and return the result as formatted JSON."""
    return json.dumps(analyze_job_description(job_description, chain=chain), indent=2, ensure_ascii=False)
