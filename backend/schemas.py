from typing import Literal

from pydantic import BaseModel, Field


class ContactInfo(BaseModel):
    """Candidate contact details extracted from the CV."""

    name: str = Field(description="Candidate full name, or 'Unknown' if missing.")
    email: str | None = Field(default=None, description="Candidate email address.")
    phone: str | None = Field(default=None, description="Candidate phone number.")
    location: str | None = Field(default=None, description="Candidate location, if available.")
    links: list[str] = Field(default_factory=list, description="LinkedIn, GitHub, portfolio, or other relevant links.")


class EducationEntry(BaseModel):
    degree: str = Field(description="Degree or qualification name.")
    institution: str = Field(description="School, university, bootcamp, or institution name.")
    field_of_study: str | None = Field(default=None, description="Major or field of study.")
    graduation_date: str | None = Field(default=None, description="Graduation date or expected graduation date.")


class ExperienceAssessment(BaseModel):
    total_years: float = Field(ge=0, description="Estimated total professional years of experience.")
    relevant_years: float = Field(ge=0, description="Estimated years of experience relevant to this JD.")
    reasoning: str = Field(description="Brief explanation of how the experience estimate was calculated.")


class ProjectHighlight(BaseModel):
    project_name: str = Field(description="Project, product, or role highlight name.")
    jd_requirement_matched: str = Field(description="Specific JD requirement this highlight supports.")
    evidence_from_cv: str = Field(description="Concrete evidence from the CV.")
    technologies: list[str] = Field(default_factory=list, description="Relevant tools, frameworks, and languages.")


class GapItem(BaseModel):
    requirement: str = Field(description="Requirement from the JD.")
    gap: str = Field(description="What is missing or weak in the CV.")
    severity: Literal["Low", "Medium", "High"] = Field(description="Impact of this gap on hiring fit.")


class CVScreeningOutput(BaseModel):
    """Strict structured output produced by the CV Screening Specialist."""

    contact_info: ContactInfo = Field(description="Candidate contact information.")
    education: list[EducationEntry] = Field(description="Education history.")
    tech_stack_skills: list[str] = Field(description="Technical stack, tools, frameworks, and skills found in the CV.")
    experience: ExperienceAssessment = Field(description="Total and JD-relevant experience assessment.")
    project_highlights: list[ProjectHighlight] = Field(description="Project highlights mapped to JD requirements.")
    ats_compatibility_score: int = Field(ge=0, le=100, description="Objective ATS matching score from 0 to 100.")
    gap_analysis: list[GapItem] = Field(description="Missing technical or soft requirements compared to the JD.")


class HiringRecommendationOutput(BaseModel):
    """Strict structured output produced by the Hiring Recommendation Agent."""

    executive_summary: str = Field(description="Concise HR dashboard summary of candidate fit.")
    strengths: list[str] = Field(description="Candidate strengths relevant to the JD.")
    weaknesses: list[str] = Field(description="Candidate weaknesses and risks relevant to the JD.")
    recommendation: Literal["Hire", "Maybe", "Reject"] = Field(description="Final hiring recommendation.")
    hr_manager_justification: str = Field(description="Clear professional justification for the HR manager.")
    acceptance_next_steps_email: str = Field(description="Empathetic next-steps email for Hire or Maybe outcomes.")
    rejection_email: str = Field(description="Polite rejection email explaining key gaps for Reject outcomes.")


class TechnicalQuestion(BaseModel):
    question: str = Field(description="Technical interview question.")
    skill_measured: str = Field(description="Technical skill being assessed by this question.")


class TechnicalInterviewQuestions(BaseModel):
    """Technical question set produced by the Interview Generation Agent."""

    technical_questions: list[TechnicalQuestion] = Field(min_length=10,description=("Five technical questions, three project-based questions, and two advanced follow-up questions."),)


class BehavioralQuestion(BaseModel):
    question: str = Field( description="Behavioral or situational interview question.")
    soft_skill_measured: str = Field(description="Soft skill being evaluated by the question.")


class BehavioralInterviewQuestions(BaseModel):
    """Behavioral question set produced by the Interview Generation Agent."""

    behavioral_questions: list[BehavioralQuestion] = Field( min_length=5, description=("Five behavioral or situational interview questions with the associated soft skill being assessed."),)

class InterviewQuestionOutput(TechnicalInterviewQuestions, BehavioralInterviewQuestions):
    """Strict structured output produced by the Interview Generation Agent."""

    final_report: str = Field(description="Markdown interview question report ready for an HR or technical interviewer.")
