import unittest

from backend.job_analysis_agent import analyze_job_description
from backend.schemas import JobAnalysisOutput


EXPECTED_KEYS = {
    "job_title",
    "experience_level",
    "years_of_experience",
    "must_have_skills",
    "nice_to_have_skills",
    "soft_skills",
    "responsibilities",
    "education_requirements",
    "certifications",
    "domain_or_industry",
    "location_requirements",
    "employment_type",
    "language_requirements",
    "summary",
}


class FakeJobAnalysisChain:
    def __init__(self, output: JobAnalysisOutput) -> None:
        self.output = output
        self.calls: list[dict[str, str]] = []

    def invoke(self, payload: dict[str, str]) -> JobAnalysisOutput:
        self.calls.append(payload)
        return self.output


class JobAnalysisAgentTests(unittest.TestCase):
    def test_normal_job_description_extracts_required_and_preferred_skills(self) -> None:
        job_description = """
        Senior Data Engineer
        We require 5+ years of experience building data pipelines with Python, SQL, and AWS.
        Preferred skills include Docker and Kubernetes. Strong communication and stakeholder
        management are required. Bachelor's degree required. AWS Certified Data Analytics is a plus.
        Finance domain experience, Cairo hybrid work, full-time role, and English fluency required.
        """
        expected = JobAnalysisOutput(
            job_title="Senior Data Engineer",
            experience_level="Senior",
            years_of_experience="5+ years",
            must_have_skills=["Python", "SQL", "AWS", "Data pipelines"],
            nice_to_have_skills=["Docker", "Kubernetes"],
            soft_skills=["Communication", "Stakeholder management"],
            responsibilities=["Build and maintain data pipelines"],
            education_requirements=["Bachelor's degree"],
            certifications=["AWS Certified Data Analytics"],
            domain_or_industry="Finance",
            location_requirements="Cairo hybrid",
            employment_type="Full-time",
            language_requirements=["English"],
            summary="Senior data engineering role focused on AWS data pipelines in finance.",
        )
        chain = FakeJobAnalysisChain(expected)

        result = analyze_job_description(job_description, chain=chain)

        self.assertEqual(set(result.keys()), EXPECTED_KEYS)
        self.assertEqual(result["experience_level"], "Senior")
        self.assertIn("Python", result["must_have_skills"])
        self.assertIn("Docker", result["nice_to_have_skills"])
        self.assertEqual(chain.calls, [{"job_description": job_description.strip()}])

    def test_job_description_with_no_clear_experience_level_returns_unknown(self) -> None:
        job_description = (
            "Join our operations team to document workflows, analyze support tickets, "
            "and coordinate improvements with product and customer success teams."
        )
        expected = JobAnalysisOutput(
            job_title="Operations Analyst",
            experience_level="Unknown",
            years_of_experience="Unknown",
            must_have_skills=["Workflow documentation", "Ticket analysis"],
            nice_to_have_skills=[],
            soft_skills=["Coordination", "Communication"],
            responsibilities=["Document workflows", "Analyze support tickets"],
            education_requirements=[],
            certifications=[],
            domain_or_industry="Customer support operations",
            location_requirements="Unknown",
            employment_type="Unknown",
            language_requirements=[],
            summary="Operations role with no explicit seniority or years of experience.",
        )
        chain = FakeJobAnalysisChain(expected)

        result = analyze_job_description(job_description, chain=chain)

        self.assertEqual(result["experience_level"], "Unknown")
        self.assertEqual(result["years_of_experience"], "Unknown")
        self.assertEqual(result["nice_to_have_skills"], [])

    def test_empty_job_description_fails_gracefully(self) -> None:
        expected = JobAnalysisOutput(
            job_title="Unknown",
            experience_level="Unknown",
            years_of_experience="Unknown",
            must_have_skills=[],
            nice_to_have_skills=[],
            soft_skills=[],
            responsibilities=[],
            education_requirements=[],
            certifications=[],
            domain_or_industry="Unknown",
            location_requirements="Unknown",
            employment_type="Unknown",
            language_requirements=[],
            summary="Unknown",
        )
        chain = FakeJobAnalysisChain(expected)

        with self.assertRaisesRegex(ValueError, "Job description text is required"):
            analyze_job_description("   ", chain=chain)

        self.assertEqual(chain.calls, [])


if __name__ == "__main__":
    unittest.main()
