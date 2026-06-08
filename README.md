# HR Resume Screening App

This project converts the old notebook-style Ollama pipeline into a small backend/frontend app that uses Groq via LangChain.

## Structure

- `backend/schemas.py` - Pydantic contracts for the screening, recommendation, and interview agents.
- `backend/job_analysis_agent.py` - Agent #2 for structured job-description requirement extraction.
- `backend/pdf_utils.py` - PDF text extraction.
- `backend/pipeline.py` - sequential LangChain pipeline using Groq.
- `frontend/app.py` - Streamlit PDF upload UI.
- `tests/test_job_analysis_agent.py` - Basic tests for Agent #2.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set:

```bash
GROQ_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

You can also paste the Groq key into the Streamlit sidebar instead of saving a `.env` file.

## Run

```bash
streamlit run frontend/app.py
```

Upload a PDF resume, paste a job description, and run the screening flow. The app returns CV screening results,
a hiring recommendation, candidate emails, and tailored technical/behavioral interview questions.

## Run Agent #2 directly

```python
from backend.job_analysis_agent import analyze_job_description_json

job_description = "Senior Python Engineer with 5+ years of experience. Must have Python and AWS. Docker is a plus."
print(analyze_job_description_json(job_description))
```

The output follows this structure:

```json
{
  "job_title": "",
  "experience_level": "",
  "years_of_experience": "",
  "must_have_skills": [],
  "nice_to_have_skills": [],
  "soft_skills": [],
  "responsibilities": [],
  "education_requirements": [],
  "certifications": [],
  "domain_or_industry": "",
  "location_requirements": "",
  "employment_type": "",
  "language_requirements": [],
  "summary": ""
}
```

## Test

```bash
python -m unittest discover -s tests
```
