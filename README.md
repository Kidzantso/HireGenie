# HR Resume Screening App

This project converts the old notebook-style Ollama pipeline into a small backend/frontend app that uses Groq via LangChain.

## Structure

- `backend/schemas.py` - Pydantic contracts for the screening, recommendation, and interview agents.
- `backend/pdf_utils.py` - PDF text extraction.
- `backend/pipeline.py` - sequential LangChain pipeline using Groq.
- `frontend/app.py` - Streamlit PDF upload UI.

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
