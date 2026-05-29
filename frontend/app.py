import json
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.pipeline import run_hr_pipeline


st.set_page_config(page_title="HR Resume Screening", page_icon="HR", layout="wide")

st.title("HR Resume Screening")

with st.sidebar:
    st.header("Setup")
    st.caption("Set GROQ_KEY or GROQ_API_KEY in your environment or a local .env file before running.")
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    if groq_key:
        os.environ["GROQ_KEY"] = groq_key
    st.code("streamlit run frontend/app.py", language="bash")

uploaded_pdf = st.file_uploader("Upload candidate CV as PDF", type=["pdf"])
jd_text = st.text_area("Job Description", height=260, placeholder="Paste the target job description here...")

run_clicked = st.button("Run Screening", type="primary", disabled=uploaded_pdf is None or not jd_text.strip())

if run_clicked and uploaded_pdf is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.getbuffer())
        tmp_path = Path(tmp_file.name)

    try:
        with st.spinner("Screening CV and preparing HR recommendation..."):
            result = run_hr_pipeline(tmp_path, jd_text)

        screening = result["cv_screening_results"]
        recommendation = result["hiring_recommendation_results"]
        interview_questions = result["interview_question_results"]

        score = screening["ats_compatibility_score"]
        st.metric("ATS Compatibility", f"{score}/100")

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Candidate Snapshot")
            st.json(
                {
                    "contact_info": screening["contact_info"],
                    "education": screening["education"],
                    "experience": screening["experience"],
                    "tech_stack_skills": screening["tech_stack_skills"],
                }
            )

            st.subheader("Project Highlights")
            for item in screening["project_highlights"]:
                st.markdown(f"**{item['project_name']}**")
                st.write(item["jd_requirement_matched"])
                st.caption(item["evidence_from_cv"])

        with right:
            st.subheader("Hiring Recommendation")
            st.info(recommendation["recommendation"])
            st.write(recommendation["executive_summary"])
            st.markdown("**HR Justification**")
            st.write(recommendation["hr_manager_justification"])

            st.markdown("**Strengths**")
            for strength in recommendation["strengths"]:
                st.write(f"- {strength}")

            st.markdown("**Weaknesses**")
            for weakness in recommendation["weaknesses"]:
                st.write(f"- {weakness}")

        st.subheader("Gap Analysis")
        st.dataframe(screening["gap_analysis"], use_container_width=True)

        st.subheader("Interview Questions")
        technical_tab, behavioral_tab, report_tab = st.tabs(["Technical", "Behavioral", "Report"])
        with technical_tab:
            for index, question in enumerate(interview_questions["technical_questions"], start=1):
                st.write(f"{index}. {question}")
        with behavioral_tab:
            for index, question in enumerate(interview_questions["behavioral_questions"], start=1):
                st.write(f"{index}. {question}")
        with report_tab:
            st.markdown(interview_questions["final_report"])

        email_tab, reject_tab, raw_tab = st.tabs(["Next Steps Email", "Rejection Email", "Raw JSON"])
        with email_tab:
            st.text_area("Acceptance / Next Steps", recommendation["acceptance_next_steps_email"], height=260)
        with reject_tab:
            st.text_area("Rejection", recommendation["rejection_email"], height=260)
        with raw_tab:
            st.code(json.dumps(result, indent=2, ensure_ascii=False), language="json")

    except Exception as exc:
        st.error(str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)
