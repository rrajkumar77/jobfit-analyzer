"""
Simple JobFit Analyzer — UI layer only.

This file should contain Streamlit calls and almost nothing else. Prompts
live in prompts.py, LLM plumbing in llm_client.py, the matching algorithm
in matching.py, masking in masking.py, config in config.py.

Scope note: this covers the core flow (upload JD -> upload resume -> mask
-> extract skills -> validate -> show report -> download). The original
app also had interview-question generation (situational/behavioral/coding)
and a template save/load sidebar. Those are independent features and
should live in their own modules (e.g. interview_questions.py) following
the same pattern as matching.py, rather than growing this file back into
a 1,900-line monolith. Porting them is a follow-up, not included here.
"""

from __future__ import annotations

import fitz  # PyMuPDF
import docx
import streamlit as st

from config import Settings, get_logger
from llm_client import JobFitLLMClient
from masking import SecurityMasker, create_masking_audit_log
from matching import JobFitValidator

logger = get_logger(__name__)

st.set_page_config(page_title="Simple JobFit Analyzer", page_icon="🎯", layout="wide")


# ── Cached singletons ─────────────────────────────────────────────────────

@st.cache_resource
def get_settings() -> Settings:
    return Settings.load()


@st.cache_resource
def get_validator() -> JobFitValidator:
    settings = get_settings()
    return JobFitValidator(settings, JobFitLLMClient(settings))


def get_masker() -> SecurityMasker:
    if "masker" not in st.session_state:
        st.session_state.masker = SecurityMasker()
    return st.session_state.masker


# ── File extraction ───────────────────────────────────────────────────────

def extract_text_from_file(file) -> str:
    try:
        if file.type == "application/pdf":
            doc = fitz.open(stream=file.read(), filetype="pdf")
            return "".join(page.get_text() for page in doc)
        if file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return "\n".join(p.text for p in docx.Document(file).paragraphs)
        if file.type == "text/plain":
            return file.read().decode("utf-8")
        st.error(f"Unsupported file type: {file.type}")
        return ""
    except Exception as e:  # noqa: BLE001
        logger.exception("Failed to extract text from uploaded file")
        st.error(f"Couldn't read that file: {e}")
        return ""


# ── Session state ─────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "jd_text": None, "resume_text": None,
        "candidate_label": None,  # masked label — safe to display/export
        "top_5_skills": None, "fit_score": None, "validations": None,
        "masking_audit_log": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()

try:
    settings = get_settings()
except RuntimeError as e:
    st.error(str(e))
    st.stop()


# ── UI ───────────────────────────────────────────────────────────────────

st.title("🎯 Simple JobFit Analyzer")
st.caption("Upload a JD and a resume — get a skills-based fit assessment. 🔒 PII protected.")

col_jd, col_resume = st.columns(2)

with col_jd:
    st.subheader("📄 Job Description")
    jd_file = st.file_uploader("Upload JD", type=["pdf", "docx", "txt"], key="jd_upload")
    if jd_file:
        raw = extract_text_from_file(jd_file)
        if raw:
            mask_result = get_masker().mask_jd(raw)
            st.session_state.jd_text = mask_result.masked_text
            st.session_state.masking_audit_log.append(
                create_masking_audit_log(mask_result, "jd")
            )
            st.success(f"✅ JD processed ({mask_result.mask_count} item(s) masked)")

with col_resume:
    st.subheader("📋 Resume")
    resume_file = st.file_uploader("Upload resume", type=["pdf", "docx", "txt"], key="resume_upload")
    if resume_file:
        raw = extract_text_from_file(resume_file)
        if raw:
            mask_result = get_masker().mask_resume(raw)
            st.session_state.resume_text = mask_result.masked_text
            # IMPORTANT: candidate_label (masked), not mask_result.candidate_name,
            # is what gets stored in session state and used everywhere downstream.
            st.session_state.candidate_label = mask_result.candidate_label
            st.session_state.masking_audit_log.append(
                create_masking_audit_log(mask_result, "resume")
            )
            st.success(f"✅ Resume processed as {mask_result.candidate_label} "
                       f"({mask_result.mask_count} item(s) masked)")

st.divider()

if st.session_state.jd_text and st.session_state.resume_text:
    if st.button("🔍 Analyze fit", type="primary"):
        validator = get_validator()
        with st.spinner("Extracting required skills from JD..."):
            st.session_state.top_5_skills = validator.extract_top_5_skills(st.session_state.jd_text)

        with st.spinner(f"Validating {st.session_state.candidate_label}..."):
            fit_score, validations = validator.validate_candidate(
                st.session_state.top_5_skills, st.session_state.resume_text,
            )
            st.session_state.fit_score = fit_score
            st.session_state.validations = validations

if st.session_state.validations:
    st.subheader(f"📊 {st.session_state.candidate_label} — Fit: {st.session_state.fit_score:.0f}/100")

    for v in st.session_state.validations:
        icon = "✅" if v.has_project_experience else "❌"
        with st.expander(f"{icon} {v.skill_name} — {v.validation_score:.0f}%"):
            st.write(f"**Evidence**: {v.evidence_summary}")
            st.write(f"**Example**: {v.project_example}")

    report_md = JobFitValidator.generate_report(
        st.session_state.candidate_label, st.session_state.fit_score, st.session_state.validations,
    )
    st.download_button(
        "📥 Download report",
        data=report_md,
        # Filename uses the masked label, never the real name.
        file_name=f"fit_report_{st.session_state.candidate_label}.md",
        mime="text/markdown",
    )
elif not (st.session_state.jd_text and st.session_state.resume_text):
    st.info("📤 Upload a JD and a resume to get started")

# ── Audit log (counts and types only — never sensitive content) ──────────
if st.session_state.masking_audit_log:
    st.divider()
    with st.expander("🔒 Security audit log"):
        import pandas as pd
        st.dataframe(pd.DataFrame(st.session_state.masking_audit_log), use_container_width=True)
