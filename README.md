# Simple JobFit Analyzer

JD-vs-resume skills matching tool. Uploads a job description and a resume,
extracts the top 5 required competencies, and scores the candidate against
each one using an LLM with semantic bridging (credits transferable
experience, not just exact keyword matches).

## What changed in this refactor

This replaces the single 1,943-line `app_simple_top5_FINAL.py` with a
modular structure:

| File | Responsibility |
|---|---|
| `config.py` | Env var loading + validation, in one place |
| `llm_client.py` | Groq wrapper: timeout, retry-with-backoff, robust JSON parsing |
| `masking.py` | PII/client-info masking — **now includes candidate name masking** |
| `prompts.py` | All prompt templates |
| `matching.py` | The actual skill-extraction and validation algorithm |
| `app.py` | Streamlit UI only |
| `tests/test_masking.py` | Regression tests for the name-leak fix |

### The security fix

The original `security_masker.py` masked email, phone, SSN, address, DOB,
and zip — but never the candidate's name. `candidate_name` was pulled from
the raw resume text and used unmasked in LLM prompts, on-screen labels, and
exported filenames. That's how a real candidate's name ended up in a
downloadable `.docx` file that got committed to a public repo.

`masking.py` now extracts the candidate's name and returns a stable,
non-reversible `candidate_label` (e.g. `Candidate-4f2a`) for everything
that leaves the local session — prompts, filenames, audit logs. The real
name is only ever held in memory for on-screen display, which is where a
recruiter legitimately needs it.

Run `pytest tests/ -v` — `test_real_name_never_appears_in_masked_text` is
the regression test that would have caught the original bug.

### Scope note

This covers the core matching flow. The original app also had interview
question generation (situational/behavioral/coding) and a template
save/load sidebar — those are independent features and belong in their own
modules (e.g. `interview_questions.py`) following the same pattern as
`matching.py`, rather than being folded back into `app.py`. Not ported here.

## Setup

```bash
cp .env.example .env
# edit .env and add your GROQ_API_KEY

pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
pip install pytest
pytest tests/ -v
```
