"""
Prompt templates.

Previously these were inline f-strings scattered across
app_simple_top5_FINAL.py, mixed in with Streamlit UI code. Pulling them
out here means: (1) they can be unit-tested/reviewed without touching UI
code, (2) prompt-engineering changes don't risk breaking the app, and
(3) it's now obvious at a glance whether a prompt ever mentions the
candidate's real name (none of them should — pass `candidate_label`,
never `candidate_name`, from masking.MaskingResult).
"""

EXTRACT_TOP_5_SKILLS = """Read this job description and extract the TOP 5 most critical skills/competencies needed.

JOB DESCRIPTION:
{jd_text}

IMPORTANT RULES:
1. Frame each skill as a DEMONSTRABLE COMPETENCY — what a candidate must be able to show evidence of doing.
2. Use broad enough language to capture transferable experience (e.g. "Stakeholder management and influencing senior leaders" not "AI executive sponsorship")
3. Each skill should be 5-15 words and describe the CAPABILITY, not just the domain.
4. Good examples:
   - "Driving adoption of new platforms or processes across large organisations"
   - "Stakeholder management and influencing at senior/executive level"
   - "Designing training, workshops or enablement programs for teams"
   - "Measuring outcomes and impact of initiatives with data"
5. Bad examples (too narrow/literal):
   - "AI adoption strategy" (too specific — misses transferable change management experience)
   - "GenAI" (just a tool name)
   - "Executive AI sponsorship" (too domain-locked)

Return as JSON array with EXACTLY 5 skills:
["Competency 1", "Competency 2", "Competency 3", "Competency 4", "Competency 5"]

Just the array, nothing else:"""


VALIDATE_CANDIDATE_BATCH = """You are an expert recruiter assessing a candidate's resume against required skills.

RESUME:
{resume_text}

SKILLS TO EVALUATE:
{skills_numbered}

CRITICAL INSTRUCTIONS:
1. Look for DIRECT experience AND transferable/adjacent experience.
   e.g. "AI adoption strategy" can be evidenced by "driving adoption of new compliance frameworks"
   e.g. "Stakeholder buy-in" can be evidenced by "partnered with cross-functional teams on X"
   e.g. "Change management" can be evidenced by "redesigned protocols reducing incidents by 15%"
2. Do NOT require the exact skill words to appear in the resume. Judge the SUBSTANCE of experience.
3. Give credit for measurable outcomes even if in a different domain.
4. A score of 0 should only be given when there is genuinely ZERO related experience.

Return a JSON array with one entry per skill (same order as the list):
[
  {{
    "skill": "exact skill name from list above",
    "has_project_experience": true,
    "score": 85,
    "evidence": "specific evidence from resume — quote actual phrases",
    "example": "one concrete project example, or 'No evidence found'"
  }}
]"""


MERGED_JD_ANALYSIS = """You are an expert recruiter. Analyse this job description and return a single JSON object.

JOB DESCRIPTION:
{jd_text}

Return ONLY valid JSON (no markdown):
{{
  "role_summary": "2-3 sentence overview of the role",
  "role_combination": "RoleA + RoleB (e.g. Product Manager + Data Analyst)",
  "experience_level": "X-Y years",
  "ideal_candidate": "1-sentence description of the ideal hire",
  "key_requirements": ["req1", "req2", "req3", "req4", "req5"],
  "top_5_skills": ["Competency 1", "Competency 2", "Competency 3", "Competency 4", "Competency 5"],
  "tech_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
}}"""


# NOTE: this template takes `candidate_label` (e.g. "Candidate-4f2a"),
# never the real candidate name — see masking.py for why.
HM_SUMMARY = """Summarize this candidate's fit for the role in a hiring-manager-friendly format.

CANDIDATE: {candidate_label}
FIT SCORE: {fit_score}/100

JD EXCERPT:
{jd_excerpt}

RESUME EXCERPT:
{resume_excerpt}

VALIDATED SKILLS:
{validated_skills}

Return a concise JSON summary suitable for a hiring manager who has 2 minutes to read it."""
