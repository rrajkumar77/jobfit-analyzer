"""
Skill extraction + candidate validation.

This is the actual matching algorithm — ported from simple_top5_validator.py
with two changes:
  1. It goes through JobFitLLMClient instead of calling groq.Groq() directly,
     so it inherits the timeout + retry + robust JSON parsing for free.
  2. The JD cache key uses hashlib.sha256 instead of Python's built-in
     hash(). Python randomizes string hash() per-process by default (PYTHONHASHSEED),
     so the original cache would silently miss across app restarts — not a
     security bug, but a correctness one that undercuts the "cost optimised"
     claim in the original docstring.

The semantic-bridging approach in the prompts (crediting transferable
experience rather than requiring exact keyword matches) is unchanged —
that part of the original design was sound and is worth keeping.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import prompts
from config import Settings, get_logger
from llm_client import JobFitLLMClient, LLMResponseError

logger = get_logger(__name__)


@dataclass
class SkillValidation:
    skill_name: str
    has_project_experience: bool
    validation_score: float  # 0-100
    evidence_summary: str
    project_example: str


class JobFitValidator:
    def __init__(self, settings: Settings, llm_client: JobFitLLMClient | None = None):
        self._settings = settings
        self._llm = llm_client or JobFitLLMClient(settings)
        self._jd_cache: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # JD skill extraction (large model, cached per JD)
    # ------------------------------------------------------------------

    def extract_top_5_skills(self, jd_text: str) -> List[str]:
        cache_key = hashlib.sha256(jd_text[:4000].encode("utf-8")).hexdigest()
        if cache_key in self._jd_cache:
            logger.info("JD skill cache hit")
            return self._jd_cache[cache_key]

        prompt = prompts.EXTRACT_TOP_5_SKILLS.format(jd_text=jd_text[:4000])

        try:
            result = self._llm.complete_json(
                prompt,
                model=self._settings.jd_model,
                temperature=0.1,
                max_tokens=400,
                label="extract_top_5_skills",
            )
            skills = result.data
            if not (isinstance(skills, list) and len(skills) >= 5):
                raise LLMResponseError("Expected a JSON array of >= 5 skills")
            skills = skills[:5]
        except (LLMResponseError, Exception) as e:  # noqa: BLE001
            logger.warning("Skill extraction failed, using fallback: %s", e)
            skills = self._fallback_extract_skills(jd_text)

        self._jd_cache[cache_key] = skills
        return skills

    # ------------------------------------------------------------------
    # Candidate validation (small model, batched — 1 call for all skills)
    # ------------------------------------------------------------------

    def validate_candidate(
        self, top_5_skills: List[str], resume_text: str
    ) -> Tuple[float, List[SkillValidation]]:
        if not top_5_skills:
            return 0.0, []

        try:
            validations = self._validate_all_skills_batched(top_5_skills, resume_text)
        except Exception as e:  # noqa: BLE001
            logger.warning("Batched validation failed, using fallback: %s", e)
            validations = [self._fallback_validate_skill(s, resume_text) for s in top_5_skills]

        fit_score = (
            sum(v.validation_score for v in validations) / len(validations)
            if validations else 0.0
        )
        return fit_score, validations

    def _validate_all_skills_batched(
        self, skills: List[str], resume_text: str
    ) -> List[SkillValidation]:
        skills_numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(skills))
        prompt = prompts.VALIDATE_CANDIDATE_BATCH.format(
            resume_text=resume_text[:2500], skills_numbered=skills_numbered,
        )

        result = self._llm.complete_json(
            prompt,
            model=self._settings.validation_model,
            temperature=0.1,
            max_tokens=1000,
            label="validate_candidate_batch",
        )
        data_list = result.data
        if not isinstance(data_list, list):
            raise LLMResponseError("Expected a JSON array from validation call")

        validations: List[SkillValidation] = []
        for i, skill in enumerate(skills):
            if i < len(data_list):
                d = data_list[i]
                validations.append(SkillValidation(
                    skill_name=skill,
                    has_project_experience=bool(d.get("has_project_experience", False)),
                    validation_score=float(d.get("score", 0)),
                    evidence_summary=d.get("evidence", ""),
                    project_example=d.get("example", "No evidence found"),
                ))
            else:
                validations.append(self._fallback_validate_skill(skill, ""))
        return validations

    # ------------------------------------------------------------------
    # Fallbacks (no LLM / LLM failure)
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_extract_skills(jd_text: str) -> List[str]:
        jd_lower = jd_text.lower()
        skill_patterns = {
            "product management": "Product Management (roadmap, stakeholder coordination, delivery)",
            "program management": "Program Management (cross-functional delivery, risk management)",
            "data science": "Data Science (ML models, analytics, data pipelines)",
            "software engineering": "Software Engineering (coding, architecture, deployment)",
            "cloud": "Cloud Infrastructure (AWS/Azure/GCP, DevOps, scaling)",
            "machine learning": "Machine Learning (model development, training, deployment)",
            "stakeholder": "Stakeholder Management (communication, alignment, leadership)",
            "agile": "Agile Methodology (sprint planning, delivery, collaboration)",
        }
        found = [label for kw, label in skill_patterns.items() if kw in jd_lower]

        generic = [
            "Technical Proficiency (tools and technologies for the role)",
            "Domain Knowledge (industry-specific expertise)",
            "Project Delivery (end-to-end execution and results)",
            "Communication Skills (stakeholder engagement and documentation)",
            "Problem Solving (analytical thinking and solution design)",
        ]
        for g in generic:
            if len(found) >= 5:
                break
            if g not in found:
                found.append(g)
        return found[:5]

    @staticmethod
    def _fallback_validate_skill(skill: str, resume_text: str) -> SkillValidation:
        skill_lower, resume_lower = skill.lower(), resume_text.lower()
        if skill_lower not in resume_lower:
            return SkillValidation(skill, False, 0, "Not found in resume", "No evidence found")

        pos = resume_lower.find(skill_lower)
        context = resume_lower[max(0, pos - 200): pos + 200]
        action_verbs = ("built", "developed", "led", "designed", "implemented",
                         "created", "delivered", "shipped", "deployed")
        has_action = any(v in context for v in action_verbs)
        has_outcome = any(p in context for p in ("%", "reduced", "increased", "improved"))

        if has_action and has_outcome:
            return SkillValidation(skill, True, 75, "Found with action verbs and outcomes", "See resume")
        if has_action:
            return SkillValidation(skill, True, 50, "Found with action verbs, no clear outcome", "See resume")
        return SkillValidation(skill, False, 25, "Mentioned but no project context", "See resume")

    # ------------------------------------------------------------------
    # Report generation (zero LLM calls)
    # ------------------------------------------------------------------

    @staticmethod
    def generate_report(
        candidate_label: str, fit_score: float, validations: List[SkillValidation]
    ) -> str:
        """
        NOTE: takes candidate_label (masked), not the real name — this is
        what gets written into any exported/downloadable file.
        """
        verdict = (
            "✅ **STRONG FIT**" if fit_score >= 75
            else "⚠️ **CONDITIONAL FIT**" if fit_score >= 60
            else "❌ **WEAK FIT**"
        )
        lines = [
            f"# Validation Report: {candidate_label}", "",
            f"## Overall Fit Score: {fit_score:.0f}/100", "", verdict, "", "---", "",
            f"## Top {len(validations)} Skills Assessment", "",
        ]
        for i, v in enumerate(validations, 1):
            icon = "✅" if v.has_project_experience else "❌"
            lines += [
                f"### {i}. {icon} {v.skill_name} - {v.validation_score:.0f}%", "",
                f"**Has Project Experience**: {'Yes' if v.has_project_experience else 'No'}", "",
                f"**Evidence**: {v.evidence_summary}", "",
                f"**Example**: {v.project_example}", "", "---", "",
            ]
        validated = sum(1 for v in validations if v.has_project_experience)
        lines += [
            "## Summary", "",
            f"- **Skills with Project Experience**: {validated}/{len(validations)}",
            f"- **Average Validation Score**: {fit_score:.0f}%", "",
            "## Recommendation", "",
        ]
        if fit_score >= 75:
            lines.append("✅ **PROCEED TO INTERVIEW** – Strong candidate with validated project experience")
        elif fit_score >= 60:
            lines.append("⚠️ **INTERVIEW WITH CAUTION** – Some gaps in project evidence. Ask targeted questions.")
        else:
            lines.append("❌ **NOT RECOMMENDED** – Insufficient project experience in key skills")
        return "\n".join(lines)
