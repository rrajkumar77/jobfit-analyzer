"""
PII and client-sensitive data masking.

This is the original security_masker.py logic, restructured, with one
material fix: candidate NAME masking, which did not exist before.

Root cause of the fix: the original app pulled `candidate_name` from the
first line of the raw resume BEFORE masking ran, then used that raw name
in LLM prompts, on-screen labels, and exported filenames. That is how a
real candidate's name (Payal Patel) ended up in a committed .docx file in
a public GitHub repo — the masking checkbox said "PII Protected" but never
touched the one field guaranteed to appear in every resume.

Design decision: recruiters legitimately need to see the candidate's real
name in the UI — hiding it there would make the tool useless. What must
NOT happen is the real name leaking into (a) prompts sent to the third-party
LLM, (b) exported file names, or (c) any audit/log artifact. So this module
extracts the name once, keeps it only in memory for UI display, and returns
a masked placeholder for everything else.
"""

from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Tuple


class SensitivityLevel(Enum):
    PII_CRITICAL = "PII_CRITICAL"
    PII_HIGH = "PII_HIGH"
    PII_MEDIUM = "PII_MEDIUM"
    CLIENT_CONFIDENTIAL = "CLIENT_CONFIDENTIAL"
    CLIENT_INTERNAL = "CLIENT_INTERNAL"


@dataclass
class MaskingResult:
    masked_text: str
    mask_count: int
    sensitivity_detected: Dict[str, int] = field(default_factory=dict)
    masking_log: List[str] = field(default_factory=list)
    # NEW: the real name, extracted for UI display only. Never put this in
    # a prompt, filename, or exported document — use candidate_label instead.
    candidate_name: str | None = None
    # NEW: a safe, stable label to use anywhere the text leaves this process
    # (LLM prompts, filenames, audit logs). e.g. "Candidate-4f2a"
    candidate_label: str | None = None


class PIIMasker:
    """Masks PII from resume documents: email, phone, address, SSN, DOB, zip, and name."""

    PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone_us": r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
        "phone_intl": r'\b\+?[1-9]\d{0,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "zip_code": r'\b\d{5}(?:-\d{4})?\b',
        "address": r'\b\d+\s+[\w\s]+(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct|circle|cir|way)\b',
        "dob_slash": r'\b(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/(?:19|20)\d{2}\b',
        "dob_dash": r'\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b',
    }

    def __init__(self):
        self.compiled_patterns = {
            key: re.compile(pattern, re.IGNORECASE) for key, pattern in self.PATTERNS.items()
        }
        self._masked_cache: Dict[str, str] = {}

    def mask_resume(self, resume_text: str) -> MaskingResult:
        masked_text = resume_text
        mask_count = 0
        sensitivity_map: Dict[str, int] = {}
        masking_log: List[str] = []
        self._masked_cache.clear()

        # NEW: extract + mask the candidate name FIRST, before anything
        # else touches the text. Everything downstream — LLM calls, UI,
        # filenames — should use candidate_label, not the raw name.
        candidate_name = self._extract_candidate_name(resume_text)
        candidate_label = self._make_safe_label(candidate_name)

        if candidate_name:
            masked_text, name_count = self._mask_name_occurrences(masked_text, candidate_name)
            mask_count += name_count
            if name_count > 0:
                sensitivity_map["name"] = name_count
                masking_log.append(f"Masked {name_count} occurrence(s) of candidate name")

        for step in (
            self._mask_emails, self._mask_phones, self._mask_ssn,
            self._mask_addresses, self._mask_dob, self._mask_zip_codes,
        ):
            masked_text, count = step(masked_text)
            mask_count += count
            if count > 0:
                key = step.__name__.replace("_mask_", "")
                sensitivity_map[key] = count
                masking_log.append(f"Masked {count} {key} item(s)")

        return MaskingResult(
            masked_text=masked_text,
            mask_count=mask_count,
            sensitivity_detected=sensitivity_map,
            masking_log=masking_log,
            candidate_name=candidate_name,
            candidate_label=candidate_label,
        )

    # ------------------------------------------------------------------
    # Name handling — the fix
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_candidate_name(resume_text: str) -> str | None:
        """
        Heuristic: the candidate's name is almost always the first
        non-empty line of a resume, and is short (not a sentence).
        This mirrors what the original app assumed implicitly when it
        used `first_line` as candidate_name — the difference is we now
        treat that extracted value as sensitive rather than displaying
        and exporting it unmasked.
        """
        for line in resume_text.splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            # Reject lines that look like a heading/contact info rather
            # than a name: too long, contains @ or digits, etc.
            if len(candidate) > 50 or "@" in candidate or any(c.isdigit() for c in candidate):
                return None
            return candidate
        return None

    @staticmethod
    def _make_safe_label(name: str | None) -> str:
        """Stable, non-reversible label to use in prompts/filenames/logs."""
        if not name:
            return "Candidate"
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:6]
        return f"Candidate-{digest}"

    @staticmethod
    def _mask_name_occurrences(text: str, name: str) -> Tuple[str, int]:
        """
        Replace every occurrence of the candidate's name — both the full
        name and each individual token (first name, last name) — since
        resume bodies routinely refer to the candidate by first name alone
        ("Payal drove adoption of...").

        Longest-match-first ordering (full name before individual tokens)
        avoids double-counting when both would otherwise match the same
        span.
        """
        if not name:
            return text, 0

        tokens = [name] + [t for t in name.split() if len(t) > 1]
        # Longest first so "Payal Patel" is masked before a leftover "Payal".
        tokens = sorted(set(tokens), key=len, reverse=True)

        total_count = 0
        for token in tokens:
            pattern = re.compile(r'\b' + re.escape(token) + r'\b', re.IGNORECASE)
            text, count = pattern.subn("[Candidate Name Redacted]", text)
            total_count += count
        return text, total_count

    # ------------------------------------------------------------------
    # Existing masking (ported as-is from the original module)
    # ------------------------------------------------------------------

    def _mask_emails(self, text: str) -> Tuple[str, int]:
        count = 0

        def replace_email(match):
            nonlocal count
            email = match.group(0)
            if email not in self._masked_cache:
                count += 1
                self._masked_cache[email] = f"candidate.email{count}@example.com"
            return self._masked_cache[email]

        return self.compiled_patterns["email"].sub(replace_email, text), count

    def _mask_phones(self, text: str) -> Tuple[str, int]:
        count = 0

        def replace_phone(match):
            nonlocal count
            phone = match.group(0)
            if phone not in self._masked_cache:
                count += 1
                self._masked_cache[phone] = f"+1-555-000-{count:04d}"
            return self._masked_cache[phone]

        masked = self.compiled_patterns["phone_us"].sub(replace_phone, text)

        # BUG FIX (found by tests/test_masking.py): the international pattern
        # used to re-run over text that phone_us had *already* masked, since
        # our own placeholder ("+1-555-000-0001") itself looks like a phone
        # number. That produced garbled double-masked output like
        # "+++1-555-002-[Address Redacted]". Guard against re-matching our
        # own placeholders before applying the intl pattern.
        def replace_intl(match):
            nonlocal count
            phone = match.group(0)
            if phone.startswith("+1-555-"):
                return phone  # already one of our own placeholders — leave it
            if not re.search(r'[+\-.\s()]', phone):
                return phone
            if phone not in self._masked_cache:
                count += 1
                self._masked_cache[phone] = f"+1-555-{count:03d}-0000"
            return self._masked_cache[phone]

        if re.search(r'\b(?:phone|tel|mobile|cell)\b', text, re.IGNORECASE):
            masked = self.compiled_patterns["phone_intl"].sub(replace_intl, masked)

        return masked, count

    def _mask_ssn(self, text: str) -> Tuple[str, int]:
        count = 0

        def replace(match):
            nonlocal count
            count += 1
            return "***-**-****"

        return self.compiled_patterns["ssn"].sub(replace, text), count

    def _mask_addresses(self, text: str) -> Tuple[str, int]:
        count = 0

        def replace(match):
            nonlocal count
            count += 1
            return "[Address Redacted]"

        return self.compiled_patterns["address"].sub(replace, text), count

    def _mask_dob(self, text: str) -> Tuple[str, int]:
        count = 0

        def replace(match):
            nonlocal count
            count += 1
            return "MM/DD/YYYY"

        masked = self.compiled_patterns["dob_slash"].sub(replace, text)
        masked = self.compiled_patterns["dob_dash"].sub(replace, masked)
        return masked, count

    def _mask_zip_codes(self, text: str) -> Tuple[str, int]:
        count = 0

        def replace(match):
            nonlocal count
            context = text[max(0, match.start() - 50): match.end() + 50].lower()
            if re.search(r'\b(?:city|state|address|zip|postal|location)\b', context):
                count += 1
                return "XXXXX"
            return match.group(0)

        return self.compiled_patterns["zip_code"].sub(replace, text), count


class ClientSensitiveMasker:
    """Masks client-sensitive info from JD documents (unchanged from original, ported)."""

    PATTERNS = {
        "client_ref": r'\b(?:client|customer)[-\s]?(?:id|ref|code|number)[:\s]*[A-Z0-9-]+\b',
        "project_code": r'\b(?:project|proj|prj)[-\s]?(?:id|code|ref)[:\s]*[A-Z0-9-]+\b',
        "confidential": r'\b(?:confidential|proprietary|internal\s+only|do\s+not\s+share)\b',
        "budget": r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:k|K|M|million|thousand))?',
        "internal_code": r'\b[A-Z]{2,4}-\d{3,6}\b',
    }
    COMPANY_SUFFIXES = {
        "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited",
        "company", "co", "group", "holdings", "ventures", "partners", "technologies",
        "tech", "systems", "solutions", "services", "consulting", "enterprises",
    }

    def __init__(self):
        self.compiled_patterns = {
            key: re.compile(pattern, re.IGNORECASE) for key, pattern in self.PATTERNS.items()
        }
        self._masked_cache: Dict[str, str] = {}
        self._company_name_cache: Set[str] = set()

    def mask_jd(self, jd_text: str, known_client_names: List[str] | None = None) -> MaskingResult:
        masked_text = jd_text
        mask_count = 0
        sensitivity_map: Dict[str, int] = {}
        masking_log: List[str] = []
        self._masked_cache.clear()
        self._company_name_cache.clear()

        if known_client_names:
            for client_name in known_client_names:
                pattern = re.compile(re.escape(client_name), re.IGNORECASE)
                matches = len(pattern.findall(masked_text))
                if matches:
                    masked_text = pattern.sub("[Client Company]", masked_text)
                    mask_count += matches
                    masking_log.append(f"Masked {matches} instance(s) of known client")

        for step in (
            self._mask_client_refs, self._mask_project_codes,
            self._mask_confidential_markers, self._mask_budget, self._mask_internal_codes,
            self._mask_company_names,
        ):
            masked_text, count = step(masked_text)
            mask_count += count
            if count > 0:
                key = step.__name__.replace("_mask_", "")
                sensitivity_map[key] = count
                masking_log.append(f"Masked {count} {key} item(s)")

        return MaskingResult(
            masked_text=masked_text, mask_count=mask_count,
            sensitivity_detected=sensitivity_map, masking_log=masking_log,
        )

    def _mask_client_refs(self, text):
        return self._simple_replace(text, "client_ref", "[CLIENT-REF-REDACTED]")

    def _mask_project_codes(self, text):
        return self._simple_replace(text, "project_code", "[PROJECT-CODE-REDACTED]")

    def _mask_confidential_markers(self, text):
        return self._simple_replace(text, "confidential", "[CONFIDENTIAL-REDACTED]")

    def _simple_replace(self, text: str, pattern_key: str, replacement: str) -> Tuple[str, int]:
        count = 0

        def replace(match):
            nonlocal count
            count += 1
            return replacement

        return self.compiled_patterns[pattern_key].sub(replace, text), count

    def _mask_budget(self, text: str) -> Tuple[str, int]:
        count = 0

        def replace(match):
            nonlocal count
            context = text[max(0, match.start() - 30): match.end() + 30].lower()
            if re.search(r'\b(?:budget|cost|price|rate|salary|compensation|pay)\b', context):
                count += 1
                return "$[REDACTED]"
            return match.group(0)

        return self.compiled_patterns["budget"].sub(replace, text), count

    def _mask_internal_codes(self, text: str) -> Tuple[str, int]:
        count = 0
        common_tech = {"AWS", "API", "SQL", "GCP", "PDF", "CSV", "XML", "JSON", "HTTP", "HTTPS", "REST", "SOAP"}

        def replace(match):
            nonlocal count
            code = match.group(0)
            if code.upper() in common_tech:
                return code
            count += 1
            return "[INTERNAL-CODE-REDACTED]"

        return self.compiled_patterns["internal_code"].sub(replace, text), count

    def _mask_company_names(self, text: str) -> Tuple[str, int]:
        count = 0
        pattern = re.compile(
            r'\b([A-Z][a-z]*(?:\s+[A-Z][a-z]*)*)\s+(' +
            '|'.join(re.escape(s) for s in self.COMPANY_SUFFIXES) + r')\.?\b',
            re.IGNORECASE,
        )
        tech_keywords = {
            "microsoft", "amazon", "google", "oracle", "adobe", "apple",
            "python", "java", "react", "angular", "node",
        }

        def replace(match):
            nonlocal count
            full_match = match.group(0)
            if any(t in full_match.lower() for t in tech_keywords):
                return full_match
            if full_match in self._company_name_cache:
                return "[Client Company]"
            self._company_name_cache.add(full_match)
            count += 1
            return "[Client Company]"

        return pattern.sub(replace, text), count


class SecurityMasker:
    """Unified interface — same call signature as the original."""

    def __init__(self):
        self.pii_masker = PIIMasker()
        self.client_masker = ClientSensitiveMasker()

    def mask_resume(self, resume_text: str) -> MaskingResult:
        return self.pii_masker.mask_resume(resume_text)

    def mask_jd(self, jd_text: str, known_client_names: List[str] | None = None) -> MaskingResult:
        return self.client_masker.mask_jd(jd_text, known_client_names)

    def get_masking_summary(self, result: MaskingResult) -> str:
        if result.mask_count == 0:
            return "✓ No sensitive information detected"
        lines = [f"🔒 Masked {result.mask_count} sensitive item(s):"]
        lines.extend(f"  • {log}" for log in result.masking_log)
        return "\n".join(lines)


def create_masking_audit_log(result: MaskingResult, doc_type: str) -> Dict:
    """Audit entry — deliberately contains no sensitive data, only counts/types."""
    import datetime

    return {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "document_type": doc_type,
        "mask_count": result.mask_count,
        "sensitivity_types": list(result.sensitivity_detected.keys()),
        "status": "success",
    }
