"""
Tests for masking.py — most importantly, tests that would have caught the
original bug: a candidate's real name leaking into masked_text,
candidate_label, or any exported artifact.

Run with: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from masking import PIIMasker, SecurityMasker  # noqa: E402


SAMPLE_RESUME = """Payal Patel
Email: payal.patel@example.com
Phone: +1-555-123-4567

PROFESSIONAL EXPERIENCE
Senior Product Manager at TechCorp Inc.
- Led cross-functional teams of 12+ engineers
- Payal drove adoption of new analytics platform across 5 business units
"""


def test_name_is_extracted():
    masker = PIIMasker()
    result = masker.mask_resume(SAMPLE_RESUME)
    assert result.candidate_name == "Payal Patel"


def test_real_name_never_appears_in_masked_text():
    """This is the regression test for the original leak."""
    masker = PIIMasker()
    result = masker.mask_resume(SAMPLE_RESUME)
    assert "Payal" not in result.masked_text
    assert "Patel" not in result.masked_text


def test_candidate_label_does_not_contain_real_name():
    masker = PIIMasker()
    result = masker.mask_resume(SAMPLE_RESUME)
    assert "Payal" not in result.candidate_label
    assert "Patel" not in result.candidate_label
    assert result.candidate_label.startswith("Candidate-")


def test_candidate_label_is_stable_for_same_name():
    """Same name should always produce the same label within a run,
    so a recruiter can tell two documents refer to the same candidate
    without ever seeing the real name."""
    masker = PIIMasker()
    r1 = masker.mask_resume(SAMPLE_RESUME)
    r2 = masker.mask_resume(SAMPLE_RESUME)
    assert r1.candidate_label == r2.candidate_label


def test_email_and_phone_still_masked():
    masker = PIIMasker()
    result = masker.mask_resume(SAMPLE_RESUME)
    assert "payal.patel@example.com" not in result.masked_text
    assert "555-123-4567" not in result.masked_text


def test_security_masker_end_to_end():
    masker = SecurityMasker()
    result = masker.mask_resume(SAMPLE_RESUME)
    assert result.candidate_label is not None
    assert "Payal" not in result.masked_text


def test_no_name_line_does_not_crash():
    """Resume that doesn't start with a clean name line shouldn't raise."""
    masker = PIIMasker()
    weird_resume = "email@example.com\nSome experience here."
    result = masker.mask_resume(weird_resume)
    assert result.candidate_label == "Candidate"  # safe fallback
