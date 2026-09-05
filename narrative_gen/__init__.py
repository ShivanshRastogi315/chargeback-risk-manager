"""
Module 3 — Grounded Narrative Generator Package (narrative_gen)

Provides template-constrained evidence generation, strict machine citation verification,
and PII / tone compliance guardrails for auditable dispute representments.
"""

from narrative_gen.generator import NarrativeGenerator, RepresentmentPacket
from narrative_gen.citation_validator import CitationValidator, ValidationReport, CitationTag, SentenceTrace
from narrative_gen.guardrails import GuardrailRunner, GuardrailReport, PIIChecker, ProhibitedLanguageChecker, RequiredFieldsChecker
from narrative_gen.templates import StructuredTemplateEngine, NarrativeSection

__all__ = [
    "NarrativeGenerator",
    "RepresentmentPacket",
    "CitationValidator",
    "ValidationReport",
    "CitationTag",
    "SentenceTrace",
    "GuardrailRunner",
    "GuardrailReport",
    "PIIChecker",
    "ProhibitedLanguageChecker",
    "RequiredFieldsChecker",
    "StructuredTemplateEngine",
    "NarrativeSection",
]
