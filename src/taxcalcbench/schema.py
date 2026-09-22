"""Snake-case dataset rows and local checks, without a legal-reasoning rubric."""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .tools import LawTools

QUESTION_HEADERS = ["no", "family", "variant", "country", "language", "tax_category", "primary_taxpayer",
                    "year_period_target", "case_year", "difficulty", "question", "final_answer_text",
                    "answer_value", "unit", "gold_steps", "status", "note"]
CITATION_HEADERS = ["question_no", "language", "citation_id", "official_title", "law_number", "article_section",
                    "paragraph", "subparagraph", "rule_applies_from", "rule_applies_until", "official_source_url",
                    "source_file", "supporting_passage"]
CATEGORIES = {
    "1000": "Income, profits and capital gains", "2000": "Social security contributions",
    "3000": "Payroll and workforce taxes", "4000": "Property taxes",
    "5000": "Goods and services taxes", "6000": "Other taxes",
}
TAXPAYERS = ["Individual", "Self-employed / sole proprietor", "Company / corporation", "Partnership",
            "Estate / trust", "Nonprofit / tax-exempt entity", "Other taxpayer"]
VARIANTS = ["original", "numbers", "threshold", "eligibility", "temporal", "missing_information"]
STATUSES = ["draft", "ready", "needs_correction", "validated"]
DIFFICULTIES = ["D2", "D3", "D4"]
UNITS = ["KZT", "INR", "PKR", "CNY", "IDR", "PLN", "EGP", "USD", "EUR", "percent",
         "percentage_points", "days", "months", "years", "L", "kg", "tonne", "m2", "count"]
Text = Annotated[str, Field(strict=True, min_length=1, max_length=40000)]
DECIMAL_PATTERN = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
INSUFFICIENT_INFORMATION = "insufficient information"
Variant = Literal["original", "numbers", "threshold", "eligibility", "temporal", "missing_information"]


def decimal_value(value: str) -> Decimal:
    """Read an exact, finite decimal string, without thousands separators."""
    if not isinstance(value, str) or re.fullmatch(DECIMAL_PATTERN, value) is None:
        raise ValueError("Use a plain decimal string without thousands separators")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("Invalid decimal answer") from exc
    if not number.is_finite() or len(number.as_tuple().digits) > 100 or abs(number.adjusted()) > 100:
        raise ValueError("Decimal answer is outside the supported range")
    return number


class Row(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="after")
    @classmethod
    def nonblank_strings(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("Use null for an unused optional cell, not blank text")
        return value


class GoldStep(Row):
    step: Text = Field(description="Explain this legal decision or calculation; for missing information, identify the missing fact and why it is necessary.")
    result: Text
    unit: Text | None = Field(description="Unit of the result; null for a dimensionless ratio or conclusion.")
    citations: list[Annotated[str, Field(pattern=r"^C[1-9][0-9]*$")]] = Field(min_length=1)

    @field_validator("citations")
    @classmethod
    def distinct_citations(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("A gold step must not repeat a citation ID")
        return value


class QuestionRow(Row):
    no: int = Field(ge=1, strict=True)
    family: Text
    variant: Variant
    country: str = Field(pattern=r"^[A-Z]{2}$")
    language: str = Field(pattern=r"^[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
    tax_category: Literal["Income, profits and capital gains", "Social security contributions",
                          "Payroll and workforce taxes", "Property taxes", "Goods and services taxes",
                          "Other taxes"]
    primary_taxpayer: Literal["Individual", "Self-employed / sole proprietor", "Company / corporation", "Partnership",
                             "Estate / trust", "Nonprofit / tax-exempt entity", "Other taxpayer"]
    year_period_target: Text
    case_year: int = Field(ge=1, le=9999, strict=True)
    difficulty: Literal["D2", "D3", "D4"]
    question: Text
    final_answer_text: Text = Field(description="Full answer in the run language. For missing information, explain the absent fact and why it prevents a unique answer.")
    answer_value: Text
    unit: Text | None
    gold_steps: list[GoldStep] = Field(min_length=1, description=(
        "Ordered steps. Final result and unit equal answer_value and unit. Cite every step's legal/calculation rule. "
        "For missing information, final result is exactly 'insufficient information' and final unit is null."))
    status: Literal["draft", "ready", "needs_correction", "validated"]
    note: Text | None

    @model_validator(mode="after")
    def answer_contract(self):
        if self.variant == "missing_information":
            if self.answer_value != INSUFFICIENT_INFORMATION or self.unit is not None:
                raise ValueError("A missing-information answer requires 'insufficient information' and unit null")
            if self.final_answer_text.strip() == INSUFFICIENT_INFORMATION:
                raise ValueError("The full answer must explain which information is missing and why")
            if (self.gold_steps[-1].result != INSUFFICIENT_INFORMATION
                    or self.gold_steps[-1].unit is not None):
                raise ValueError("The final missing-information gold step requires 'insufficient information' and unit null")
            if not any(step.step.strip() != INSUFFICIENT_INFORMATION for step in self.gold_steps):
                raise ValueError("Gold steps must explain which information is missing and why")
        else:
            decimal_value(self.answer_value)
            if self.unit is None:
                raise ValueError("A numeric answer requires a unit")
        return self


class CitationRow(Row):
    question_no: int = Field(ge=1, strict=True)
    language: str = Field(pattern=r"^[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
    citation_id: str = Field(pattern=r"^C[1-9][0-9]*$")
    official_title: Text
    law_number: Text | None
    article_section: Text
    paragraph: Text | None
    subparagraph: Text | None
    rule_applies_from: str | None
    rule_applies_until: str | None
    official_source_url: Text | None
    source_file: Text = Field(description="Use an existing source inventory ID; the program fills its saved raw path.")
    supporting_passage: Text

    @field_validator("rule_applies_from", "rule_applies_until")
    @classmethod
    def iso_date(cls, value: str | None) -> str | None:
        if value is not None:
            if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is None:
                raise ValueError("Applicability dates use YYYY-MM-DD")
            date.fromisoformat(value)
        return value

    @model_validator(mode="after")
    def ordered_dates(self):
        if (self.rule_applies_from is not None and self.rule_applies_until is not None
                and self.rule_applies_until < self.rule_applies_from):
            raise ValueError("Applicability end date precedes its start")
        return self


class FamilyDraft(Row):
    topic: Text
    questions: list[QuestionRow] = Field(min_length=1)
    citations: list[CitationRow] = Field(min_length=1)


class QuestionDraft(Row):
    """One writer's handoff, assembled into a family before export."""

    question: QuestionRow
    citations: list[CitationRow] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def citations_join_question(self):
        if any(citation.question_no != self.question.no or citation.language != self.question.language
               for citation in self.citations):
            raise ValueError("Every citation must refer to this question number and language")
        return self


class MissingDerivation(Row):
    """A compact edit to an approved case; inherited law is never regenerated."""

    omitted_text: Text
    missing_fact: Text
    final_answer_text: Text
    gold_steps: list[GoldStep] = Field(min_length=1)
    note: Text


class EvidenceSpan(Row):
    """An internal pointer into verified local text, never an exported citation."""

    source_id: Text
    start_line: int = Field(ge=1, strict=True)
    end_line: int = Field(ge=1, strict=True)
    purpose: Text

    @model_validator(mode="after")
    def readable_range(self):
        if self.end_line < self.start_line:
            raise ValueError("Evidence line range is reversed")
        if self.end_line - self.start_line >= 400:
            raise ValueError("Split evidence into ranges of at most 400 lines")
        return self


class LegalParameter(Row):
    """A statutory calculation dependency and its official-source evidence."""

    name: Text
    value: Text
    evidence: list[EvidenceSpan] = Field(min_length=1, max_length=30)


class PlannedCase(Row):
    """One substantive scenario for a writer to turn into a workbook row."""

    question_no: int = Field(ge=1, strict=True)
    variant: Variant
    case_year: int = Field(ge=1, le=9999, strict=True)
    tax_category: Literal["Income, profits and capital gains", "Social security contributions",
                          "Payroll and workforce taxes", "Property taxes", "Goods and services taxes", "Other taxes"]
    primary_taxpayer: Literal["Individual", "Self-employed / sole proprietor", "Company / corporation", "Partnership",
                             "Estate / trust", "Nonprofit / tax-exempt entity", "Other taxpayer"]
    scenario_outline: Text
    legal_difference: Text = Field(description=(
        "For the original, describe the interacting rules. For a regular variant, identify its substantive change. "
        "For temporal cases, preserve the referenced scenario and change only its year, checking that year's law. "
        "For missing information, identify the necessary fact omitted from the referenced case."))
    legal_assumptions: list[Text] = Field(min_length=1, max_length=20, description=(
        "Material taxpayer facts, exclusions, rule applicability and calculation endpoint that the writer "
        "must make explicit; legal assertions must be supported by evidence."))
    evidence: list[EvidenceSpan] = Field(min_length=1, max_length=30)
    legal_parameters: list[LegalParameter] = Field(default_factory=list, description=(
        "All statutory rates, thresholds, indices, minima and other legal calculation parameters "
        "needed for this case, each with its value and official evidence. Exclude factual scenario "
        "amounts. Every regular case requires at least one parameter; an exemption may use value 0."))
    derived_from: int | None = Field(default=None, ge=1, strict=True, description=(
        "REQUIRED non-null for temporal and missing_information: question number of the earlier regular parent. "
        "Use null for the original. A temporal variant preserves its parent's facts except the tax year."))
    missing_fact: Text | None = Field(default=None, description=(
        "For missing_information only, the one material fact intentionally removed and why law needs it."))

    @model_validator(mode="after")
    def variant_metadata(self):
        if self.variant in {"temporal", "missing_information"} and self.derived_from is None:
            raise ValueError("Temporal and missing-information cases require derived_from")
        if self.variant == "original" and self.derived_from is not None:
            raise ValueError("The original case cannot derive from another case")
        if (self.variant == "missing_information") != (self.missing_fact is not None):
            raise ValueError("Only missing_information cases require a missing_fact explanation")
        return self


class FamilyPlan(Row):
    """Internal planning handoff; not part of either workbook-shaped export."""

    family_id: Text
    topic: Text
    cases: list[PlannedCase] = Field(min_length=1)
    category_reallocation_reason: Text | None = Field(default=None, description=(
        "Explain why official evidence supports another enabled category better than the assigned target; "
        "otherwise null. This planning metadata is not exported."))

    @model_validator(mode="after")
    def distinct_cases_and_dependencies(self):
        if len({case.question_no for case in self.cases}) != len(self.cases):
            raise ValueError("A plan requires distinct question numbers")
        if self.cases[0].variant != "original" or sum(case.variant == "original" for case in self.cases) != 1:
            raise ValueError("A plan requires exactly one original first")
        preceding = {}
        for case in self.cases:
            if case.derived_from is not None:
                parent = preceding.get(case.derived_from)
                if parent is None or parent.variant == "missing_information":
                    raise ValueError("derived_from must identify an earlier regular case in this family")
                if case.variant == "temporal" and case.case_year == parent.case_year:
                    raise ValueError("A temporal case must use a different year from its referenced case")
                if case.variant in {"temporal", "missing_information"} and (
                    case.tax_category != parent.tax_category or case.primary_taxpayer != parent.primary_taxpayer
                ):
                    raise ValueError("Temporal and missing-information cases preserve category and taxpayer")
                if case.variant == "missing_information" and case.case_year != parent.case_year:
                    raise ValueError("A missing-information case preserves its referenced case year")
            preceding[case.question_no] = case
        return self


class ReviewItem(Row):
    """One regular-question verdict; export row schemas remain strict."""

    model_config = ConfigDict(extra="ignore")

    question_no: int = Field(ge=1, strict=True)
    passed: bool = Field(strict=True)
    issues: list[Text]
    recomputed_answer: Text | None
    unit: Text | None
    calculation: Text | None

    @model_validator(mode="after")
    def verdict_contract(self):
        if self.issues:
            self.passed = False
        if self.passed:
            if any(value is None for value in (self.recomputed_answer, self.unit, self.calculation)):
                raise ValueError("A passed review requires a recomputed answer, unit and calculation")
            decimal_value(self.recomputed_answer)
        elif not self.issues:
            raise ValueError("A rejected question requires a concrete issue")
        return self


class ReviewReport(Row):
    family_id: Text
    reviewer: Literal["A", "B"]
    questions: list[ReviewItem] = Field(min_length=1)
    family_issues: list[Text]
    plan_revision_needed: bool = Field(default=False, strict=True, description=(
        "True only when a planned year or scenario needs revision, such as a historical rule version "
        "that the available official sources cannot establish. Ordinary answer/citation repairs are false."))


def _source_quote(quote: str, text: str) -> str:
    if quote in text:
        return quote
    normalized, starts, ends = [], [], []
    cursor = 0
    while cursor < len(text):
        start = cursor
        character = text[cursor]
        cursor += 1
        if character.isspace():
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            character = " "
        normalized.append(character)
        starts.append(start)
        ends.append(cursor)
    target = re.sub(r"\s+", " ", quote)
    searchable = "".join(normalized)
    index = searchable.find(target)
    if not target or index < 0:
        raise ValueError("Supporting passage has no contiguous exact or whitespace-equivalent source match")
    if searchable.find(target, index + 1) >= 0:
        raise ValueError("Supporting passage has ambiguous whitespace-equivalent source matches")
    return text[starts[index]:ends[index + len(target) - 1]]


def _step_failures(question: QuestionRow, available: set[str]) -> list[str]:
    failures = []
    for number, step in enumerate(question.gold_steps, 1):
        if not set(step.citations).issubset(available):
            failures.append(f"Gold step {number} contains a missing citation for this question/language")
    final = question.gold_steps[-1]
    if question.variant == "missing_information":
        if final.result != INSUFFICIENT_INFORMATION or final.unit is not None:
            failures.append("Final gold_steps result/unit differs from the missing-information answer")
    else:
        try:
            if decimal_value(final.result) != decimal_value(question.answer_value) or final.unit != question.unit:
                failures.append("Final gold_steps result/unit differs from answer_value/unit")
        except ValueError:
            failures.append("Final gold_steps result must be the numeric answer_value")
    return failures


def normalize_and_check_family(
    draft: FamilyDraft, config: dict, inventory: dict, source_dir: Path, *,
    family_id: str, question_numbers: list[int],
) -> tuple[FamilyDraft, list[str]]:
    """Bind source locations/quotes, then check a drafted family before review.

    This never approves a family or claims that quoted law entails its answer.
    Preserve the original draft separately; the return value is a fresh record.
    """
    result = FamilyDraft.model_validate(draft.model_dump(mode="json", by_alias=True))
    failures: list[str] = []
    if not question_numbers or len(set(question_numbers)) != len(question_numbers):
        raise ValueError("Assign one or more distinct question numbers")
    if len(result.questions) != len(question_numbers) or {q.no for q in result.questions} != set(question_numbers):
        failures.append("Family must contain the assigned question numbers exactly once")
    if len({q.no for q in result.questions}) != len(result.questions):
        failures.append("Family has duplicate question numbers")
    if sum(q.variant == "original" for q in result.questions) != 1:
        failures.append("Family requires exactly one original")
    periods = {period["label"]: period for period in config["generation"]["periods"]}
    questions = {(q.no, q.language): q for q in result.questions}
    tools = LawTools(inventory, source_dir)
    source_files: dict[str, list[str]] = {}
    for source_id, document in tools.documents.items():
        source_files.setdefault(document.get("raw_path", ""), []).append(source_id)
    citation_keys = set()
    citation_ids: dict[tuple[int, str], set[str]] = {}
    for citation in result.citations:
        label = f"Question {citation.question_no}/{citation.language}/{citation.citation_id}"
        key = (citation.question_no, citation.language)
        unique = (*key, citation.citation_id)
        if unique in citation_keys:
            failures.append(f"{label}: duplicate citation ID")
        citation_keys.add(unique)
        citation_ids.setdefault(key, set()).add(citation.citation_id)
        if key not in questions:
            failures.append(f"{label}: citation does not join an assigned question/language")
        source_id = citation.source_file if citation.source_file in tools.documents else None
        if source_id is None:
            matches = source_files.get(citation.source_file, [])
            source_id = matches[0] if len(matches) == 1 else None
        if source_id is None:
            failures.append(f"{label}: Source file must identify one inventory source")
            continue
        document = tools.documents[source_id]
        if document.get("country") != config["country"] or document.get("language") != config["language"]:
            failures.append(f"{label}: source country/language differs from this run")
        if document.get("kind") == "navigation":
            failures.append(f"{label}: navigation page is not an eligible legal source")
        citation.source_file = document["raw_path"]
        citation.official_source_url = document.get("final_url") or document["url"]
        try:
            citation.supporting_passage = _source_quote(citation.supporting_passage, tools.source_text(source_id))
            if "[UNAVAILABLE PROVISION" in citation.supporting_passage:
                raise ValueError("Unavailable provision markers are not legal evidence")
        except ValueError as exc:
            failures.append(f"{label}: {exc}")
    signatures = set()
    for question in result.questions:
        label = f"Question {question.no}/{question.language}"
        if any(isinstance(value, str) and value.lstrip().startswith("=")
               for value in question.model_dump().values()):
            failures.append(f"{label}: materialize workbook values instead of spreadsheet formula strings")
        if (question.family, question.country, question.language) != (family_id, config["country"], config["language"]):
            failures.append(f"{label}: family, country or language differs from the assignment")
        if question.status == "validated":
            failures.append(f"{label}: an author cannot assign validated status")
        if question.variant != "original" and question.note is None:
            failures.append(f"{label}: a variant needs a change explanation in note")
        period = next((item for item in periods.values()
                       if item["start"] <= question.case_year <= item["end"]), None)
        if period is None:
            failures.append(f"{label}: Case Year does not match its configured target period")
        else:
            # This is derived metadata, just like the bound source path/URL.
            question.year_period_target = period["label"]
        if question.variant != "missing_information" and question.unit not in set(UNITS) | {config.get("currency")}:
            allowed_units = sorted(unit for unit in set(UNITS) | {config.get("currency")} if unit)
            failures.append(f"{label}: unit is not a workbook/configured unit; allowed values: {', '.join(allowed_units)}")
        available = citation_ids.get((question.no, question.language), set())
        if not available:
            failures.append(f"{label}: no citation rows")
        failures.extend(f"{label}: {failure}" for failure in _step_failures(question, available))
        steps_text = "\n".join(" ".join((step.step, step.result, step.unit or "", ",".join(step.citations)))
                               for step in question.gold_steps)
        signature = re.sub(r"[0-9]+(?:[.,][0-9]+)*", "#", " ".join(
            (question.question + "\n" + steps_text).casefold().split()))
        if signature in signatures and question.variant != "temporal":
            failures.append(f"{label}: duplicates another case with only numeric/date changes")
        if question.variant != "temporal":
            signatures.add(signature)
    result.questions.sort(key=lambda row: question_numbers.index(row.no) if row.no in question_numbers else len(question_numbers))
    result.citations.sort(key=lambda row: (row.question_no, row.language, int(row.citation_id[1:])))
    return result, list(dict.fromkeys(failures))
