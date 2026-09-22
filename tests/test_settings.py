from collections import Counter, defaultdict
from copy import deepcopy

import pytest

from taxcalcbench.schema import CATEGORIES
from taxcalcbench.settings import allocate_counts, build_assignments, category_options, validate_generation


def config():
    return {"country": "XY", "language": "fr", "generation": {
        "families": 50, "regular_questions_per_family": 5,
        "missing_information_per_family": 1, "min_periods_per_family": 2,
        "categories": [{"code": code, "label": label, "weight": 1} for code, label in CATEGORIES.items()],
        "periods": [{"label": "recent", "start": 2020, "end": 2026, "weight": 3},
                    {"label": "middle", "start": 2010, "end": 2019, "weight": 1},
                    {"label": "early", "start": 2000, "end": 2009, "weight": 1}],
    }}


def test_default_family_layout_balances_periods_in_every_category():
    settings = config()
    before = deepcopy(settings)
    assignments = build_assignments(settings, 25, ("A", "B"))
    numbers = []
    for creator, families in assignments.items():
        category_counts = Counter(family["category"] for family in families)
        assert sorted(category_counts.values()) == [4, 4, 4, 4, 4, 5]
        periods = Counter()
        for index, family in enumerate(families):
            assert family["family_id"] == f"{creator}-F{index + 1:02d}"
            assert len(family["regular_question_numbers"]) == 5
            assert len(family["missing_question_numbers"]) == 1
            assert family["question_numbers"] == family["regular_question_numbers"] + family["missing_question_numbers"]
            mix = Counter(slot["label"] for slot in family["period_slots"])
            assert len(mix) >= 2
            assert [slot["question_no"] for slot in family["period_slots"]] == family["regular_question_numbers"]
            periods.update(mix)
            numbers.extend(family["question_numbers"])
        assert periods == {"recent": 75, "middle": 25, "early": 25}
    assert numbers == list(range(1, 301))
    assert settings == before


def test_one_family_still_has_temporal_variation_and_optional_missing_rows():
    settings = config()
    settings["generation"]["missing_information_per_family"] = 0
    family = build_assignments(settings, 1, ("A",))["A"][0]
    assert family["question_numbers"] == [1, 2, 3, 4, 5]
    assert family["missing_question_numbers"] == []
    assert [slot["label"] for slot in family["period_slots"]] == ["recent", "middle", "early", "recent", "recent"]


def test_fifty_family_single_creator_default_has_balanced_category_year_totals():
    families = build_assignments(config(), 50)["A"]
    totals, by_category = Counter(), defaultdict(Counter)
    for family in families:
        mix = Counter(slot["label"] for slot in family["period_slots"])
        assert mix == {"recent": 3, "middle": 1, "early": 1}
        totals.update(mix)
        by_category[family["category"]].update(mix)
    assert totals == {"recent": 150, "middle": 50, "early": 50}
    assert sum(len(family["missing_question_numbers"]) for family in families) == 50
    assert sorted(Counter(family["category"] for family in families).values()) == [8, 8, 8, 8, 9, 9]
    for counts in by_category.values():
        assert counts["recent"] == 3 * counts["middle"] == 3 * counts["early"]


def test_exact_integer_family_mixes_are_generic_and_survive_any_default_transfer():
    settings = config()
    families = build_assignments(settings, 50)["A"]
    labels = list(CATEGORIES.values())
    selected = {}
    for family in families:
        assert category_options(settings, families, selected, family["family_id"]) == labels
        selected[family["family_id"]] = labels[0]
    settings["generation"]["regular_questions_per_family"] = 8
    settings["generation"]["periods"][0]["weight"] = 2
    families = build_assignments(settings, 7)["A"]
    assert all(Counter(slot["label"] for slot in family["period_slots"]) == {
        "recent": 4, "middle": 2, "early": 2,
    } for family in families)


@pytest.mark.parametrize("regular", [4, 6])
def test_fractional_mix_rejects_category_transfer_that_breaks_coverage(regular):
    settings = config()
    settings["generation"]["regular_questions_per_family"] = regular
    families = build_assignments(settings, 25)["A"]
    family = families[0]
    options = category_options(settings, families, {}, family["family_id"])
    assert family["category"] in options
    assert len(options) < len(CATEGORIES)
    weights = [period["weight"] for period in settings["generation"]["periods"]]
    for candidate in CATEGORIES.values():
        by_category = defaultdict(Counter)
        for assignment in families:
            category = candidate if assignment is family else assignment["category"]
            by_category[category].update(slot["label"] for slot in assignment["period_slots"])
        violations = []
        for mix in by_category.values():
            expected = allocate_counts(sum(mix.values()), weights)
            violations.extend(abs(mix[period["label"]] - target) > 1
                              for period, target in zip(settings["generation"]["periods"], expected, strict=True))
        assert (candidate in options) is not any(violations)


def test_prior_category_selections_change_available_transfers():
    settings = config()
    settings["generation"]["regular_questions_per_family"] = 4
    families = build_assignments(settings, 25)["A"]
    before = deepcopy(families)
    assert category_options(settings, families, {}, "A-F01") == [CATEGORIES["1000"]]
    selected = {"A-F02": CATEGORIES["1000"]}
    assert category_options(settings, families, selected, "A-F01") == list(CATEGORIES.values())
    assert families == before
    assert selected == {"A-F02": CATEGORIES["1000"]}


def test_category_options_excludes_disabled_categories_and_unknown_selections():
    settings = config()
    settings["generation"]["categories"][-1].update(enabled=False, reason="Not in this run's scope")
    families = build_assignments(settings, 6)["A"]
    assert CATEGORIES["6000"] not in category_options(settings, families, {}, "A-F01")
    with pytest.raises(ValueError, match="unknown family"):
        category_options(settings, families, {}, "A-F99")
    with pytest.raises(ValueError, match="known families and enabled categories"):
        category_options(settings, families, {"A-F01": CATEGORIES["6000"]}, "A-F01")


def test_changed_question_count_keeps_exact_totals_and_category_period_balance():
    settings = config()
    settings["generation"]["regular_questions_per_family"] = 6
    families = build_assignments(settings, 25, ("A",))["A"]
    totals, per_category = Counter(), defaultdict(Counter)
    for family in families:
        mix = Counter(slot["label"] for slot in family["period_slots"])
        assert len(family["question_numbers"]) == 7
        assert len(mix) >= 2
        totals.update(mix)
        per_category[family["category"]].update(mix)
    assert totals == {"recent": 90, "middle": 30, "early": 30}
    for counts in per_category.values():
        total = sum(counts.values())
        for period, weight in (("recent", 3), ("middle", 1), ("early", 1)):
            assert abs(counts[period] - total * weight / 5) <= 1.5


def test_source_supported_category_reallocation_uses_configured_weights():
    settings = config()
    settings["generation"]["categories"][-1].update(enabled=False, reason="No applicable national tax found")
    settings["generation"]["categories"][0]["weight"] = 2
    families = build_assignments(settings, 24, ("A",))["A"]
    assert Counter(family["category"] for family in families) == {
        CATEGORIES["1000"]: 8, CATEGORIES["2000"]: 4, CATEGORIES["3000"]: 4,
        CATEGORIES["4000"]: 4, CATEGORIES["5000"]: 4,
    }


def test_nondivisible_family_count_uses_all_slots_without_manual_quotas():
    families = build_assignments(config(), 13, ("A",))["A"]
    assert sorted(Counter(family["category"] for family in families).values()) == [2, 2, 2, 2, 2, 3]


def test_infeasible_period_weights_fail_explicitly():
    settings = config()
    settings["generation"]["periods"][0]["weight"] = 100
    with pytest.raises(ValueError, match="cannot supply"):
        build_assignments(settings, 25)
    settings["generation"]["min_periods_per_family"] = 1
    families = build_assignments(settings, 25)["A"]
    assert Counter(slot["label"] for family in families for slot in family["period_slots"]) == {
        "recent": 123, "middle": 1, "early": 1,
    }


def test_four_regular_questions_can_cover_two_periods_with_exact_ratios():
    settings = config()
    settings["generation"]["regular_questions_per_family"] = 4
    families = build_assignments(settings, 25)["A"]
    assert all(len({slot["label"] for slot in family["period_slots"]}) >= 2 for family in families)
    assert Counter(slot["label"] for family in families for slot in family["period_slots"]) == {
        "recent": 60, "middle": 20, "early": 20,
    }


@pytest.mark.parametrize(("family_count", "regular", "category_weights", "period_weights", "minimum"), [
    (4, 6, [1] * 6, [3, 1, 1], 2),
    (50, 4, [4, 4, 1, 1, 1, 2], [5, 1, 3], 1),
])
def test_rounding_is_not_concentrated_in_the_last_category(
    family_count, regular, category_weights, period_weights, minimum,
):
    settings = config()
    generation = settings["generation"]
    generation.update(regular_questions_per_family=regular, min_periods_per_family=minimum)
    for category, weight in zip(generation["categories"], category_weights, strict=True):
        category["weight"] = weight
    for period, weight in zip(generation["periods"], period_weights, strict=True):
        period["weight"] = weight
    families = build_assignments(settings, family_count)["A"]
    totals, categories = Counter(), defaultdict(Counter)
    for family in families:
        mix = Counter(slot["label"] for slot in family["period_slots"])
        assert len(mix) >= minimum
        totals.update(mix)
        categories[family["category"]].update(mix)
    labels = [period["label"] for period in generation["periods"]]
    assert [totals[label] for label in labels] == allocate_counts(family_count * regular, period_weights)
    for counts in categories.values():
        expected = allocate_counts(sum(counts.values()), period_weights)
        assert all(abs(counts[label] - target) <= 1 for label, target in zip(labels, expected, strict=True))


@pytest.mark.parametrize(("total", "weights", "expected"), [
    (25, [1] * 6, [5, 4, 4, 4, 4, 4]),
    (125, [75, 25, 25], [75, 25, 25]),
    (7, [0.5, 0.5], [4, 3]),
    (0, [1, 2], [0, 0]),
])
def test_integer_apportionment(total, weights, expected):
    assert allocate_counts(total, weights) == expected


@pytest.mark.parametrize("weights", [[], [0], [-1], [True], ["1"], [float("nan")], [float("inf")]])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        allocate_counts(5, weights)


@pytest.mark.parametrize(("field", "value"), [
    ("families", 0), ("families", True),
    ("regular_questions_per_family", 0), ("regular_questions_per_family", 1),
    ("missing_information_per_family", -1), ("missing_information_per_family", "1"),
    ("min_periods_per_family", True),
])
def test_invalid_generation_counts(field, value):
    settings = config()
    settings["generation"][field] = value
    with pytest.raises(ValueError):
        validate_generation(settings)


@pytest.mark.parametrize("change", ["overlap", "duplicate_period", "duplicate_category", "unknown_category",
                                   "disabled_without_reason", "all_disabled", "reversed_dates"])
def test_invalid_coverage_definitions(change):
    settings = config()
    generation = settings["generation"]
    if change == "overlap":
        generation["periods"][1]["end"] = 2020
    elif change == "duplicate_period":
        generation["periods"][1]["label"] = "recent"
    elif change == "duplicate_category":
        generation["categories"][1] = dict(generation["categories"][0])
    elif change == "unknown_category":
        generation["categories"][0]["code"] = "7000"
    elif change == "disabled_without_reason":
        generation["categories"][0]["enabled"] = False
    elif change == "all_disabled":
        for category in generation["categories"]:
            category.update(enabled=False, reason="Unavailable")
    else:
        generation["periods"][0]["end"] = 2019
    with pytest.raises(ValueError):
        validate_generation(settings)
