"""Small, deterministic coverage allocations from editable country settings."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from typing import Sequence

from .schema import CATEGORIES


def _positive_int(value: object, name: str, *, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


def _weight(value: object) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Coverage weights must be positive finite numbers")
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("Coverage weights must be positive finite numbers") from exc
    if result <= 0:
        raise ValueError("Coverage weights must be positive finite numbers")
    return result


def allocate_counts(total: int, weights: Sequence[int | float]) -> list[int]:
    """Largest-remainder apportionment; input order breaks equal remainders."""
    _positive_int(total, "total", minimum=0)
    if not weights:
        raise ValueError("At least one coverage weight is required")
    values = [_weight(value) for value in weights]
    quotas = [total * value / sum(values) for value in values]
    counts = [int(quota) for quota in quotas]
    order = sorted(range(len(values)), key=lambda index: (quotas[index] - counts[index], -index),
                   reverse=True)
    for index in order[:total - sum(counts)]:
        counts[index] += 1
    return counts


def validate_generation(config: dict) -> None:
    """Validate counts and weights; legal category applicability belongs to planning."""
    generation = config["generation"]
    if "blocked_reason" in generation and (not isinstance(generation["blocked_reason"], str)
                                           or not generation["blocked_reason"].strip()):
        raise ValueError("generation.blocked_reason must be a nonblank explanation")
    _positive_int(generation["families"], "families")
    regular = generation.get("regular_questions_per_family", 5)
    _positive_int(regular, "regular_questions_per_family")
    _positive_int(generation.get("missing_information_per_family", 1), "missing_information_per_family",
                  minimum=0)
    _positive_int(generation.get("min_periods_per_family", 2), "min_periods_per_family")
    categories, periods = generation["categories"], generation["periods"]
    if not isinstance(categories, list) or not categories or not isinstance(periods, list) or not periods:
        raise ValueError("At least one category and one year period are required")
    codes, labels, active = set(), set(), 0
    for category in categories:
        code, label = category["code"], category["label"]
        if code not in CATEGORIES or CATEGORIES[code] != label:
            raise ValueError("Categories must use an OECD code and its configured taxonomy label")
        if code in codes or label in labels:
            raise ValueError("Category codes and labels must be unique")
        codes.add(code)
        labels.add(label)
        if type(category.get("enabled", True)) is not bool:
            raise ValueError("Category enabled must be a boolean")
        if not category.get("enabled", True):
            if not isinstance(category.get("reason"), str) or not category["reason"].strip():
                raise ValueError("A disabled category requires a reason")
        else:
            active += 1
        _weight(category.get("weight", 1))
    if not active:
        raise ValueError("At least one category must remain enabled")
    labels, intervals = set(), []
    for period in periods:
        label = period["label"]
        if not isinstance(label, str) or not label.strip() or label in labels:
            raise ValueError("Year period labels must be nonblank and unique")
        labels.add(label)
        for key in ("start", "end"):
            _positive_int(period[key], f"Period {key}")
        if not 1 <= period["start"] <= period["end"] <= 9999:
            raise ValueError("Invalid year period boundaries")
        if any(period["start"] <= end and start <= period["end"] for start, end in intervals):
            raise ValueError("Year periods must not overlap")
        intervals.append((period["start"], period["end"]))
        _weight(period.get("weight", 1))
    if generation.get("min_periods_per_family", 2) > min(regular, len(periods)):
        raise ValueError("min_periods_per_family cannot exceed regular_questions_per_family or the number "
                         "of configured periods")


def _category_slots(categories: list[dict], count: int) -> list[str]:
    quotas = allocate_counts(count, [category.get("weight", 1) for category in categories])
    return [category["label"] for turn in range(max(quotas))
            for category, quota in zip(categories, quotas, strict=True) if turn < quota]


def _period_counts(generation: dict, categories: list[str]) -> list[list[int]]:
    """Keep aggregate targets exact, family periods diverse and category mixes close."""
    regular = generation.get("regular_questions_per_family", 5)
    periods, families = generation["periods"], len(categories)
    weights = [_weight(period.get("weight", 1)) for period in periods]
    per_family = [regular * weight / sum(weights) for weight in weights]
    if all(quota.denominator == 1 for quota in per_family):
        # An exact family mix remains balanced under any category transfer.
        return [[int(quota) for quota in per_family] for _ in categories]
    targets = allocate_counts(families * regular, [period.get("weight", 1) for period in periods])
    minimum = generation.get("min_periods_per_family", 2)
    capacities = [min(target, families) for target in targets]
    seed_total = families * minimum
    if sum(capacities) < seed_total:
        raise ValueError("Period weights and regular_questions_per_family cannot supply "
                         "min_periods_per_family distinct periods in every family. Increase the question "
                         "count, rebalance period weights, or reduce min_periods_per_family.")
    # Choose enough distinct-period appearances, never more than one of a
    # period per family. Larger weights receive more of these appearances.
    seeds = [0] * len(periods)
    for position in range(seed_total):
        chosen = max((p for p, capacity in enumerate(capacities) if seeds[p] < capacity), key=lambda p: (
            Fraction((position + 1) * targets[p], families * regular) - seeds[p], -p))
        seeds[chosen] += 1
    remaining_seeds = list(seeds)
    result = [[0] * len(periods) for _ in categories]
    category_seeded = {category: Counter() for category in categories}
    for index, category in enumerate(categories):
        # Highest remaining counts first constructs the distinct appearances
        # without leaving a late family with only one period available.
        selected = sorted(range(len(periods)), key=lambda p: (
            remaining_seeds[p], -category_seeded[category][p], -p), reverse=True)[:minimum]
        for period in selected:
            result[index][period] += 1
            remaining_seeds[period] -= 1
            category_seeded[category][period] += 1
    remaining = [target - seed for target, seed in zip(targets, seeds, strict=True)]
    residual = list(remaining)
    extra_total = sum(residual)
    if not extra_total:
        return _balance_categories(result, categories, generation)
    category_sizes = Counter(categories)
    desired = {category: [max(Fraction(0), Fraction(size * target, families) - category_seeded[category][p])
                          for p, target in enumerate(targets)] for category, size in category_sizes.items()}
    category_used = {category: Counter() for category in categories}
    global_used: Counter = Counter()
    for index, category in enumerate(categories):
        for _ in range(regular - minimum):
            own = category_used[category]
            own_next, global_next = sum(own.values()) + 1, sum(global_used.values()) + 1
            chosen = max((period for period, left in enumerate(remaining) if left), key=lambda period: (
                own_next * desired[category][period] / sum(desired[category]) - own[period],
                Fraction(global_next * residual[period], extra_total) - global_used[period], -period))
            result[index][chosen] += 1
            own[chosen] += 1
            global_used[chosen] += 1
            remaining[chosen] -= 1
    return _balance_categories(result, categories, generation)


def _balance_categories(counts: list[list[int]], categories: list[str], generation: dict) -> list[list[int]]:
    """Exchange period slots to avoid concentrating rounding error in late categories."""
    weights = [period.get("weight", 1) for period in generation["periods"]]
    totals = {category: [sum(row[p] for row, own in zip(counts, categories, strict=True) if own == category)
                         for p in range(len(weights))] for category in dict.fromkeys(categories)}
    desired = {category: allocate_counts(sum(row), weights) for category, row in totals.items()}
    minimum = generation.get("min_periods_per_family", 2)

    def score(category: str, row: list[int]) -> tuple[int, int]:
        difference = [abs(actual - target) for actual, target in zip(row, desired[category], strict=True)]
        return sum(max(0, value - 1) for value in difference), sum(value * value for value in difference)

    while any(score(category, row)[0] for category, row in totals.items()):
        changed = False
        for left, left_category in enumerate(categories):
            for right in range(left + 1, len(categories)):
                right_category = categories[right]
                if left_category == right_category:
                    continue
                before = tuple(sum(values) for values in zip(score(left_category, totals[left_category]),
                                                            score(right_category, totals[right_category]), strict=True))
                for old in range(len(weights)):
                    for new in range(len(weights)):
                        if old == new or not counts[left][old] or not counts[right][new]:
                            continue
                        rows = [list(counts[left]), list(counts[right])]
                        totals_after = [list(totals[left_category]), list(totals[right_category])]
                        for row, total, remove, add in zip(rows, totals_after, (old, new), (new, old), strict=True):
                            row[remove] -= 1
                            row[add] += 1
                            total[remove] -= 1
                            total[add] += 1
                        if any(sum(value > 0 for value in row) < minimum for row in rows):
                            continue
                        after = tuple(sum(values) for values in zip(score(left_category, totals_after[0]),
                                                                   score(right_category, totals_after[1]), strict=True))
                        if after < before:
                            counts[left], counts[right] = rows
                            totals[left_category], totals[right_category] = totals_after
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
            if changed:
                break
        if not changed:
            raise ValueError("The requested period weights and family counts cannot be balanced across categories "
                             "with the configured minimum periods; adjust these settings before generation.")
    return counts


def build_assignments(config: dict, target: int, creators: Sequence[str] = ("A",)) -> dict[str, list[dict]]:
    """Assign family IDs, regular/missing rows and per-question period constraints.

    Missing-information rows are additional and excluded from temporal quotas.
    Category weights may be adjusted by the portfolio planner before this call.
    """
    validate_generation(config)
    _positive_int(target, "target")
    if not creators or len(set(creators)) != len(creators):
        raise ValueError("Creator identifiers must be nonempty and unique")
    generation = config["generation"]
    regular = generation.get("regular_questions_per_family", 5)
    missing = generation.get("missing_information_per_family", 1)
    category_slots = _category_slots([category for category in generation["categories"]
                                      if category.get("enabled", True)], target)
    period_counts = _period_counts(generation, category_slots)
    assignments = {}
    for creator_index, creator in enumerate(creators):
        families = []
        for index, (category, counts) in enumerate(zip(category_slots, period_counts, strict=True)):
            first = (creator_index * target + index) * (regular + missing) + 1
            regular_numbers = list(range(first, first + regular))
            missing_numbers = list(range(first + regular, first + regular + missing))
            # First visit every period, then fill extra slots. The planner can
            # build the temporal contrast immediately after the original case.
            ordered_periods = [period for turn in range(max(counts))
                               for period, count in zip(generation["periods"], counts, strict=True)
                               if turn < count]
            slots = [dict(question_no=number, **{key: period[key] for key in ("label", "start", "end")})
                     for number, period in zip(regular_numbers, ordered_periods, strict=True)]
            families.append({"family_id": f"{creator}-F{index + 1:02d}", "question_numbers": regular_numbers + missing_numbers,
                             "regular_question_numbers": regular_numbers, "missing_question_numbers": missing_numbers,
                             "category": category, "period_slots": slots})
        assignments[creator] = families
    return assignments


def category_options(
    config: dict, assignments: Sequence[dict], selected_categories: dict[str, str], family_id: str,
) -> list[str]:
    """Allow a category transfer only when all planned category/year mixes stay balanced.

    Already planned families use their selected category; the rest retain their
    preferred assignment. Missing-information rows have no period slots here.
    """
    generation = config["generation"]
    enabled = [category["label"] for category in generation["categories"] if category.get("enabled", True)]
    identities = {assignment["family_id"] for assignment in assignments}
    if family_id not in identities:
        raise ValueError("Category options requested for an unknown family")
    if set(selected_categories) - identities or set(selected_categories.values()) - set(enabled):
        raise ValueError("Selected categories must refer to known families and enabled categories")
    periods = generation["periods"]
    labels = [period["label"] for period in periods]
    weights = [period.get("weight", 1) for period in periods]
    mixes = {assignment["family_id"]: Counter(slot["label"] for slot in assignment["period_slots"])
             for assignment in assignments}
    if any(set(mix) - set(labels) for mix in mixes.values()):
        raise ValueError("An assignment refers to an unknown year period")
    options = []
    for candidate in enabled:
        totals: dict[str, Counter] = {}
        for assignment in assignments:
            identity = assignment["family_id"]
            category = candidate if identity == family_id else selected_categories.get(identity, assignment["category"])
            totals.setdefault(category, Counter()).update(mixes[identity])
        if all(all(abs(mix[label] - target) <= 1 for label, target in zip(
            labels, allocate_counts(sum(mix.values()), weights), strict=True)) for mix in totals.values()):
            options.append(candidate)
    return options
