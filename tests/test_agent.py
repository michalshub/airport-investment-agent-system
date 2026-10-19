"""Answer verification: numbers in the LLM answer must match tool outputs within displayed rounding."""

from src.agent import find_unverified_numbers, tool_numbers

CALLS = [{"result": {"passengers_ttm": 36039571, "delay_rate_15_pct": 19.32, "weight": 0.35,
                     "period": "2025-08 to 2026-07", "share": 41.35}}]


def test_rounded_and_scaled_values_are_verified():
    known = tool_numbers(CALLS)
    answer = "LAX handled 36.0M (36,039,571) passengers; delay rate 19.3% (about 19%); weight 35%; share 41.35%."
    assert find_unverified_numbers(answer, known) == []


def test_invented_or_miscalculated_numbers_are_flagged():
    known = tool_numbers(CALLS)
    assert find_unverified_numbers("Delay rate is 23.4% and traffic 123 thousand.", known) == ["23.4%", "123"]


def test_substring_matches_do_not_count():
    assert find_unverified_numbers("Cancellation rate is 395.", tool_numbers(CALLS)) == ["395"]


def test_years_ranks_and_small_counts_are_ignored():
    assert find_unverified_numbers("In 2026 the top 3 airports over 12 months", []) == []
