"""System role and user-prompt text for `SupervisorAgent` (accept / revise / replan decision)."""

from __future__ import annotations

SUPERVISOR_ROLE = """
You are the Slide Deck Supervisor. Your pipeline automatically handles standard routing.
Your specific job is to handle two subjective edge cases based on the reviewer's feedback and the revision history:

1. **Early Replanning (replan)**: If critical or structural issues repeat cycle after cycle without converging, the plan itself might be flawed. You may choose to preemptively `replan` before the hard cycle cap is reached.
2. **Accepting with Minor Flaws (accept)**: If the ONLY remaining actionable issues are "minor" and they are persistent (count >= 2), you must decide if they are worth another rewrite cycle. If the minor issues seem overly pedantic or stylistic, choose `accept` to finish the presentation. Otherwise, choose `revise`.

If neither edge case applies, simply default to `revise`.

Your reasoning should be concise. Explain your evaluation of the recurrence and whether it warrants an early replan, an accept override, or standard revision.
"""


def build_supervisor_user_prompt(
    *,
    query: str,
    cycle_number: int,
    max_cycles: int,
    severity_counts: dict[str, int],
    summaries: str,
    recurring_lines: str,
) -> str:
    """User message for the supervisor LLM call (edge-case routing on critic feedback)."""
    return "\n".join(
        [
            f"Query:\n{query}",
            f"Cycle number: {cycle_number} (Max: {max_cycles})",
            f"Severity counts: {severity_counts}",
            "Critic results:",
            summaries,
            "Recurring issue fingerprints (count >= 2):",
            recurring_lines,
            "",
            "Based on the edge cases in your system instructions, decide whether to accept, revise, or replan.",
        ]
    )
