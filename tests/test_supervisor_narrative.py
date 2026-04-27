"""Unit tests for narrative critic dispatch and deck-scoped rewrite splitting (no LLM / DB)."""

from __future__ import annotations

import unittest

from src.agents.supervisor import (
    _build_critic_assignments,
    _build_rewrite_assignments,
)
from src.state import (
    SlideBlueprint,
    SlideGroup,
    PresentationPlan,
)


def _bp(n: int) -> SlideBlueprint:
    return SlideBlueprint(
        slide_number=n,
        working_title=f"t{n}",
        narrative_role="hook",
        intent="i",
        source_chunk_ids=["c1"],
    )


def _minimal_plan_one_group() -> PresentationPlan:
    return PresentationPlan(
        title="T",
        subtitle="S",
        thesis="th",
        target_audience="all",
        estimated_duration_minutes=5,
        narrative_arc_summary="arc",
        slide_groups=[SlideGroup(slide_blueprints=[_bp(1)], rationale="r")],
        reasoning="plan",
    )


def _plan_two_groups_12_34() -> PresentationPlan:
    return PresentationPlan(
        title="T",
        subtitle="S",
        thesis="th",
        target_audience="all",
        estimated_duration_minutes=5,
        narrative_arc_summary="arc",
        slide_groups=[
            SlideGroup(slide_blueprints=[_bp(1), _bp(2)], rationale="g0"),
            SlideGroup(slide_blueprints=[_bp(3), _bp(4)], rationale="g1"),
        ],
        reasoning="plan",
    )


class TestNarrativeCriticWiring(unittest.TestCase):
    def test_critic_assignments_include_grounding_and_narrative(self) -> None:
        plan = _minimal_plan_one_group()
        a = _build_critic_assignments(plan=plan, cycle_number=1)
        self.assertEqual(len(a), 2)
        self.assertEqual(a[0]["check_type"], "grounding_consistency")
        self.assertEqual(a[0]["group_idx"], 0)
        self.assertEqual(a[1]["check_type"], "narrative_coherence")
        self.assertEqual(a[1]["group_idx"], -1)
        self.assertEqual(a[1]["scope_type"], "deck")
        self.assertEqual(a[1]["assignment_id"], "critic-c1-narrative")
        self.assertEqual(a[1]["chunk_ids"], [])

    def test_deck_result_splits_and_clips_slide_numbers(self) -> None:
        plan = _plan_two_groups_12_34()
        deck = {
            "dispatch_id": "critic-1",
            "assignment_id": "critic-c1-narrative",
            "cycle_number": 1,
            "check_type": "narrative_coherence",
            "scope_type": "deck",
            "scope_id": "deck",
            "group_idx": -1,
            "target_slide_numbers": [1, 2, 3, 4],
            "actionable": True,
            "rewrite_instructions": "ignored",
            "summary": "s",
            "issues": [
                {
                    "issue_code": "I1",
                    "severity": "major",
                    "issue_type": "flow",
                    "location": "l",
                    "fingerprint": "f1",
                    "affected_slide_numbers": [2, 3, 4],
                    "rewrite_instruction": "fix transition",
                }
            ],
        }
        rewrites = _build_rewrite_assignments(
            plan=plan, results=[deck], cycle_number=1
        )
        self.assertEqual(len(rewrites), 2)
        r0 = next(r for r in rewrites if r["group_idx"] == 0)
        r1 = next(r for r in rewrites if r["group_idx"] == 1)
        self.assertEqual(r0["target_slide_numbers"], [2])
        self.assertIn("Slide(s) 2", r0["rewrite_instructions"])
        self.assertEqual(r1["target_slide_numbers"], [3, 4])
        self.assertIn("Slide(s) 3, 4", r1["rewrite_instructions"])

    def test_deck_issue_non_overlapping_group_gets_no_rewrite(self) -> None:
        plan = _plan_two_groups_12_34()
        deck = {
            "dispatch_id": "critic-1",
            "assignment_id": "critic-c1-narrative",
            "cycle_number": 1,
            "check_type": "narrative_coherence",
            "scope_type": "deck",
            "scope_id": "deck",
            "group_idx": -1,
            "target_slide_numbers": [1, 2, 3, 4],
            "actionable": True,
            "rewrite_instructions": "",
            "summary": "s",
            "issues": [
                {
                    "issue_code": "I1",
                    "severity": "major",
                    "issue_type": "flow",
                    "location": "l",
                    "fingerprint": "f1",
                    "affected_slide_numbers": [5, 6],
                    "rewrite_instruction": "n/a",
                }
            ],
        }
        rewrites = _build_rewrite_assignments(
            plan=plan, results=[deck], cycle_number=1
        )
        self.assertEqual(rewrites, [])

    def test_deck_skips_empty_affected_slide_numbers(self) -> None:
        plan = _plan_two_groups_12_34()
        deck = {
            "dispatch_id": "critic-1",
            "assignment_id": "critic-c1-narrative",
            "cycle_number": 1,
            "check_type": "narrative_coherence",
            "scope_type": "deck",
            "scope_id": "deck",
            "group_idx": -1,
            "target_slide_numbers": [1, 2, 3, 4],
            "actionable": True,
            "rewrite_instructions": "",
            "summary": "s",
            "issues": [
                {
                    "issue_code": "I1",
                    "severity": "major",
                    "issue_type": "flow",
                    "location": "l",
                    "fingerprint": "f1",
                    "affected_slide_numbers": [],
                    "rewrite_instruction": "x",
                }
            ],
        }
        rewrites = _build_rewrite_assignments(
            plan=plan, results=[deck], cycle_number=1
        )
        self.assertEqual(rewrites, [])

    def test_group_scoped_grounding_rewrite_unchanged_shape(self) -> None:
        plan = _minimal_plan_one_group()
        g = {
            "dispatch_id": "critic-1",
            "assignment_id": "critic-c1-g0",
            "cycle_number": 1,
            "check_type": "grounding_consistency",
            "scope_type": "group",
            "scope_id": "0",
            "group_idx": 0,
            "target_slide_numbers": [1],
            "actionable": True,
            "rewrite_instructions": "fix it",
            "summary": "s",
            "issues": [
                {
                    "issue_code": "I1",
                    "severity": "major",
                    "issue_type": "g",
                    "location": "l",
                    "fingerprint": "f1",
                    "affected_slide_numbers": [1],
                    "rewrite_instruction": "cite",
                }
            ],
        }
        rewrites = _build_rewrite_assignments(
            plan=plan, results=[g], cycle_number=1
        )
        self.assertEqual(len(rewrites), 1)
        self.assertEqual(rewrites[0]["assignment_id"], "rewrite-critic-c1-g0")
        self.assertEqual(rewrites[0]["group_idx"], 0)
        self.assertEqual(rewrites[0]["target_slide_numbers"], [1])
        self.assertEqual(rewrites[0]["rewrite_instructions"], "fix it")


if __name__ == "__main__":
    unittest.main()
