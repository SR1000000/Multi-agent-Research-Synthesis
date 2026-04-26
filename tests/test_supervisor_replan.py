"""Unit tests for supervisor replan routing and ``force_replan_at_max_cycles``.

These tests mock the research database and backup helper so no SQLite file is required.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from langgraph.graph import END

from src.agents.supervisor import SupervisorAgent
from src.state import (
    MAX_CYCLES,
    PresentationPlan,
    SlideBlueprint,
    SlideGroup,
    make_initial_review_state,
)


def _minimal_presentation_plan() -> PresentationPlan:
    bp = SlideBlueprint(
        slide_number=1,
        working_title="t",
        narrative_role="hook",
        intent="i",
        source_chunk_ids=["c1"],
    )
    return PresentationPlan(
        title="T",
        subtitle="S",
        thesis="th",
        target_audience="all",
        estimated_duration_minutes=5,
        narrative_arc_summary="arc",
        slide_groups=[SlideGroup(slide_blueprints=[bp], rationale="r")],
        reasoning="plan",
    )


def _patch_db_and_backup():
    """Return (ResearchDatabase patch target, mock instance) for context manager usage."""
    mock_db = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_db
    mock_ctx.__exit__.return_value = None
    return mock_ctx, mock_db


class TestSupervisorReplan(unittest.TestCase):
    """Focused tests for max-cycle forced replan and dispatch counter carry-forward."""

    @patch("src.agents.supervisor.backup_replan_debug_snapshot")
    @patch("src.agents.supervisor.ResearchDatabase")
    def test_path_a_plan1_at_cap_forced_replan(
        self, mock_db_class: MagicMock, mock_backup: MagicMock
    ) -> None:
        """No critic batch yet, at cap, rewrites pending → forced replan; counter carried."""
        mock_ctx, mock_db = _patch_db_and_backup()
        mock_db_class.return_value = mock_ctx
        call_order: list[str] = []

        def _backup(*_a, **_k) -> None:
            call_order.append("backup")

        def _save(*_a, **_k) -> None:
            call_order.append("save")

        mock_backup.side_effect = _backup
        mock_db.save_review_event.side_effect = _save

        review = make_initial_review_state(max_cycles=MAX_CYCLES)
        review["cycle_number"] = MAX_CYCLES
        review["last_rewrite_assignment_ids"] = ["rewrite-x"]
        review["dispatch_counter"] = 42

        state = {
            "session_id": "sess",
            "query": "q",
            "plan_number": 1,
            "force_replan_at_max_cycles": True,
            "presentation_plan": _minimal_presentation_plan(),
            "review": review,
            "critic_results": [],
            "review_summaries": [],
        }
        cmd = SupervisorAgent().run(state)

        self.assertEqual(call_order, ["backup", "save"])
        self.assertEqual(cmd.goto, "planner")
        self.assertEqual(cmd.update["plan_number"], 2)
        self.assertIsNone(cmd.update["presentation_plan"])
        self.assertEqual(cmd.update["review"]["dispatch_counter"], 42)
        self.assertEqual(cmd.update["review"]["phase"], "initial_write")

    @patch("src.agents.supervisor.backup_replan_debug_snapshot")
    @patch("src.agents.supervisor.ResearchDatabase")
    def test_path_a_plan2_at_cap_forced_increments_to_3(
        self, mock_db_class: MagicMock, mock_backup: MagicMock
    ) -> None:
        mock_ctx, mock_db = _patch_db_and_backup()
        mock_db_class.return_value = mock_ctx

        review = make_initial_review_state(max_cycles=MAX_CYCLES)
        review["cycle_number"] = MAX_CYCLES
        review["last_rewrite_assignment_ids"] = ["r1"]
        review["dispatch_counter"] = 3

        state = {
            "session_id": "sess",
            "query": "q",
            "plan_number": 2,
            "force_replan_at_max_cycles": True,
            "presentation_plan": _minimal_presentation_plan(),
            "review": review,
            "critic_results": [],
        }
        cmd = SupervisorAgent().run(state)
        self.assertEqual(cmd.goto, "planner")
        self.assertEqual(cmd.update["plan_number"], 3)

    @patch("src.agents.supervisor.ResearchDatabase")
    def test_path_a_plan3_at_cap_accepts_not_forced(
        self, mock_db_class: MagicMock
    ) -> None:
        """After ``MAX_REPLANS`` forced replans, further cap hits use normal accept."""
        mock_ctx, mock_db = _patch_db_and_backup()
        mock_db_class.return_value = mock_ctx

        review = make_initial_review_state(max_cycles=MAX_CYCLES)
        review["cycle_number"] = MAX_CYCLES
        review["last_rewrite_assignment_ids"] = ["r1"]
        review["last_issue_counts"] = {"critical": 0, "major": 0, "minor": 1}

        state = {
            "session_id": "sess",
            "query": "q",
            "plan_number": 3,
            "force_replan_at_max_cycles": True,
            "presentation_plan": _minimal_presentation_plan(),
            "review": review,
            "critic_results": [],
        }
        cmd = SupervisorAgent().run(state)
        self.assertEqual(cmd.goto, END)
        self.assertTrue(cmd.update["review"].get("export_ready"))
        mock_db.save_review_event.assert_not_called()

    @patch("src.agents.supervisor.ResearchDatabase")
    def test_below_cap_flag_does_not_force_replan(self, mock_db_class: MagicMock) -> None:
        mock_ctx, mock_db = _patch_db_and_backup()
        mock_db_class.return_value = mock_ctx
        mock_db.list_review_events.return_value = []

        review = make_initial_review_state(max_cycles=MAX_CYCLES)
        review["cycle_number"] = 0
        review["phase"] = "awaiting_supervisor"

        state = {
            "session_id": "sess",
            "query": "q",
            "plan_number": 1,
            "force_replan_at_max_cycles": True,
            "presentation_plan": _minimal_presentation_plan(),
            "review": review,
            "critic_results": [],
        }
        cmd = SupervisorAgent().run(state)
        self.assertEqual(cmd.goto, "plan_executor")
        self.assertEqual(cmd.update["review"]["cycle_number"], 1)

    @patch("src.agents.supervisor.ResearchDatabase")
    def test_at_cap_without_flag_accepts_path_a(
        self, mock_db_class: MagicMock
    ) -> None:
        mock_ctx, mock_db = _patch_db_and_backup()
        mock_db_class.return_value = mock_ctx

        review = make_initial_review_state(max_cycles=MAX_CYCLES)
        review["cycle_number"] = MAX_CYCLES
        review["last_rewrite_assignment_ids"] = ["r1"]

        state = {
            "session_id": "sess",
            "query": "q",
            "plan_number": 1,
            "force_replan_at_max_cycles": False,
            "presentation_plan": _minimal_presentation_plan(),
            "review": review,
            "critic_results": [],
        }
        cmd = SupervisorAgent().run(state)
        self.assertEqual(cmd.goto, END)
        mock_db.save_review_event.assert_not_called()

    @patch("src.agents.supervisor.SupervisorAgent._call", autospec=True)
    @patch("src.agents.supervisor.backup_replan_debug_snapshot")
    @patch("src.agents.supervisor.ResearchDatabase")
    def test_path_c_at_cap_forced_before_llm(
        self,
        mock_db_class: MagicMock,
        mock_backup: MagicMock,
        mock_llm: MagicMock,
    ) -> None:
        """Critic batch present, not post-rewrite: ``at_cap_forced`` replans without LLM (path C)."""
        mock_ctx, _mock_db = _patch_db_and_backup()
        mock_db_class.return_value = mock_ctx
        mock_ctx_db = _mock_db
        mock_ctx_db.list_review_events.return_value = []

        did = "dispatch-99"
        review = make_initial_review_state(max_cycles=MAX_CYCLES)
        review["cycle_number"] = MAX_CYCLES
        review["last_critic_dispatch_id"] = did
        review["last_rewrite_assignment_ids"] = []
        review["phase"] = "awaiting_supervisor"

        critic = {
            "dispatch_id": did,
            "assignment_id": "a1",
            "cycle_number": MAX_CYCLES,
            "check_type": "grounding_consistency",
            "scope_type": "group",
            "scope_id": "0",
            "group_idx": 0,
            "target_slide_numbers": [1],
            "actionable": True,
            "rewrite_instructions": "fix",
            "summary": "issues",
            "issues": [
                {
                    "issue_code": "X",
                    "severity": "major",
                    "issue_type": "grounding",
                    "location": "s1",
                    "fingerprint": "fp1",
                    "affected_slide_numbers": [1],
                    "rewrite_instruction": "r",
                }
            ],
        }
        state = {
            "session_id": "sess",
            "query": "q",
            "plan_number": 1,
            "force_replan_at_max_cycles": True,
            "presentation_plan": _minimal_presentation_plan(),
            "review": review,
            "critic_results": [critic],
        }
        cmd = SupervisorAgent().run(state)
        self.assertEqual(cmd.goto, "planner")
        mock_llm.assert_not_called()
        mock_backup.assert_called_once()

    @patch("src.agents.supervisor.backup_replan_debug_snapshot")
    @patch("src.agents.supervisor.ResearchDatabase")
    def test_path_b_post_rewrite_at_cap_forced(
        self, mock_db_class: MagicMock, mock_backup: MagicMock
    ) -> None:
        """Post-rewrite with critic results: at cap + flag → replan (path B)."""
        mock_ctx, mock_db = _patch_db_and_backup()
        mock_db_class.return_value = mock_ctx
        mock_db.list_review_events.return_value = []

        did = "dispatch-batch-1"
        review = make_initial_review_state(max_cycles=MAX_CYCLES)
        review["cycle_number"] = MAX_CYCLES
        review["last_critic_dispatch_id"] = did
        review["last_rewrite_assignment_ids"] = ["rw-1"]
        review["dispatch_counter"] = 5

        critic = {
            "dispatch_id": did,
            "assignment_id": "a1",
            "cycle_number": MAX_CYCLES,
            "check_type": "grounding_consistency",
            "scope_type": "group",
            "scope_id": "0",
            "group_idx": 0,
            "target_slide_numbers": [1],
            "actionable": False,
            "rewrite_instructions": "",
            "summary": "ok",
            "issues": [],
        }
        state = {
            "session_id": "sess",
            "query": "q",
            "plan_number": 1,
            "force_replan_at_max_cycles": True,
            "presentation_plan": _minimal_presentation_plan(),
            "review": review,
            "critic_results": [critic],
        }
        cmd = SupervisorAgent().run(state)
        self.assertEqual(cmd.goto, "planner")
        self.assertEqual(cmd.update["plan_number"], 2)
        mock_backup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
