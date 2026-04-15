from langgraph.graph import StateGraph, END
from langgraph.types import Command, Send
from src.state import ResearchState
from src.agents import (
    planner_node,
    writer_node,
    critic_node,
    supervisor_node,
    slide_planner_node,
    slide_writer_node,
)
from src.util import MAX_REVISIONS, MAX_REPLANS


class ResearchGraph:
    def __init__(self, slides_mode: bool = False):
        self._graph = self._build(slides_mode)

    def _build(self, slides_mode: bool):
        g = StateGraph(ResearchState)

        # ── Synthesis pipeline ────────────────────────────────────────────
        g.add_node('planner',    self._planner_with_guard)
        g.add_node('writer',     self._writer_with_guard)
        g.add_node('critic',     critic_node)
        g.add_node('supervisor', supervisor_node)

        g.add_edge('planner', 'writer')
        g.add_edge('writer',  'critic')
        g.add_edge('critic',  'supervisor')
        # supervisor routes via Command — no conditional_edges needed

        # ── Slide-generation pipeline ─────────────────────────────────────
        # slide_planner returns a list[Send] which LangGraph fans out to
        # as many slide_writer instances as the plan requires.  Each
        # slide_writer instance then runs independently in parallel.
        g.add_node('slide_planner', slide_planner_node)
        g.add_node('slide_writer',  slide_writer_node)

        if slides_mode:
            g.set_entry_point('slide_planner')
        else:
            g.set_entry_point('planner')

        return g.compile()

    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------

    def _planner_with_guard(self, state: ResearchState) -> Command:
        replan_count = state.get('replan_count', 0)
        if replan_count >= MAX_REPLANS:
            msg = f'[planner] failure: MAX_REPLANS ({MAX_REPLANS}) reached'
            return Command(
                update={'errors': [{'node': 'planner', 'error': 'MAX_REPLANS reached'}], 'messages': [msg]},
                goto=END,
            )
        return planner_node(state)

    def _writer_with_guard(self, state: ResearchState) -> Command:
        revision_count = state.get('revision_count', 0)
        if revision_count >= MAX_REVISIONS:
            msg = f'[writer] failure: MAX_REVISIONS ({MAX_REVISIONS}) reached'
            return Command(
                update={'errors': [{'node': 'writer', 'error': 'MAX_REVISIONS reached'}], 'messages': [msg]},
                goto=END,
            )
        return writer_node(state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def invoke(self, initial_state: dict, config: dict = None):
        return self._graph.invoke(initial_state, config=config)

    def stream(self, initial_state: dict, config: dict = None, stream_mode: str = "values"):
        return self._graph.stream(initial_state, config=config, stream_mode=stream_mode)

    def invoke_slides(self, initial_state: dict, config: dict = None):
        """
        Run only the slide-generation pipeline (slide_planner → slide_writer).
        Requires the state to contain: doc_id, max_slides, session_id.
        """
        return self._graph.invoke(
            initial_state,
            config=config,
        )


def build_graph(slides_mode: bool = False) -> ResearchGraph:
    return ResearchGraph(slides_mode=slides_mode)
