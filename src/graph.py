from langgraph.graph import StateGraph, END
from src.state import ResearchState
from src.agents import (
    planner_node,
    plan_executor_node,
    slide_writer_node,
    critic_node,
    supervisor_node,
)
from src.tools.registry import resolve_agent_tools


class ResearchGraph:
    def __init__(
        self,
        tool_registry: dict | None = None,
        agent_tool_allowlist: dict[str, list[str]] | None = None,
    ):
        self._tool_registry = tool_registry or {}
        self._agent_tool_allowlist = agent_tool_allowlist or {}
        self._agent_tools = {
            agent_name: resolve_agent_tools(self._tool_registry, allowed_tools)
            for agent_name, allowed_tools in self._agent_tool_allowlist.items()
        }
        self._graph = self._build()

    def _build(self):
        g = StateGraph(ResearchState)

        g.add_node("planner",      self._planner_node)
        g.add_node("plan_executor", plan_executor_node)
        g.add_node("slide_writer",  slide_writer_node)
        g.add_node("critic",        critic_node)
        g.add_node("supervisor",    supervisor_node)

        # Linear start: planner produces the plan, plan_executor dispatches it
        g.add_edge("planner", "plan_executor")

        # plan_executor fans out via Send(); workers loop back to plan_executor.
        g.add_edge("slide_writer", "plan_executor")
        g.add_edge("critic", "plan_executor")

        g.set_entry_point("planner")
        return g.compile()

    def stream(self, initial_state: dict, config: dict = None, stream_mode: str = "values"):
        return self._graph.stream(initial_state, config=config, stream_mode=stream_mode)

    def invoke(self, initial_state: dict, config: dict = None):
        return self._graph.invoke(initial_state, config=config)

    def _planner_node(self, state: ResearchState):
        return planner_node(
            state,
            tools_for_agent=self._agent_tools.get("planner", {}),
        )


def build_graph(
    tool_registry: dict | None = None,
    agent_tool_allowlist: dict[str, list[str]] | None = None,
) -> ResearchGraph:
    return ResearchGraph(
        tool_registry=tool_registry,
        agent_tool_allowlist=agent_tool_allowlist,
    )