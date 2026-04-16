from datetime import datetime, timezone
from langgraph.types import Command
from src.state import ResearchState, Draft
from src.agents.base import BaseLLMAgent, _render_history, _plan_to_text

class WriterAgent(BaseLLMAgent):
    def __init__(
        self,
        tools_for_agent: dict | None = None,
    ):
        super().__init__(
            'writer',
            tools_for_agent=tools_for_agent,
        )

    def run(self, state: ResearchState) -> Command:
        self._set_session_id(state)
        plan_str   = _plan_to_text(state['plan'])
        doc_ctx    = state.get('document_context', '')
        is_revision = len(state.get('revision_history', [])) > 0
        
        initial_user = (
            f"Context from user given documents:\n{doc_ctx}\n\n"
            f"Plan:\n{plan_str}\n\n"
            "By synthesizing the context and following the plan, write the full draft."
        )
        
        if not is_revision:
            turns = [{'role': 'user', 'content': initial_user}]
        else:
            history_str = _render_history(state['revision_history'], 'revision')
            turns = [
                {'role': 'user',      'content': initial_user},
                {'role': 'assistant', 'content': state['draft']['document']},
                {'role': 'user',      'content': f'Revise the draft.\n\n{history_str}'},
            ]
            
        call_out = self._call(
            turns,
            use_tools=True,
            session_id=state.get("session_id"),
            max_tool_calls=4,
        )
        document = call_out["content"]
        version  = (state['draft']['version'] + 1) if state.get('draft') else 1
        draft = Draft(
            version=version, document=document,
            word_count=len(document.split()),
            action='revision' if is_revision else 'initial',
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        msg = f'[writer] draft v{version} ({draft["word_count"]} words, action={draft["action"]})'
        return Command(update={
            'draft': draft,
            'messages': [msg],
            'retrieval_queries': call_out["retrieval_queries"],
            'tool_calls': call_out["tool_calls"],
            'tool_results': call_out["tool_results"],
        })

def writer_node(
    state: ResearchState,
    *,
    tools_for_agent: dict | None = None,
) -> Command:
    return WriterAgent(
        tools_for_agent=tools_for_agent,
    ).run(state)
