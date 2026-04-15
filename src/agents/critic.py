from langgraph.types import Command
from src.state import ResearchState, CritiqueOutput
from src.agents.base import BaseLLMAgent, _plan_to_text, _render_history

class CriticAgent(BaseLLMAgent):
    def __init__(self): super().__init__('critic')

    def run(self, state: ResearchState) -> Command:
        self._set_session_id(state)
        plan_str  = _plan_to_text(state['plan'])
        draft_str = state['draft']['document']
        rev_count = state.get('revision_count', 0)
        
        rev_hist = _render_history(state.get('revision_history', []), 'revision')
        rep_hist = _render_history(state.get('replan_history', []), 'replan')
        history_str = f"{rev_hist}\n{rep_hist}".strip()
        
        user = (
            f'Delivery Plan:\n{plan_str}\n\n'
            f'Current Cycle: {rev_count + 1}\n'
            f'Prior History:\n{history_str or "First review cycle."}\n\n'
            f'Current Draft:\n{draft_str}\n\n'
            'Compare the current draft against the plan and prior history. '
            'If prior issues were resolved and the draft is strong, do not seek new minor polish. '
            'Identify remaining issues. Mark recurring ones. Assign severity.'
        )
        turns = [{'role': 'user', 'content': user}]
        result: CritiqueOutput = self._call(turns, schema=CritiqueOutput)
        n  = len(result.issues)
        c  = sum(1 for i in result.issues if i.severity == 'critical')
        msg = f'[critic] {n} issues (critical={c})'
        return Command(update={
            'critique': result,
            'messages': [msg],
        })

def critic_node(state: ResearchState) -> Command:
    return CriticAgent().run(state)
