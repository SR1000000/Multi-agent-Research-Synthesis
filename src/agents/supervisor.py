from langgraph.types import Command
from langgraph.graph import END
from src.state import ResearchState, SupervisorOutput
from src.agents.base import BaseLLMAgent

class SupervisorAgent(BaseLLMAgent):
    def __init__(self): super().__init__('supervisor')

    def run(self, state: ResearchState) -> Command:
        self._set_session_id(state)
        plan      = state['plan']
        draft     = state['draft']
        critique  = state['critique']
        rev_hist  = state.get('revision_history', [])
        rep_hist  = state.get('replan_history', [])

        history_block = ''
        if rev_hist:
            history_block += f'Revision history ({len(rev_hist)} cycles):\n'
            history_block += '\n'.join(f'  Cycle {i+1}: {h}' for i, h in enumerate(rev_hist))
            history_block += '\n'
        if rep_hist:
            history_block += f'Replan history ({len(rep_hist)} cycles):\n'
            history_block += '\n'.join(f'  Cycle {i+1}: {h}' for i, h in enumerate(rep_hist))

        issues_str = '\n'.join(
            f'[{i.severity.upper()}] {i.id} @ {i.location}: {i.description}'
            for i in critique.issues
        )
        
        user = (
            f'Query:\n{state["query"]}\n\n'
            f'Success Criteria:\n' + '\n'.join(f'- {c}' for c in plan.success_criteria) + '\n\n'
            f'Draft (v{draft["version"]}):\n{draft["document"]}\n\n'
            f'Critique Summary:\n{critique.summary}\n\n'
            f'Current Issues:\n{issues_str}\n\n'
            + (f'Prior Cycle History:\n{history_block}\n' if history_block else '')
            + 'Decide: accept, revise, or replan. Include a feedback string.'
        )
        
        turns = [{'role': 'user', 'content': user}]
        result: SupervisorOutput = self._call(turns, schema=SupervisorOutput)
        
        updates: dict = {
            'messages': [f'[supervisor] {result.decision}: {result.reasoning}'],
        }
        
        if result.decision == 'revise':
            updates['revision_history'] = [result.feedback]
            updates['revision_count']   = state.get('revision_count', 0) + 1
            return Command(update=updates, goto='writer')
        elif result.decision == 'replan':
            updates['replan_history'] = [result.feedback]
            updates['replan_count']   = state.get('replan_count', 0) + 1
            updates['revision_count'] = 0
            return Command(update=updates, goto='planner')
        else:  # accept
            return Command(update=updates, goto=END)

def supervisor_node(state: ResearchState) -> Command:
    return SupervisorAgent().run(state)
