from langgraph.types import Command
from src.state import ResearchState, DeliveryPlan
from src.agents.base import BaseLLMAgent, _render_history, _plan_to_text

class PlannerAgent(BaseLLMAgent):
    def __init__(self): super().__init__('planner')

    def _build_document_context(self, doc_id: str, chunks: list) -> str:
        """Combine document chunks into a concise overview for the agents."""
        if not chunks:
            return f"Document {doc_id}: (no content available)"
        
        context = f"document {doc_id}:\n"
        for i, c in enumerate(chunks):
            text = str(getattr(c, 'text', '')) or str(c.get('text', '')) if not isinstance(c, str) else c
            context += f"  chunk {i+1}: {text.strip()}\n"
        return context

    def run(self, state: ResearchState) -> Command:
        self._set_session_id(state)
        # TODO: we will improve this later in the future by adding tools that the planner can use.
        # document context will include:
        # overview of all documents
        # we will give planner a tool that fetch related chunks from a document given a query
        # the agents receive documents overview, it fetches related chunks from each document 
        # and builds the document context it needs

        # Build document context only once in planner run
        doc_context = state.get('document_context', '')
        if not doc_context and state.get('source_chunks'):
            doc_context = self._build_document_context(state.get('doc_id', 'unknown'), state['source_chunks'])
        
        is_replan = len(state.get('replan_history', [])) > 0
        initial_user = (
            f"Context from user given documents:\n{doc_context}\n\n"
            f"Query:\n{state['query']}\n\n"
            "Based on the context and query, create a structured delivery plan."
        )
        
        if not is_replan:
            turns = [{'role': 'user', 'content': initial_user}]
        else:
            history_str = _render_history(state['replan_history'], 'replan')
            turns = [
                {'role': 'user',      'content': initial_user},
                {'role': 'assistant', 'content': _plan_to_text(state['plan'])},
                {'role': 'user',      'content': f'The plan failed.\n\n{history_str}\n\nCreate a revised plan.'},
            ]
        
        plan = self._call(turns, schema=DeliveryPlan)
        msg = f'[planner] plan created: {plan.title} (replan={is_replan})'
        
        return Command(update={
            'plan': plan, 
            'draft': None, 
            'messages': [msg],
            'document_context': doc_context
        })

def planner_node(state: ResearchState) -> Command:
    return PlannerAgent().run(state)
