from app.tools.retrieve_tool import retrieve_documents
from app.tools.evaluate_tool import evaluate_context
from app.tools.rewrite_query_tool import rewrite_query
from app.tools.answer_question import answer_question

TOOL_FUNCTIONS ={
    "retrieve_documents":retrieve_documents,
    "evaluate_context": evaluate_context,
    "rewrite_query": rewrite_query,
    "answer_question": answer_question
}