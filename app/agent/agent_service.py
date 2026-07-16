import json

from groq import (
    Groq,
    RateLimitError,
    BadRequestError
)

from app.core.config import settings
from app.tools.tool_registry import TOOLS
from app.tools.tool_executor import TOOL_FUNCTIONS


class AgentService:

    def __init__(self):

        self.client = Groq(
            api_key=settings.GROQ_API_KEY
        )

    def run(self, question: str):

        messages = [
            {
                "role": "system",
                "content": """
You are an Agentic RAG assistant.

Available tools:
1. retrieve_documents
2. evaluate_context
3. rewrite_query
4. answer_question

Workflow:
1. Call retrieve_documents.
2. Call evaluate_context.
3. If evaluation is NO, rewrite the query.
4. Retrieve again using the rewritten query.
5. If the second evaluation is still NO, stop and say the answer is not available in the uploaded documents.
6. If evaluation is YES, call answer_question.
7. Never answer using your own knowledge.
"""
            },
            {
                "role": "user",
                "content": question
            }
        ]

        max_iterations = 5

        for iteration in range(max_iterations):

            print(f"\n========== ITERATION {iteration + 1} ==========")

            try:

                response = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto"
                )

            except RateLimitError:

                return {
                    "answer": "Groq daily token limit reached. Please try again later."
                }

            except BadRequestError:

                return {
                    "answer":"I couldn't find enough information in the uploaded documents "
            "to answer your question. Please try asking about the uploaded "
            "documents or upload a document that contains the information you need."
                }

            message = response.choices[0].message

            # Final response
            if not message.tool_calls:

                print("\n========== FINAL ANSWER ==========")
                print(message.content)

                return {
                    "answer": message.content
                }

            messages.append(message)

            for tool_call in message.tool_calls:

                tool_name = tool_call.function.name

                arguments = json.loads(
                    tool_call.function.arguments
                )

                print(f"Tool -> {tool_name}")

                if tool_name not in TOOL_FUNCTIONS:

                    raise ValueError(
                        f"Unknown tool: {tool_name}"
                    )

                tool_result = TOOL_FUNCTIONS[tool_name](
                    **arguments
                )

                # ------------------------------
                # retrieve_documents
                # ------------------------------
                if tool_name == "retrieve_documents":

                    print(f"Retrieved {len(tool_result)} chunks")

                    if not tool_result:

                        return {
                            "answer": (
                                "I couldn't find enough relevant information "
                                "in the uploaded documents."
                            )
                        }

                    context = "\n\n".join(
                        chunk["text"]
                        for chunk in tool_result
                    )

                    tool_content = json.dumps(
                        {
                            "context": context,
                            "chunks": tool_result
                        }
                    )

                # ------------------------------
                # evaluate_context
                # ------------------------------
                elif tool_name == "evaluate_context":

                    print(f"Context Evaluation : {tool_result}")

                    # Stop after the SECOND failed evaluation
                    if (
                        tool_result == "NO"
                        and iteration >= 1
                    ):

                        return {
                            "answer": (
                                "I couldn't find enough relevant information "
                                "in the uploaded documents to answer your question."
                            )
                        }

                    tool_content = str(tool_result)

                # ------------------------------
                # rewrite_query
                # ------------------------------
                elif tool_name == "rewrite_query":

                    print(f"Rewritten Query : {tool_result}")

                    tool_content = str(tool_result)

                # ------------------------------
                # answer_question
                # ------------------------------
                elif tool_name == "answer_question":

                    print("Answer generated.")

                    tool_content = str(tool_result)

                # ------------------------------
                # Other tools
                # ------------------------------
                elif isinstance(tool_result, (dict, list)):

                    tool_content = json.dumps(
                        tool_result
                    )

                else:

                    tool_content = str(tool_result)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_content
                    }
                )

        return {
            "answer": "Maximum agent iterations reached."
        }