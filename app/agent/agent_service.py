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

You have access to the following tools:

- retrieve_documents:
  Retrieve relevant information from the uploaded documents.

- evaluate_context:
  Evaluate whether the retrieved context is sufficient to answer the user's question.

- rewrite_query:
  Rewrite the user's question to improve document retrieval.

- answer_question:
  Generate the final answer using ONLY the retrieved context.

Guidelines:

1. Always begin by using retrieve_documents.
2. After retrieval, use evaluate_context.
3. If evaluate_context returns YES, use answer_question.
4. If evaluate_context returns NO, use rewrite_query once and retrieve_documents again.
5. Evaluate the new context again.
6. If the second evaluation is still NO, tell the user that the answer is not available in the uploaded documents.
7. Never answer using your own knowledge.
8. Always use the provided tool-calling interface.
9. Never output XML tags, function names, or implementation details.
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
                    model="openai/gpt-oss-120b",
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto"
                )

            except RateLimitError:

                return {
                    "answer": "Groq daily token limit reached. Please try again later."
                }

            except BadRequestError as e:

                print(f"Groq Error: {e}")

                return {
                    "answer": (
                        "I couldn't find enough information in the uploaded "
                        "documents to answer your question. "
                        "Please ask a question related to the uploaded documents."
                    )
                }

            message = response.choices[0].message

            # Final answer from the model
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

                # -------------------------
                # retrieve_documents
                # -------------------------
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

                # -------------------------
                # evaluate_context
                # -------------------------
                elif tool_name == "evaluate_context":

                    print(f"Context Evaluation : {tool_result}")

                    if tool_result == "NO" and iteration >= 1:

                        return {
                            "answer": (
                                "I couldn't find enough relevant information "
                                "in the uploaded documents to answer your question."
                            )
                        }

                    tool_content = str(tool_result)

                # -------------------------
                # rewrite_query
                # -------------------------
                elif tool_name == "rewrite_query":

                    print(f"Rewritten Query : {tool_result}")

                    tool_content = str(tool_result)

                # -------------------------
                # answer_question
                # -------------------------
                elif tool_name == "answer_question":

                    print("Answer generated.")

                    tool_content = str(tool_result)

                # -------------------------
                # Other tools
                # -------------------------
                elif isinstance(tool_result, (dict, list)):

                    tool_content = json.dumps(tool_result)

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