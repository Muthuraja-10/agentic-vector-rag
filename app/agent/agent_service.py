import json

from groq import Groq, RateLimitError

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
1. First call retrieve_documents.
2. Then call evaluate_context.
3. If evaluate_context returns NO, call rewrite_query.
4. Retrieve again using the rewritten query.
5. Repeat until the context is sufficient.
6. Finally call answer_question.
7. Never answer from your own knowledge.
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

            message = response.choices[0].message

            # Final answer from LLM
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

                print(f"\nTool -> {tool_name}")

                if tool_name not in TOOL_FUNCTIONS:

                    raise ValueError(
                        f"Unknown tool: {tool_name}"
                    )

                tool_result = TOOL_FUNCTIONS[tool_name](
                    **arguments
                )

                # Clean logging
                if tool_name == "retrieve_documents":

                    print(f"Retrieved {len(tool_result)} chunks")

                    if not tool_result:

                        return {
                            "answer": "No relevant documents were found."
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

                elif tool_name == "evaluate_context":

                    print(f"Context Evaluation : {tool_result}")

                    if (
                        tool_result == "NO"
                        and iteration == max_iterations - 1
                    ):

                        return {
                            "answer": (
                                "I couldn't find enough relevant information "
                                "in the uploaded documents to answer your question."
                            )
                        }

                    tool_content = str(tool_result)

                elif tool_name == "rewrite_query":

                    print(f"Rewritten Query : {tool_result}")

                    tool_content = str(tool_result)

                elif tool_name == "answer_question":

                    print("Final answer generated.")

                    tool_content = str(tool_result)

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