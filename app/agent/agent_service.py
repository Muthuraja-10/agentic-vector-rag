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
  Evaluate whether the retrieved context is sufficient to answer
  the user's question.

- rewrite_query:
  Rewrite the user's question to improve document retrieval
  when the retrieved context is insufficient.

- answer_question:
  Generate the final answer using ONLY the retrieved context.


Guidelines:

1. If the user sends a simple greeting or casual conversation such as
   "hello", "hi", "hey", "good morning", "how are you",
   "who are you", or "what can you do",
   respond naturally without using any tools.

2. For questions that require information from the uploaded documents,
   begin by calling retrieve_documents.

3. After retrieval, call evaluate_context.

4. If evaluate_context returns YES, call answer_question.

5. If evaluate_context returns NO, call rewrite_query once.

6. After rewrite_query, call retrieve_documents again using the
   rewritten query.

7. After the second retrieval, call evaluate_context again.

8. If the second evaluation returns NO, tell the user that the answer
   is not available in the uploaded documents.

9. Never answer document-related questions using your own knowledge.

10. For document-related questions, always use the provided
    tool-calling interface.

11. Never output XML tags, function names, tool-call syntax,
    or implementation details.
"""
            },
            {
                "role": "user",
                "content": question
            }
        ]

        # Maximum number of LLM/tool iterations.
        # Allows the complete retry flow:
        #
        # 1. retrieve
        # 2. evaluate -> NO
        # 3. rewrite
        # 4. retrieve
        # 5. evaluate
        # 6. answer
        # 7. final response
        max_iterations = 7

        # Allows only one query rewrite attempt.
        rewrite_attempted = False

        for iteration in range(max_iterations):

            print(
                f"\n========== ITERATION {iteration + 1} =========="
            )

            # --------------------------------
            # Ask the LLM what to do next
            # --------------------------------

            try:

                response = self.client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto"
                )

            except RateLimitError:

                return {
                    "answer": (
                        "Groq daily token limit reached. "
                        "Please try again later."
                    )
                }

            except BadRequestError as e:

                print(f"Groq Error: {e}")

                return {
                    "answer": (
                        "I couldn't process the request. "
                        "Please try asking a question related "
                        "to the uploaded documents."
                    )
                }

            # Get the assistant message
            message = response.choices[0].message

            # --------------------------------
            # No tool call = final response
            # --------------------------------

            if not message.tool_calls:

                print(
                    "\n========== FINAL ANSWER =========="
                )

                print(message.content)

                return {
                    "answer": message.content
                }

            # --------------------------------
            # Store assistant tool-call message
            # --------------------------------

            messages.append(message)

            # --------------------------------
            # Execute requested tools
            # --------------------------------

            for tool_call in message.tool_calls:

                tool_name = tool_call.function.name

                print(
                    f"Tool -> {tool_name}"
                )

                # --------------------------------
                # Parse tool arguments
                # --------------------------------

                try:

                    arguments = json.loads(
                        tool_call.function.arguments
                    )

                except json.JSONDecodeError:

                    print(
                        f"Invalid JSON arguments for tool: "
                        f"{tool_name}"
                    )

                    return {
                        "answer": (
                            "The agent generated an invalid "
                            "tool request."
                        )
                    }

                # --------------------------------
                # Check whether tool exists
                # --------------------------------

                if tool_name not in TOOL_FUNCTIONS:

                    print(
                        f"Unknown tool requested: {tool_name}"
                    )

                    return {
                        "answer": (
                            "The agent requested an unknown tool."
                        )
                    }

                # --------------------------------
                # Execute actual Python tool
                # --------------------------------

                try:

                    tool_result = TOOL_FUNCTIONS[tool_name](
                        **arguments
                    )

                except Exception as e:

                    print(
                        f"Tool execution error "
                        f"({tool_name}): {e}"
                    )

                    return {
                        "answer": (
                            "An error occurred while executing "
                            "the requested tool."
                        )
                    }

                # --------------------------------
                # retrieve_documents
                # --------------------------------

                if tool_name == "retrieve_documents":

                    print(
                        f"Retrieved {len(tool_result)} chunks"
                    )

                    # No retrieval results
                    if not tool_result:

                        return {
                            "answer": (
                                "I couldn't find enough relevant "
                                "information in the uploaded documents."
                            )
                        }

                    # Combine retrieved chunks
                    # into the context sent to the LLM
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

                # --------------------------------
                # evaluate_context
                # --------------------------------

                elif tool_name == "evaluate_context":

                    print(
                        f"Context Evaluation : {tool_result}"
                    )

                    # --------------------------------
                    # First NO
                    # --------------------------------
                    #
                    # If rewriting has not happened yet,
                    # allow the agent to continue.
                    #
                    # The LLM should call rewrite_query.
                    # --------------------------------

                    if tool_result == "NO":

                        if not rewrite_attempted:

                            print(
                                "Context insufficient. "
                                "Rewrite is allowed."
                            )

                        # --------------------------------
                        # Second NO
                        # --------------------------------
                        #
                        # Rewrite has already happened,
                        # so stop the agent.
                        # --------------------------------

                        else:

                            print(
                                "Context still insufficient "
                                "after rewrite. Stopping agent."
                            )

                            return {
                                "answer": (
                                    "I couldn't find enough relevant "
                                    "information in the uploaded documents "
                                    "to answer your question."
                                )
                            }

                    tool_content = str(
                        tool_result
                    )

                # --------------------------------
                # rewrite_query
                # --------------------------------

                elif tool_name == "rewrite_query":

                    # Mark rewrite as used BEFORE
                    # continuing the agent loop.
                    rewrite_attempted = True

                    print(
                        f"Rewrite attempted: "
                        f"{rewrite_attempted}"
                    )

                    print(
                        f"Rewritten Query : "
                        f"{tool_result}"
                    )

                    tool_content = str(
                        tool_result
                    )

                # --------------------------------
                # answer_question
                # --------------------------------

                elif tool_name == "answer_question":

                    print(
                        "Answer generated."
                    )

                    tool_content = str(
                        tool_result
                    )

                # --------------------------------
                # Other result types
                # --------------------------------

                elif isinstance(
                    tool_result,
                    (dict, list)
                ):

                    tool_content = json.dumps(
                        tool_result
                    )

                else:

                    tool_content = str(
                        tool_result
                    )

                # --------------------------------
                # Send tool result back to LLM
                # --------------------------------

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_content
                    }
                )

        # --------------------------------
        # Maximum iterations reached
        # --------------------------------

        print(
            "\nMaximum agent iterations reached."
        )

        return {
            "answer": (
                "Maximum agent iterations reached."
            )
        }