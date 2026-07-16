TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_documents",
            "description": (
                "Retrieve the most relevant document chunks "
                "from the knowledge base using the user's question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The user's question."
                    }
                },
                "required": [
                    "question"
                ]
            }
        }
    },  # <-- Comma added here

    {
        "type": "function",
        "function": {
            "name": "evaluate_context",
            "description": (
                "Evaluate whether the retrieved document context "
                "is sufficient to answer the user's question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The user's question."
                    },
                    "context": {
                        "type": "string",
                        "description": "The retrieved document context."
                    }
                },
                "required": [
                    "question",
                    "context"
                ]
            }
        }
    },
    {
    "type": "function",

    "function": {

        "name": "rewrite_query",

        "description": (
            "Rewrite the user's question into a better search query "
            "when retrieval quality is poor."
        ),

        "parameters": {

            "type": "object",

            "properties": {

                "question": {

                    "type": "string"

                }

            },

            "required": [

                "question"

            ]

        }

    }

},
{
    "type": "function",
    "function": {
        "name": "answer_question",
        "description": (
            "Generate the final answer using only the retrieved document context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The user's question."
                },
                "context": {
                    "type": "string",
                    "description": "The retrieved document context."
                }
            },
            "required": [
                "question",
                "context"
            ]
        }
    }
}
]