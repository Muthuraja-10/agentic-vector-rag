# 🤖 Agentic Vector RAG

An **Agentic Retrieval-Augmented Generation (RAG)** system built with **FastAPI, Groq, Pinecone, and Python**.

Unlike a basic RAG pipeline that always performs retrieval and generation in a fixed sequence, this project uses an **LLM-driven agent with tools** to decide how to process a user's question.

The agent can retrieve relevant document chunks, evaluate whether the retrieved context is sufficient, rewrite the query when retrieval quality is insufficient, retrieve again, and finally generate an answer grounded only in the uploaded documents.

---

## 🚀 Project Overview

The goal of this project is to build a practical **Agentic RAG pipeline** that can reason about retrieval quality instead of blindly trusting the first search results.

The system supports document upload and processing, semantic vector retrieval, context evaluation, query rewriting, and grounded answer generation.

### Core Flow

```text
User Question
      ↓
Agent
      ↓
retrieve_documents
      ↓
Evaluate Retrieved Context
      ↓
   ┌───────────────┐
   │ Context Good? │
   └───────┬───────┘
           │
       ┌───┴────YES────────────┐
       │                       │
       ↓                       ↓
answer_question            Rewrite Query
                               ↓
                         retrieve_documents
                               ↓
                         evaluate_context
                               ↓
                         answer / stop
```

The agent is limited to a controlled number of iterations to avoid endless tool execution.

---

## 🧠 Why Agentic RAG?

A traditional RAG system usually looks like:

```text
Question
   ↓
Embedding
   ↓
Vector Search
   ↓
Context
   ↓
LLM
   ↓
Answer
```

This approach assumes that the first retrieval step is good enough.

This project introduces an **agentic decision layer**:

```text
Question
   ↓
LLM Agent
   ↓
Choose Tool
   ↓
Retrieve
   ↓
Evaluate
   ↓
Is Context Sufficient?
   ├── YES → Answer
   │
   └── NO
        ↓
   Rewrite Query
        ↓
   Retrieve Again
        ↓
   Evaluate Again
        ↓
   Answer / Reject
```

This makes the retrieval process more adaptive.

---

## ✨ Key Features

### 📄 Document Upload

Documents are uploaded through the FastAPI backend.

The ingestion pipeline:

```text
Uploaded File
     ↓
Save File
     ↓
Read Document
     ↓
Clean Text
     ↓
Recursive Chunking
     ↓
Generate Embeddings
     ↓
Store in Pinecone
```

The repository includes a document reader, text cleaner, and recursive chunker for preprocessing.

---

### ✂️ Recursive Text Chunking

Large documents are divided into smaller chunks before embedding.

This improves retrieval granularity and allows the vector database to return smaller, relevant portions of the source document.

---

### 🔢 Embedding Generation

The project uses Pinecone's inference API with:

```text
multilingual-e5-large
```

Separate embedding modes are used for document passages and user queries:

```text
Document Chunk
     ↓
input_type = passage
     ↓
Embedding
```

and

```text
User Query
     ↓
input_type = query
     ↓
Embedding
```

---

### 🗄️ Pinecone Vector Search

Document chunks are stored as vectors in Pinecone together with metadata such as:

```text
filename
chunk_index
text
```

During retrieval, the query embedding is compared against stored vectors and the most relevant matches are returned.

The retrieval service currently requests the top **3** results for the agentic retrieval tool.

---

## 🤖 Agentic Tool System

The main intelligence of the project is the agent service.

The agent has access to four tools:

### 1. `retrieve_documents`

Retrieves the most relevant document chunks from the knowledge base.

```text
Question
   ↓
Query Embedding
   ↓
Pinecone Search
   ↓
Relevant Chunks
```

### 2. `evaluate_context`

The LLM evaluates whether the retrieved context is actually sufficient to answer the question.

It returns only:

```text
YES
```

or

```text
NO
```

The evaluator explicitly rejects contexts that are unrelated, incomplete, or only superficially matched.

### 3. `rewrite_query`

When retrieval is insufficient, the agent can rewrite the user's question into a better search query and attempt retrieval again.

### 4. `answer_question`

Generates the final response using the retrieved document context.

The system prompt requires document-related answers to be based only on retrieved context rather than the model's general knowledge.

---

## 🔄 Agent Decision Process

The agent follows a controlled workflow.

### Step 1 — Receive Question

```text
User
 ↓
Question
```

### Step 2 — Decide Whether Tools Are Needed

Simple conversational messages such as greetings can be answered directly.

Document-related questions are routed through the tool-calling workflow.

### Step 3 — Retrieve Documents

The agent calls:

```text
retrieve_documents
```

The query is embedded and searched against Pinecone.

### Step 4 — Evaluate Context

The retrieved context is passed to:

```text
evaluate_context
```

The evaluator determines whether the evidence is sufficient.

### Step 5 — Successful Retrieval

When evaluation returns:

```text
YES
```

the agent calls:

```text
answer_question
```

### Step 6 — Failed Retrieval

When evaluation returns:

```text
NO
```

the agent calls:

```text
rewrite_query
```

and performs another retrieval attempt.

### Step 7 — Final Decision

After the second retrieval and evaluation:

```text
YES → Generate answer
NO  → Tell user the answer is not available
```

The implementation also limits the agent loop to a maximum of five iterations.

---

## 🏗️ Architecture

```text
                    ┌──────────────────────┐
                    │        User          │
                    └──────────┬───────────┘
                               │
                               ↓
                    ┌──────────────────────┐
                    │      FastAPI API     │
                    └──────────┬───────────┘
                               │
                               ↓
                    ┌──────────────────────┐
                    │     AgentService     │
                    │                      │
                    │   Groq LLM Agent     │
                    └──────────┬───────────┘
                               │
               ┌───────────────┼────────────────┐
               │               │                │
               ↓               ↓                ↓
       retrieve_documents  evaluate_context  rewrite_query
               │               │                │
               │               └──────┬─────────┘
               │                      │
               ↓                      │
       ┌──────────────────┐           │
       │ RetrievalService │←──────────┘
       └────────┬─────────┘
                │
                ↓
       ┌──────────────────┐
       │ EmbeddingService │
       └────────┬─────────┘
                │
                ↓
       ┌──────────────────┐
       │ Pinecone Vector  │
       │      Store       │
       └────────┬─────────┘
                │
                ↓
       Relevant Documents
                │
                ↓
       ┌──────────────────┐
       │  answer_question │
       └────────┬─────────┘
                │
                ↓
          Final Answer
```

---

## 📥 Document Ingestion Architecture

```text
             Uploaded Document
                    │
                    ↓
             DocumentReader
                    │
                    ↓
              TextCleaner
                    │
                    ↓
            RecursiveChunker
                    │
                    ↓
          EmbeddingService
                    │
                    ↓
             Pinecone Index
```

The ingestion service orchestrates this entire flow and stores the resulting chunks and embeddings in Pinecone.

---

## 🛠️ Technology Stack

| Technology               | Purpose                                 |
| ------------------------ | --------------------------------------- |
| Python                   | Core application language               |
| FastAPI                  | REST API backend                        |
| Uvicorn                  | ASGI server                             |
| Groq                     | LLM inference and agent reasoning       |
| `openai/gpt-oss-120b`    | LLM used by the agent                   |
| Pinecone                 | Vector database and embedding inference |
| `multilingual-e5-large`  | Document/query embeddings               |
| Pydantic                 | Data validation and schemas             |
| PyMuPDF                  | PDF document processing                 |
| python-docx              | DOCX document processing                |
| LangChain Text Splitters | Text chunking                           |
| python-dotenv            | Environment configuration               |

These dependencies are defined in the repository's `requirements.txt`.

---

## 📂 Project Structure

```text
agentic-vector-rag/
│
├── app/
│   │
│   ├── agent/
│   │   └── agent_service.py
│   │
│   ├── api/
│   │   ├── agent_routes.py
│   │   └── document_routes.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   └── pinecone_client.py
│   │
│   ├── schemas/
│   │   ├── agent_schema.py
│   │   ├── chat_schema.py
│   │   └── document_schema.py
│   │
│   ├── services/
│   │   ├── chat_service.py
│   │   ├── embedding_service.py
│   │   ├── ingestion_service.py
│   │   ├── llm_service.py
│   │   ├── retrieval_service.py
│   │   └── vector_service.py
│   │
│   ├── tools/
│   │   ├── answer_question.py
│   │   ├── evaluate_tool.py
│   │   ├── retrieve_tool.py
│   │   ├── rewrite_query_tool.py
│   │   ├── tool_executor.py
│   │   └── tool_registry.py
│   │
│   ├── utils/
│   │   ├── document_reader.py
│   │   ├── recursive_chunker.py
│   │   └── text_cleaner.py
│   │
│   └── main.py
│
├── backend
├── requirements.txt
└── .gitignore
```

The current repository tree contains dedicated modules for agent logic, APIs, services, tools, schemas, utilities, and the application entry point.

---

## 🔌 API Layer

The project exposes separate API routes for:

```text
Agent operations
Document operations
```

The repository contains:

```text
app/api/agent_routes.py
app/api/document_routes.py
```

The application entry point is:

```text
app/main.py
```

---

## ⚙️ Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/Muthuraja-10/agentic-vector-rag.git

cd agentic-vector-rag
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file with the credentials required by the application, including the Groq and Pinecone configuration used by the services.

Example:

```env
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

### 5. Start the FastAPI server

```bash
uvicorn app.main:app --reload
```

The API can then be accessed through:

```text
http://127.0.0.1:8000
```

FastAPI's interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

## 💬 Example Workflow

Suppose a PDF containing company information is uploaded.

The system processes it:

```text
PDF
 ↓
Text Extraction
 ↓
Cleaning
 ↓
Chunking
 ↓
Embeddings
 ↓
Pinecone
```

The user then asks:

```text
"What is the company's revenue for 2025?"
```

The agent may execute:

```text
retrieve_documents
        ↓
evaluate_context
        ↓
YES
        ↓
answer_question
```

For a poorly phrased question:

```text
retrieve_documents
        ↓
evaluate_context
        ↓
NO
        ↓
rewrite_query
        ↓
retrieve_documents
        ↓
evaluate_context
        ↓
answer_question
```

This is the central difference between the project's agentic workflow and a fixed retrieval pipeline.

---

## 🧱 Design Principles

### Grounded Answers

The system is designed so document questions are answered from retrieved context rather than from the LLM's general knowledge.

### Retrieval Validation

Retrieval is not automatically treated as correct. The context is explicitly evaluated before the final answer is generated.

### Query Refinement

Poor retrieval can trigger query rewriting and another search attempt.

### Tool-Based Agent Architecture

Agent capabilities are exposed as registered tools instead of putting the entire workflow into one large function. The repository currently registers four agent tools.

### Separation of Responsibilities

The codebase separates:

```text
Agent Logic
API Layer
Business Services
Tools
Schemas
Utilities
Vector Storage
```

This makes the project easier to extend as additional capabilities are added.

---

## 🎯 What I Learned Building This Project

This project explores practical implementation of:

* Retrieval-Augmented Generation
* Vector embeddings
* Semantic search
* Pinecone vector databases
* LLM tool calling
* Agentic workflows
* Retrieval evaluation
* Query rewriting
* Document ingestion
* FastAPI backend architecture
* Grounded LLM responses
* Modular AI application design

---

## 🔮 Future Improvements

The project can be extended toward more advanced Agentic RAG architectures:

```text
Current
   ↓
Agentic Vector RAG
   ↓
Multi-Agent RAG
   ↓
Hybrid Search
(BM25 + Vector)
   ↓
Graph RAG
   ↓
Long-Term Memory
   ↓
Conversational RAG
   ↓
MCP-based Tool Integration
```

Possible future improvements include:

* Hybrid keyword + vector retrieval
* Reranking
* Better retrieval evaluation
* Multi-agent collaboration
* Conversational memory
* Graph-based retrieval
* MCP tool integration
* Observability and tracing
* Evaluation datasets and automated benchmarking

---

## 📌 Project Status

**Status:** Active learning / development project

The current implementation contains:

```text
✅ Document ingestion
✅ PDF / DOCX processing support
✅ Recursive chunking
✅ Embedding generation
✅ Pinecone vector storage
✅ Semantic retrieval
✅ LLM-based context evaluation
✅ Query rewriting
✅ Tool-calling agent
✅ Grounded answer generation
✅ FastAPI API layer
```

---

## 👨‍💻 Author

**Muthuraja**

GitHub:
https://github.com/Muthuraja-10

Repository:
https://github.com/Muthuraja-10/agentic-vector-rag

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐.
