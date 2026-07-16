from fastapi import FastAPI

from app.api.agent_routes import agent_router
from app.api.document_routes import document_router

app = FastAPI(title="Agentic Vector Based RAG")

app.include_router(agent_router)
app.include_router(document_router)

@app.get("/")
def home():

    return{
       "message": "The agentic rag is running"
                  " You can check the Swagger UI using the /docs endpoint"
    }