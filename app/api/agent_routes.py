from fastapi import APIRouter

from app.agent.agent_service import AgentService
from app.schemas.agent_schema import AgentRequest


agent_router = APIRouter(
    prefix="/agent",
    tags=["Agent"]
)

agent_service = AgentService()


@agent_router.post("/")
def run_agent(request: AgentRequest):

    return agent_service.run(
        request.question
    )


