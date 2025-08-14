from utils.agent_state import AgentState
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

class RouteQuery(BaseModel):
    next_agent: str = Field(
        description="Must be 'SchemaModeler' or 'ETLAgent'.", 
        enum=["SchemaModeler", "ETLAgent"]
    )

async def orchestrator_node(llm, state: AgentState):
    print("--- SUPERVISOR ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a supervisor routing user requests.
        - 'SchemaModeler': Use this agent if the user ONLY wants to understand, analyze, or generate a schema (like LinkML) for their data. This is for analysis tasks.
        - 'ETLAgent': Use this agent if the user wants to ACTUALLY load, process, transform, and save data into a database. This is for operational ETL tasks.
        Based on the last user request, choose the most appropriate agent."""),
        MessagesPlaceholder(variable_name="messages")
    ])
    chain = prompt | llm.with_structured_output(RouteQuery)
    result = await chain.ainvoke({"messages": state['messages']})
    print(f"Supervisor decision: Route to {result.next_agent}")
    return {"next": result.next_agent}