from utils.agent_state import AgentState
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from typing import Optional, List
from langchain.schema import BaseMessage

class RouteQuery(BaseModel):
    next_agent: str = Field(
        description="Must be 'SchemaModeler', 'ETLAgent', 'DataAnalyst', or 'Fallback'.", 
        enum=["SchemaModeler", "ETLAgent", "DataAnalyst", "Fallback"]
    )

async def orchestrator_node(llm, state: AgentState):
    print("--- SUPERVISOR ---")
    
    # Supervisor-Prompt mit Fallback
    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a supervisor routing user requests.
        - 'SchemaModeler': Use this agent if the user ONLY wants to understand, analyze, or generate a schema (like LinkML) for their data. This is for analysis tasks.
        - 'ETLAgent': Use this agent if the user wants to ACTUALLY load, process, transform, and save data into a database. This is for operational ETL tasks.
        - 'DataAnalyst': Use this agent if the user wants to query existing data in a SQL database, perform analysis, or export data (e.g., to CSV).
        - 'Fallback': Use this if the user's request doesn't clearly match any of the other agents. Respond normally, but **mention that this is outside your usual responsibility**.
        Based on the last user request, choose the most appropriate agent.
        """),
        MessagesPlaceholder(variable_name="messages")
    ])

    # Supervisor-Chain
    chain = supervisor_prompt | llm.with_structured_output(RouteQuery)
    result = await chain.ainvoke({"messages": state['messages']})

    # Fallback: User kann weiter auf den Verlauf Bezug nehmen
    if result.next_agent == "Fallback":
        print("Supervisor decision: No clear agent match. Using fallback LLM response.")

        fallback_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are now directly answering the user.
            Note: This request is outside your usual responsibility.
            You can refer to the entire conversation history to answer follow-up questions.
            """),
            MessagesPlaceholder(variable_name="messages")
        ])

        fallback_response = await (fallback_prompt | llm).ainvoke({"messages": state['messages']})
        return {"next": "Fallback", "response": fallback_response}

    print(f"Supervisor decision: Route to {result.next_agent}")
    return {"next": result.next_agent}
