from utils.custom_tools import get_postgres_toolkit
from pathlib import Path
from utils.agent_state import AgentState
from langchain_core.messages import AIMessage

from langgraph.prebuilt import create_react_agent


def create_data_analyst_executor(llm): 
    sql_tools = get_postgres_toolkit(llm)


async def run_etl_agent_node(llm, state: AgentState):
    print("--- AGENT: Data Analyst ---")
    linkml_schema = state.get("linkml_schema", "No LinkML schema provided.")

    agent_executor = create_data_analyst_executor(llm)

    result = await agent_executor.ainvoke({
        "input": full_data_context,
        "chat_history": state['messages'][:-1],
        "linkml_schema": linkml_schema
    })
    
    return {"messages": [AIMessage(content=result["output"])]}