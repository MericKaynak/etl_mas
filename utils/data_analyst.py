from utils.custom_tools import get_postgres_toolkit
from pathlib import Path
from utils.agent_state import AgentState
from langchain_core.messages import AIMessage

from langchain.agents import create_react_agent,AgentExecutor
from langchain.prompts import PromptTemplate


def create_data_analyst_executor(llm):
    sql_tools = get_postgres_toolkit(llm)

    with open("prompts/data_analyst/system_prompt.txt", "r", encoding="utf-8") as f:
        DATA_ANALYST_SYSTEM_PROMPT = f.read()

    prompt = PromptTemplate.from_template(DATA_ANALYST_SYSTEM_PROMPT)

    agent = create_react_agent(llm, sql_tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=sql_tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=10
    )

    return agent_executor


async def run_data_analyst_node(llm, state: AgentState):
    print("--- AGENT: Data Analyst ---")

    agent_executor = create_data_analyst_executor(llm)

    result = await agent_executor.ainvoke({
        "input": state['messages'][-1].content,
        "chat_history": state['messages'][:-1],
        "dialect": "Postgres",
        "top_k": 5
    })
    
    return {"messages": [AIMessage(content=result["output"])]}