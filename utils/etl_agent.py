from langchain.agents import create_react_agent, AgentExecutor
from utils.custom_tools import QuietPythonREPLTool,PostgreSQLToolkit
from langchain.prompts import PromptTemplate


def create_etl_agent_executor(llm):
    python_tool = QuietPythonREPLTool(description="Dient dazu Daten zu laden, transformieren und zu speichern.")
    sql_tools = PostgreSQLToolkit.get_tools(llm)
    tools = [python_tool] + sql_tools

    with open("prompts\etl_agent\system_prompt.txt", "r", encoding="utf-8") as f:
        ETL_AGENT_SYSTEM_PROMPT = f.read()

    prompt = PromptTemplate.from_template(ETL_AGENT_SYSTEM_PROMPT)
    
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=20
    )

    return agent_executor