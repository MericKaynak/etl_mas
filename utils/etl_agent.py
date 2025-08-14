from langchain.agents import create_react_agent, AgentExecutor
from utils.custom_tools import QuietPythonREPLTool,PostgreSQLToolkit
from langchain.prompts import PromptTemplate
from utils.agent_state import AgentState
from utils.helper_utils import generate_file_analysis_context
from pathlib import Path
from langchain_core.messages import AIMessage


KB_DIR = Path("knowledge_base/persistant")

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

async def run_etl_agent_node(llm, state: AgentState):
    print("--- AGENT: ETLAgent ---")
    user_message = state['messages'][-1].content
    linkml_schema = state.get("linkml_schema", "No LinkML schema provided.")

    agent_executor = create_etl_agent_executor(llm)

    all_paths = [KB_DIR / f for f in state.get('selected_files', [])] + [Path(f) for f in state.get('uploaded_files', [])]
    full_data_context = generate_file_analysis_context(all_paths)

    result = await agent_executor.ainvoke({
        "input": full_data_context,
        "chat_history": state['messages'][:-1],
        "linkml_schema": linkml_schema
    })
    
    return {"messages": [AIMessage(content=result["output"])]}


# KNOTEN 3: Supervisor (Der Router)
from langchain_core.pydantic_v1 import BaseModel, Field
class RouteQuery(BaseModel):
    next_agent: str = Field(
        description="Must be 'SchemaModeler' or 'ETLAgent'.", 
        enum=["SchemaModeler", "ETLAgent"]
    )
