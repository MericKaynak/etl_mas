import json
from pathlib import Path
from langchain.prompts.chat import ChatPromptTemplate
from utils.schema_analyzer import SchemaAnalyzer
import json
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from utils.schema_analyzer import SchemaAnalyzer
from utils.agent_state import AgentState 
from utils.helper_utils import generate_file_analysis_context

SCHEMA_PROMPT_DIR = Path("prompts/schema_modeler")
SCHEMA_PROMPT_DIR.mkdir(exist_ok=True, parents=True)
KB_DIR = Path("knowledge_base/persistant")

async def run_schema_modeler_node(llm, state: AgentState):
    print("--- AGENT: SchemaModeler ---")
    user_message = state['messages'][-1].content
    with open(SCHEMA_PROMPT_DIR / "system_prompt.txt", "r") as f:
        system_prompt = f.read()
    all_paths = [KB_DIR / f for f in state.get('selected_files', [])] + [Path(f) for f in state.get('uploaded_files', [])]
    full_data_context = generate_file_analysis_context(all_paths)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "User Question: {input}\n\nData Context:\n{data_context}")
    ])
    chain = prompt | llm
    response = await chain.ainvoke({"chat_history": state['messages'][:-1], "input": user_message, "data_context": full_data_context})
    return {"messages": [response]}
    