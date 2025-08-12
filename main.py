import os
import sys
import json
import uuid
import functools
from pathlib import Path


import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from typing import List, TypedDict, Annotated, Optional
import operator
import asyncio
from dotenv import load_dotenv
from utils.custom_tools import QuietPythonREPLTool,PostgreSQLToolkit
from utils.schema_analyzer import SchemaAnalyzer

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialisieren Sie das LLM einmal
llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)

# Verzeichnisse für Daten und Prompts
KB_DIR = Path("knowledge_base/persistant")
SCHEMA_PROMPT_DIR = Path("prompts/schema_modeler")
EXTRACTOR_PROMPT_DIR = Path("prompts/data_extractor")
TEMP_UPLOADS_DIR = Path("temp_uploads")

# Sicherstellen, dass die Verzeichnisse existieren
KB_DIR.mkdir(exist_ok=True)
SCHEMA_PROMPT_DIR.mkdir(exist_ok=True, parents=True)
EXTRACTOR_PROMPT_DIR.mkdir(exist_ok=True, parents=True)
TEMP_UPLOADS_DIR.mkdir(exist_ok=True)

from langchain.prompts import PromptTemplate



with open("prompts/etl_agent", "r", encoding="utf-8") as f:
    ETL_AGENT_SYSTEM_PROMPT = f.read()



_app_container = {"app": None}
_app_lock = asyncio.Lock()



# --- 2. Definition des zentralen Zustands (Gedächtnis) ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    selected_files: List[str]
    uploaded_files: List[str]
    linkml_schema: Optional[str]
    next: str


# --- 3. Implementierung der Agenten-Knoten (unverändert) ---

# Helferklasse für das Python-Tool


# KNOTEN 1: Schema Modeler
async def run_schema_modeler_node(state: AgentState):
    print("--- AGENT: SchemaModeler ---")
    user_message = state['messages'][-1].content
    with open(SCHEMA_PROMPT_DIR / "system_prompt.txt", "r") as f:
        system_prompt = f.read()
    all_paths = [KB_DIR / f for f in state.get('selected_files', [])] + [Path(f) for f in state.get('uploaded_files', [])]
    combined_content = []
    for path in all_paths:
        try:
            analyzer = SchemaAnalyzer(str(path))
            schema_result = analyzer.analyze()
            snippets = analyzer.get_file_snippets(n=10)

            file_block = f"""\n=== File: {path.name} ===\n
            📊 Schema: {json.dumps(schema_result, indent=2)}
            📄 Head: {snippets.get("head", "")}
            📄 Middle: {snippets.get("middle", "")}
            📄 Tail: {snippets.get("tail", "")}"""
            combined_content.append(file_block)
        except Exception as e:
            combined_content.append(f"=== File: {path.name} ===\n Error: {str(e)}\n")
    full_data_context = "\n\n".join(combined_content)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "User Question: {input}\n\nData Context:\n{data_context}")
    ])
    chain = prompt | llm
    response = await chain.ainvoke({"chat_history": state['messages'][:-1], "input": user_message, "data_context": full_data_context})
    return {"messages": [response]}

# KNOTEN 2: Data Extractor
async def run_etl_agent_node(state: AgentState):
    print("--- AGENT: ETLAgent ---")
    user_message = state['messages'][-1].content
    linkml_schema = state.get("linkml_schema", "No LinkML schema provided.")

    # 1. Python-Tool initialisieren (wie zuvor)
    python_tool = QuietPythonREPLTool(description="Dient dazu Daten zu laden, transformieren und zu speichern.")

    try:
        sql_tools = PostgreSQLToolkit.get_tools(llm)
        print("--- Successfully connected to PostgreSQL. ---")
    except Exception as e:
        print(f"--- DATABASE CONNECTION FAILED: {e} ---")
        # Geben Sie eine Fehlermeldung zurück, wenn die DB nicht erreichbar ist
        error_message = AIMessage(content=f"Error: Could not connect to the database. Please check the connection settings and ensure the database is running. Details: {e}")
        return {"messages": [error_message]}

    # 3. Beide Toolsets kombinieren
    tools = [python_tool] + sql_tools

    # 4. Agenten mit dem neuen Prompt erstellen
    prompt = PromptTemplate.from_template(ETL_AGENT_SYSTEM_PROMPT)
    
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=15
    )

    # Den Agenten mit dem Kontext aufrufen
    # Wir übergeben die Dateipfade direkt im Input, damit der Agent weiß, welche Dateien er verarbeiten soll.
    all_paths = [str(KB_DIR / f) for f in state.get('selected_files', [])] + state.get('uploaded_files', [])
    file_context = "\n".join(all_paths)
    
    agent_input = (
        f"User question: {user_message}\n\n"
        f"LinkML Schema to use: {linkml_schema}\n\n"
        f"Process the following data files: {file_context}"
    )

    result = await agent_executor.ainvoke({
        "input": agent_input,
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

async def supervisor_node(state: AgentState):
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


workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("SchemaModeler", run_schema_modeler_node)
workflow.add_node("ETLAgent", run_etl_agent_node) # Neuer Knoten

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {
        "SchemaModeler": "SchemaModeler",
        "ETLAgent": "ETLAgent"
    }
)

workflow.add_edge("SchemaModeler", END)
workflow.add_edge("ETLAgent", END)

# ======================= KORRIGIERTER TEIL =======================
# Importiere den asynchronen Saver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Diese Funktion wird die App bei der ersten Anfrage asynchron kompilieren.
# Sie stellt sicher, dass der Checkpointer im richtigen Kontext erstellt wird.
async def get_app():
    async with _app_lock:
        if _app_container["app"] is None:
            print("--- Compiling Graph with Async Checkpointer (first time)... ---")
            
            checkpointer_manager = AsyncSqliteSaver.from_conn_string("agent_memory.db")
            checkpointer = await checkpointer_manager.__aenter__()

            # Speichere den Manager, um die Verbindung später schließen zu können
            _app_container["manager"] = checkpointer_manager
            
            # Kompiliere den Graphen mit dem *korrekten* Checkpointer-Objekt
            _app_container["app"] = workflow.compile(checkpointer=checkpointer)

    return _app_container["app"]

# Funktion zum sauberen Beenden der Datenbankverbindung beim Schließen von Gradio
async def on_shutdown():
    if manager := _app_container.get("manager"):
        print("--- Closing Checkpointer Connection ---")
        await manager.__aexit__(None, None, None)

# --- 5. Gradio-Interface (leicht angepasst) ---

# --- 5. Gradio-Interface (mit Streaming-Logik angepasst) ---

def list_kb_files():
    """Listet unterstützte Dateien aus dem Knowledge-Base-Verzeichnis auf."""
    if not KB_DIR.exists():
        return []
    return [str(p.name) for p in KB_DIR.glob("*") if p.suffix.lower() in [".csv", ".json", ".xlsx", ".xml"]]

async def respond(message: str, chat_history: list, selected_files: list, uploaded_files: list, session_id: str):
    """
    Hauptlogik zum Aufrufen des LangGraph-Agenten. 
    Diese Funktion nutzt `astream_events`, um Echtzeit-Feedback über den Agentenstatus 
    und die LLM-Antworten an die Benutzeroberfläche zu streamen.
    """
    # Rufe die kompilierte App-Instanz ab (wird beim ersten Mal erstellt)
    app = await get_app()
    
    uploaded_file_paths = []
    if uploaded_files:
        for file_obj in uploaded_files:
            # Stelle sicher, dass der Dateiname für das temporäre Verzeichnis sicher ist
            safe_filename = Path(file_obj.name).name
            temp_path = TEMP_UPLOADS_DIR / safe_filename
            # Kopiere die hochgeladene Datei an den temporären Speicherort
            with open(temp_path, "wb") as f_out, open(file_obj.name, "rb") as f_in:
                f_out.write(f_in.read())
            uploaded_file_paths.append(str(temp_path))

    input_state = {
        "messages": [HumanMessage(content=message)],
        "selected_files": selected_files or [],
        "uploaded_files": uploaded_file_paths
    }
    
    config = {"configurable": {"thread_id": session_id}}
    
    async for event in app.astream_events(input_state, config=config, version="v1"):
        kind = event["event"]
        name = event.get("name", "")

        if kind == "on_chain_start":
            if name in ["supervisor", "SchemaModeler", "DataExtractor"]:
                 yield f"**```\n🕵️ Agent '{name}' started...\n```**\n"

        elif kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # Streame das Token direkt an die UI
                yield content
        
        elif kind == "on_tool_start":
            tool_input = event['data'].get('input', '')
            yield f"\n\n**```\n⚙️ Calling Tool `{event['name']}`...**\n"
            if isinstance(tool_input, str) and tool_input:
                 yield f"```python\n{tool_input.strip()}\n```\n"

        elif kind == "on_tool_end":
            tool_output = event['data'].get('output', '')
            yield f"**```\n✅ Tool finished.**\n```\n"
            if tool_output:
                if len(tool_output) > 400:
                    tool_output = tool_output[:400] + "\n... (output truncated)"
                yield f"Tool output:\n```\n{tool_output}\n```\n\n"


async def chat_wrapper(message: str, history: list, sel_files: list, up_files: list, sess_id: str):
    """
    Wrapper-Funktion, die von Gradio aufgerufen wird. Sie verwaltet die Session-ID und streamt die Antwort.
    DIES IST DIE KORRIGIERTE VERSION.
    """
    if not message.strip() and not sel_files and not up_files:
        history.append([None, "Bitte stellen Sie eine Frage oder wählen Sie eine Datei aus."])
        yield history, sess_id
        return
        
    if sess_id is None:
        sess_id = str(uuid.uuid4())
        print(f"--- New Session Initialized: {sess_id} ---")

    history.append([message, ""])
    
    async for chunk in respond(message, history, sel_files, up_files, sess_id):
        history[-1][1] += chunk
        yield history, sess_id
    
    yield history, sess_id


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Multi-Agent LinkML Chatbot")
    session_id = gr.State(value=None)
    
    with gr.Accordion("📂 Select or upload files"):
        kb_files = gr.CheckboxGroup(label="Select files from knowledge_base", choices=list_kb_files())
        uploads = gr.File(label="Upload your own files", file_count="multiple", type="filepath")

    chatbot_display = gr.Chatbot(label="Conversation", height=600, bubble_full_width=False, render_markdown=True)
    msg_box = gr.Textbox(label="Your message", placeholder="Ask a question about the data's schema or content...", scale=4)

    msg_box.submit(
        chat_wrapper,
        [msg_box, chatbot_display, kb_files, uploads, session_id],
        [chatbot_display, session_id]
    ).then(lambda: "", outputs=[msg_box])

    
    demo.unload(on_shutdown)
    gr.Examples(
            examples=[
                ["Can you generate a LinkML schema that reflects the data structure, its format, and relationships?"],
                ["Extract, transform and load the Data into the database base of the info provided by the linkml schema"]
            ],
            inputs=msg_box
        )


demo.launch()