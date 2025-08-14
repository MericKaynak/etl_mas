import os
import sys
import json
import uuid
import functools
from pathlib import Path

import gradio as gr
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
import asyncio
from dotenv import load_dotenv
from utils.schema_modeler_agent import run_schema_modeler_node
from utils.agent_state import AgentState
from utils.etl_agent import run_etl_agent_node
from utils.orchestrator import orchestrator_node
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

load_dotenv()
api_key = os.getenv("LLM_API_KEY")
llm_provider = os.getenv("LLM_PROVIDER")
llm_model = os.getenv("LLM_MODEL")


llm = init_chat_model(
    llm_model, model_provider=llm_provider, temperature=0
)

KB_DIR = Path("knowledge_base/persistant")
KB_DIR.mkdir(exist_ok=True)

EXTRACTOR_PROMPT_DIR = Path("prompts/data_extractor")
TEMP_UPLOADS_DIR = Path("temp_uploads")


EXTRACTOR_PROMPT_DIR.mkdir(exist_ok=True, parents=True)
TEMP_UPLOADS_DIR.mkdir(exist_ok=True)

from langchain.prompts import PromptTemplate

with open("prompts\etl_agent\system_prompt.txt", "r", encoding="utf-8") as f:
    ETL_AGENT_SYSTEM_PROMPT = f.read()



_app_container = {"app": None}
_app_lock = asyncio.Lock()


workflow = StateGraph(AgentState)

workflow.add_node("Orchestrator", functools.partial(orchestrator_node, llm))
workflow.add_node("SchemaModeler", functools.partial(run_schema_modeler_node, llm))
workflow.add_node("ETLAgent", functools.partial(run_etl_agent_node, llm)) 


workflow.set_entry_point("Orchestrator")

workflow.add_conditional_edges(
    "Orchestrator",
    lambda x: x["next"],
    {
        "SchemaModeler": "SchemaModeler",
        "ETLAgent": "ETLAgent"
    }
)

workflow.add_edge("SchemaModeler", END)
workflow.add_edge("ETLAgent", END)



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
                 yield f"```\n🕵️ Agent '{name}' started...\n```\n"

        elif kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                yield content
        
        elif kind == "on_tool_start":
            tool_input = event['data'].get('input', '')
            yield f"\n\n```\n⚙️ Calling Tool `{event['name']}`...\n"
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