import sys
import os
import json
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_openai import ChatOpenAI
import getpass
from langchain.prompts import PromptTemplate
from utils.schema_analyzer import SchemaAnalyzer
from pathlib import Path
from langchain.prompts.chat import ChatPromptTemplate
from agent_state import AgentState 


class QuietPythonREPLTool(PythonREPLTool):
    def _run(self, query: str) -> str:
        result = super()._run(query)
        # Schneide lange Ausgaben ab
        if len(result) > 1000:
            return result[:1000] + "\n... (output was cut, it is too long)"
        return result

class DataExtractor:
    """
    A class to analyze the schema of various data files and generate
    structured responses using a large language model (LLM).

    This modeler loads prompts, reads file contents, extracts schema
    and data snippets, then streams responses based on the schema using LangChain.

    Attributes:
        kb_dir (Path): Path to the knowledge base directory containing data files.
        supported_exts (List[str]): List of supported file extensions.
        llm: Initialized chat model passed externally.
        message_prompt (str): User input/message prompt template.
        data_modeler_prompt (ChatPromptTemplate): LangChain chat prompt combining system and user messages.
    """

    class QuietPythonREPLTool(PythonREPLTool):
        def _run(self, query: str) -> str:
            result = super()._run(query)
            # Schneide lange Ausgaben ab
            if len(result) > 1000:
                return result[:1000] + "\n... (output was cut, it is too long)"
            return result
    

    def __init__(
        self,
        llm,
        kb_dir: Path = Path("knowledge_base"),
        prompt_dir: Path = Path("prompts/data_extractor/")
    ):
        """
        Initialize the SchemaModeler.

        Args:
            llm: A pre-initialized chat model (e.g., LangChain-compatible LLM).
            kb_dir (Path): Directory path to knowledge base files.
            prompt_dir (Path): Directory path to system and message prompts.
        """
        if llm is None:
            raise ValueError("An LLM instance must be provided.")
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

        self.llm = llm
        self.kb_dir = kb_dir
        self.supported_exts = [".csv", ".json", ".xlsx", ".xml"]

        # Load prompts from files
        self.system_prompt_path = prompt_dir / "system_prompt.txt"
        self.message_prompt_path = prompt_dir / "message_prompt.txt"


    async def extract_to_csv(self,selected_files, uploaded_files, linkml_schema: str):
            """
            Analysiert die Struktur (Schema) der ausgewählten und hochgeladenen Dateien und übergibt diese
            an ein LLM zur Beantwortung einer Frage.

            Für jede Datei wird:
            - Das Schema extrahiert
            - Die ersten, mittleren und letzten Zeilen gelesen
            - Ein strukturierter Prompt an das LLM erstellt

            Args:
                selected_files (List[str]): Namen der ausgewählten Dateien im Knowledge-Base-Verzeichnis.
                uploaded_files (List[UploadedFile]): Liste hochgeladener Dateien (z. B. aus Web-UI).

            Yields:
                str: Laufend erzeugter Text aus der LLM-Antwort, zeilenweise gestreamt.
            """
        
            selected_paths = [self.kb_dir / name for name in selected_files]
            uploaded_paths = [Path(f.name) for f in uploaded_files or []]
            all_paths = selected_paths + uploaded_paths
            combined_content = []

            with open(self.system_prompt_path, "r") as f:
                self.system_prompt = f.read()
            with open(self.message_prompt_path, "r") as f:
                self.message_prompt = f.read()

            self.data_extracter_prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("human", "{input}"),
            ])

            system_prompt = PromptTemplate.from_templat(system_prompt).format(linkml_schema)
            
            python_tool = self.QuietPythonREPLTool(description="Führt Python-Code aus.")
            self.agent = create_react_agent(llm=self.llm, tools=[python_tool], prompt=system_prompt)

            for path in all_paths:
                try:
                    analyzer = SchemaAnalyzer(str(path))
                    schema_result = analyzer.analyze()
                    snippets = analyzer.get_file_snippets(n=10)

                    schema_str = json.dumps(schema_result, indent=2)
                    head_str = snippets.get("head", "")
                    middle_str = snippets.get("middle", "")
                    tail_str = snippets.get("tail", "")

                    file_block = f"""\n=== File: {path.name} ===\n
                    📊 Schema:
                    {schema_str}

                    📄 Head:
                    {head_str}

                    📄 Middle:
                    {middle_str}

                    📄 Tail:
                    {tail_str}
                    """
                    combined_content.append(file_block)

                except Exception as e:
                    combined_content.append(f"=== File: {path.name} ===\n Error while parsing: {str(e)}\n")

            full_data_context = "\n\n".join(combined_content)
            user_input = f"User question: {self.message_prompt}\n\nThe Schema of the Data is:\n{full_data_context}"
            prompt_messages = self.schema_modeler_prompt.invoke({"input": user_input})

            buffer = ""
            async for chunk in self.llm.astream(prompt_messages):
                if hasattr(chunk, "content") and chunk.content:
                    buffer += chunk.content
                    yield buffer



async def run_data_extractor_node(state: AgentState, llm, kb_dir: Path, prompt_dir: Path):
    """
    Ein LangGraph-Knoten, der einen ReAct-Agenten mit einem Python-Tool ausführt,
    um Daten basierend auf einem Schema zu extrahieren oder zu bearbeiten.
    """
    print("--- IM DATA EXTRACTOR KNOTEN ---")

    # 1. Benötigte Daten aus dem zentralen Zustand extrahieren
    user_message = state['messages'][-1].content
    selected_files = state.get('selected_files', [])
    uploaded_files = state.get('uploaded_files', [])
    linkml_schema = state.get('linkml_schema', "") # Hole das LinkML-Schema

    # 2. Schema-Analyse durchführen, um Kontext für den Agenten zu schaffen
    selected_paths = [kb_dir / name for name in selected_files]
    uploaded_paths = [Path(f.name) for f in uploaded_files or []]
    all_paths = selected_paths + uploaded_paths
    combined_content = []

    for path in all_paths:
        try:
            analyzer = SchemaAnalyzer(str(path))
            schema_result = analyzer.analyze()
            snippets = analyzer.get_file_snippets(n=10)
            file_block = f"""\n=== File: {path.name} (Full Path: {path.resolve()}) ===\n
            📊 Schema: {json.dumps(schema_result, indent=2)}
            📄 Snippets:
            {snippets.get("head", "")}
            ...
            {snippets.get("tail", "")}"""
            combined_content.append(file_block)
        except Exception as e:
            combined_content.append(f"=== File: {path.name} ===\n Error while parsing: {str(e)}\n")
    
    full_data_context = "\n\n".join(combined_content)

    # 3. Den ReAct-Agenten und den Executor erstellen
    # Lade den System-Prompt für den Agenten
    with open(prompt_dir / "system_prompt.txt", "r") as f:
        system_prompt_template = f.read()

    # Formatiere den Prompt mit den spezifischen Schemata
    react_prompt_template = PromptTemplate.from_template(system_prompt_template)
    react_prompt = react_prompt_template.format(linkml_schema=linkml_schema)
    
    # Erstelle das Python-Tool
    python_tool = QuietPythonREPLTool(description="Führt Python-Code aus, um Daten zu analysieren und zu manipulieren. Nutze Pandas (import pandas as pd), um Dateien zu lesen (z.B. pd.read_csv('path/to/file.csv')).")
    
    # Erstelle den Agenten
    agent = create_react_agent(llm=llm, tools=[python_tool], prompt=react_prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=[python_tool], 
        verbose=True, # Sehr nützlich für das Debugging!
        handle_parsing_errors=True
    )

    # 4. Den Agenten ausführen
    # Der Input muss den Kontext UND die eigentliche Aufgabe enthalten
    agent_input = (f"User question: {user_message}\n\n"
                   f"You have access to the following files and their schemas:\n{full_data_context}")

    print(f"\n--- Starte AgentExecutor mit Input ---\n{agent_input}\n----------------------------------")
    
    result = await agent_executor.ainvoke({"input": agent_input})

    # 5. Den Zustand mit dem Ergebnis des Agenten aktualisieren
    return {
        "messages": [AIMessage(content=result["output"])]
    }