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
        system_prompt_path = prompt_dir / "system_prompt.txt"
        message_prompt_path = prompt_dir / "message_prompt.txt"

        with open(system_prompt_path, "r") as f:
            system_prompt = f.read()
        with open(message_prompt_path, "r") as f:
            self.message_prompt = f.read()

        self.data_extracter_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        python_tool = self.QuietPythonREPLTool(description="Führt Python-Code aus.")
        self.agent = create_react_agent(llm=llm, tools=[python_tool], prompt=system_prompt)


    async def extract_to_csv(self,selected_files, uploaded_files):
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




# Verwende das neue Tool

tools = [python_tool]

# Rest bleibt gleich
llm = ChatOpenAI(temperature=0, model="gpt-4o")
agent = create_react_agent(llm=llm, tools=tools, prompt=system_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True)




file_path = "../knowledge_base/ChinookData.json"
# --- 5. Interaktion mit dem Agenten ---
user_input = f"Ich will das du mir einen Code generiert welcher eine Daten File extrahiert passende pandas DataFrame. Wichtig nur ausfuehren schaue dir die Daten nicht vorher an. Hierzu geben ich dir eine LinkML file welche dazu dient die Struktur der Daten zu verstehen und die Daten in diesen Format zu uebernehemn in ein relationales Datenbankschema welches spawter in eine sql Datenbank ingested wird. Die Daten koenen Im unterschiedlichen Formaten vorliegen, wie z.B. XML, JSON, CSV oder Excel. Gib mir nur den Code also auch nicht in einem Codeblock. Die LinkML Datei ist: {linkml_schema}. The schema of the data is: {schema_str}. The file is a JSON. \n Tail of the file: {tail_str} \n Head: {head_str} \n Middle: {middle_str}"

response = agent_executor.invoke({
    "input": (
        f"Erstelle ein Python-Skript, das die Dateien '{file_path}' extrahiert. "
        "Verwende das gegebene LinkML-Schema ({linkml_schema}), das die Struktur der Daten vorgibt. "
        "Die Datei ist im JSON-Format. Beispiele aus der Datei: "
        f"Head: {head_str}, Middle: {middle_str}, Tail: {tail_str}. "
        "Das Ziel ist ein korrekter DataFrame nach dem Schema: {schema_str}. "
        "Der Agent soll prüfen, ob der DataFrame gültig ist. Wenn nicht, soll er den Code anpassen und erneut versuchen."
        "Gib mir am Ende eine Zusammenfassung welcge Tballen du extrahiert hast und konntest es sollen alle die in der linkmlfile Beschreiben sind vorkommen."
    )
})

print(response)

