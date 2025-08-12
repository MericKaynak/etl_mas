import json
from pathlib import Path
from langchain.prompts.chat import ChatPromptTemplate
from utils.schema_analyzer import SchemaAnalyzer
import json
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from utils.schema_analyzer import SchemaAnalyzer
from agent_state import AgentState 



class SchemaModeler:
    """
    Eine Klasse zur Analyse von Daten-Dateien und deren Schemata mithilfe eines LLMs (Large Language Model).

    Die Klasse lädt benutzerdefinierte Prompts, liest Dateien (lokal oder hochgeladen), extrahiert das Schema
    sowie repräsentative Ausschnitte (Head, Middle, Tail), und übergibt diese strukturiert an ein LLM
    zur Beantwortung einer benutzerdefinierten Frage.

    Unterstützt werden verschiedene Dateiformate: JSON, CSV, Excel und XML.

    Attribute:
        kb_dir (Path): Pfad zum "knowledge_base"-Verzeichnis mit den analysierbaren Dateien.
        supported_exts (List[str]): Liste der unterstützten Dateierweiterungen.
        llm: Vorinitialisiertes Chat-Modell (z. B. LangChain-kompatibles Modell wie ChatOpenAI).
        message_prompt (str): Prompt-Vorlage mit der Benutzerfrage.
        schema_modeler_prompt (ChatPromptTemplate): Kombinierter System- und Benutzerprompt für das LLM.
    """

    def __init__(
        self,
        llm,
        memory,
        kb_dir: Path = Path("knowledge_base/persistant"),
        prompt_dir: Path = Path("prompts/schema_modeler")
    ):
        """
        Initialisiert den SchemaModeler mit LLM, Pfaden zu Knowledge-Base und Prompt-Dateien.

        Args:
            llm: Ein initialisiertes Chat-Modell, das mit LangChain kompatibel ist.
            kb_dir (Path): Pfad zum Knowledge-Base-Verzeichnis mit den Daten.
            prompt_dir (Path): Pfad zu den Prompt-Textdateien ("system_prompt.txt", "message_prompt.txt").
        """
        if llm is None:
            raise ValueError("Ein LLM-Objekt muss übergeben werden.")
       
        self.memory = memory
        self.llm = llm
        self.kb_dir = kb_dir
        self.supported_exts = [".csv", ".json", ".xlsx", ".xml"]

        # Lade Prompt-Inhalte
        system_prompt_path = prompt_dir / "system_prompt.txt"
        message_prompt_path = prompt_dir / "message_prompt.txt"

        with open(system_prompt_path, "r") as f:
            system_prompt = f.read()
        with open(message_prompt_path, "r") as f:
            self.message_prompt = f.read()

        # Kombinierter Prompt für LangChain-Chat-Template
        self.schema_modeler_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

    async def get_schema(self,message, selected_files, uploaded_files):
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
        memory_vars = self.memory.load_memory_variables({})
        summary = memory_vars.get("summary", [])
        chat_history = memory_vars.get("chat_history", [])
                
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
                print(file_block)

            except Exception as e:
                combined_content.append(f"=== File: {path.name} ===\n Error while parsing: {str(e)}\n")
        

        full_data_context = "\n\n".join(combined_content)
        user_input = f"Template Prompt: {self.message_prompt}\n\nUser Question: message{message} \n\nThe Schema of the Data is:\n{full_data_context}"
        prompt_messages = self.schema_modeler_prompt.format_messages(
            message,
            summary=summary,           
            chat_history=chat_history
        )
        buffer = ""
        async for chunk in self.llm.astream(prompt_messages):
            if hasattr(chunk, "content") and chunk.content:
                buffer += chunk.content
                yield buffer

        self.memory.save_context(
            {"input": user_input},
            {"output": buffer}
        )

async def run_schema_modeler_node(state: AgentState, llm, kb_dir: Path, prompt_dir: Path):
    """
    Ein LangGraph-Knoten, der das Schema von Dateien analysiert und eine Antwort generiert.

    Erfüllt dieselben funktionalen Anforderungen wie die ursprüngliche SchemaModeler-Klasse.
    """
    print("--- IM SCHEMA MODELER KNOTEN ---")
    
    # 1. Benötigte Daten aus dem zentralen Zustand extrahieren
    # Die letzte Nachricht ist die aktuelle Benutzeranfrage
    user_message = state['messages'][-1]
    
    selected_files = state.get('selected_files', [])
    uploaded_files = state.get('uploaded_files', [])

    # 2. Prompts laden (dies geschieht nur einmal, wenn der Knoten aufgerufen wird)
    system_prompt_path = prompt_dir / "system_prompt.txt"
    message_prompt_path = prompt_dir / "message_prompt.txt" # Dies ist Ihre Vorlage
    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()
    with open(message_prompt_path, "r") as f:
        template_prompt = f.read()

    # 3. Dateipfade zusammenstellen und Schema-Analyse durchführen (identisch zu Ihrer Logik)
    selected_paths = [kb_dir / name for name in selected_files]
    uploaded_paths = [Path(f.name) for f in uploaded_files or []]
    all_paths = selected_paths + uploaded_paths
    combined_content = []

    for path in all_paths:
        try:
            # Annahme: SchemaAnalyzer ist verfügbar
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
            combined_content.append(f"=== File: {path.name} ===\n Error while parsing: {str(e)}\n")

    full_data_context = "\n\n".join(combined_content)

    # 4. Prompt für das LLM zusammenbauen
    # Die Konversationshistorie wird explizit für den Kontext übergeben
    chat_history = state['messages'][:-1]
    
    # Der Input für die "human"-Vorlage enthält jetzt alles
    user_input = (f"Template Prompt: {template_prompt}\n\n"
                  f"User Question: {user_message.content}\n\n"
                  f"The Schema of the Data is:\n{full_data_context}")

    # Wir verwenden MessagesPlaceholder, um die Historie dynamisch einzufügen
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(placeholder_name="chat_history"),
        ("human", "{input}"),
    ])
    
    # 5. LLM-Kette erstellen und aufrufen
    chain = prompt_template | llm
    
    response = await chain.ainvoke({
        "chat_history": chat_history,
        "input": user_input
    })

    # 6. Zustand aktualisieren und zurückgeben
    # Das ist der Ersatz für `memory.save_context` und `yield`
    return {
        "messages": [AIMessage(content=response.content)]
    }
