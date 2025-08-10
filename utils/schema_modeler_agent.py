import json
from pathlib import Path
from langchain.prompts.chat import ChatPromptTemplate
from utils.schema_analyzer import SchemaAnalyzer

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

    async def get_schema(self, selected_files, uploaded_files):
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
