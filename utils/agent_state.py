import operator
from typing import List, TypedDict, Annotated, Optional
from langchain_core.messages import BaseMessage
from pathlib import Path

# Ein Platzhalter-Typ für hochgeladene Dateien. 
# Passen Sie dies ggf. an Ihr UI-Framework an (z.B. fastapi.UploadFile).
class UploadFile:
    name: str

class AgentState(TypedDict):
    """
    Definiert den zentralen Zustand (das Gedächtnis) für das Multi-Agenten-System.
    
    Attributes:
        messages (List[BaseMessage]): Die vollständige Konversationshistorie.
        selected_files (List[str]): Vom Benutzer ausgewählte Dateinamen aus der Knowledge Base.
        uploaded_files (List[UploadFile]): Vom Benutzer direkt hochgeladene Dateien.
        linkml_schema (Optional[str]): Ein optionales LinkML-Schema, das an den DataExtractor übergeben wird.
        next (str): Bestimmt den nächsten auszuführenden Knoten (Agenten). Wird vom Supervisor gesetzt.
    """
    # Die gesamte Konversationshistorie (wird automatisch von LangGraph verwaltet)
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Eingabedaten für die Agenten
    selected_files: List[str]
    uploaded_files: List[UploadFile]
    
    # Spezifische Eingabe für den DataExtractor
    linkml_schema: Optional[str]
    
    # Wird vom Supervisor/Router gesetzt, um den nächsten Schritt zu bestimmen
    next: str