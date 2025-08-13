import json
from pathlib import Path
from typing import Dict, Any, List
from utils.schema_analyzer import SchemaAnalyzer

def generate_file_analysis_context(all_paths: Path) -> str:
    """
    Analysiert eine Liste von Dateien aus einem State-Objekt und erstellt einen formatierten Kontext-String.

    Diese Funktion sammelt Dateipfade aus 'selected_files' (relativ zum
    Knowledge-Base-Verzeichnis) und 'uploaded_files' (absolute Pfade),
    analysiert jede Datei auf ihr Schema und extrahiert Code-Ausschnitte
    (Head, Middle, Tail). Fehler bei einzelnen Dateien werden abgefangen
    und im Ergebnis-String vermerkt.

    Args:
        state: Ein Dictionary, das die Schlüssel 'selected_files' und/oder
               'uploaded_files' mit Listen von Dateinamen enthalten kann.
        kb_dir: Der Basispfad (als Path-Objekt) für die Wissensdatenbank,
                auf den sich die 'selected_files' beziehen.

    Returns:
        Ein einziger, formatierter String, der die Analyseergebnisse
        für alle verarbeiteten Dateien enthält.
    """
    # 1. Sammle alle Dateipfade aus dem State

    
    if not all_paths:
        return "Keine Dateien zur Analyse ausgewählt oder hochgeladen."

    # 2. Verarbeite jeden Pfad und sammle die Ergebnisse
    combined_content: List[str] = []
    for path in all_paths:
        # Überspringe, falls der Pfad aus irgendeinem Grund nicht existiert
        if not path.is_file():
            combined_content.append(f"=== File: {str(path)} ===\n Error: Datei nicht gefunden unter dem Pfad {str(path)}\n")
            continue

        try:
            # Annahme: SchemaAnalyzer wird mit dem Dateipfad als String initialisiert
            analyzer = SchemaAnalyzer(str(path))
            schema_result = analyzer.analyze()
            snippets = analyzer.get_file_snippets(n=10)

            file_block = f"""
            === File: {str(path)} ===
            📊 Schema: {json.dumps(schema_result, indent=2)}
            📄 Head: {snippets.get("head", "")}
            📄 Middle: {snippets.get("middle", "")}
            📄 Tail: {snippets.get("tail", "")}"""
            combined_content.append(file_block)
            
        except Exception as e:
            # Fängt jeden Fehler während der Analyse einer Datei ab
            combined_content.append(f"=== File: {path.name} ===\n Error: {str(e)}\n")

    # 3. Kombiniere alle Blöcke zu einem finalen String
    full_data_context = "\n\n".join(combined_content)
    return full_data_context