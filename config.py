"""Configuración central del agente de análisis de ventas."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR: Path = Path(__file__).parent
VENTAS_CSV: Path = BASE_DIR / "data" / "ventas.csv"
CHECKPOINT_DB: Path = BASE_DIR / "data" / "checkpoints.sqlite"
TRACES_DIR: Path = BASE_DIR / "traces"

GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# Freno de seguridad: máximo de pasos del grafo por invocación (evita bucles y costos).
RECURSION_LIMIT: int = 12
# Cuántos mensajes recientes se envían al LLM (evita el "estado sucio" / contexto enorme).
MAX_CONTEXT_MESSAGES: int = 30

SYSTEM_PROMPT: str = (
    "Sos un analista de datos de ventas. Respondé en español, de forma clara y breve.\n"
    "- Usá SIEMPRE las herramientas para obtener datos; nunca inventes cifras.\n"
    "- Si una herramienta devuelve un error o datos incompletos, leé el mensaje, corregí los "
    "parámetros (por ejemplo usando los nombres exactos que te sugiere) y volvé a intentarlo. "
    "Si aun así no se puede, pedile una aclaración al usuario.\n"
    "- Recordá lo que el usuario ya dijo en la conversación (producto, región, mes) y reutilizalo "
    "cuando haga preguntas de seguimiento.\n"
    "- Los meses se expresan como AAAA-MM (por ejemplo 2026-03)."
)
