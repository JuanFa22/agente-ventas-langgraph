"""Registro de la traza ReAct (Reason -> Act -> Observe) en JSON y log legible."""
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


@dataclass
class TraceEvent:
    paso: int
    nodo: str
    tipo: str  # "razonamiento" | "accion" | "observacion" | "respuesta_final"
    contenido: str
    detalle: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


def _texto(content: Any) -> str:
    """Normaliza contenido (str o lista de bloques) a texto plano."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content).strip()
    return str(content)


class TraceRecorder:
    def __init__(self, thread_id: str, pregunta: str) -> None:
        self.thread_id = thread_id
        self.pregunta = pregunta
        self.eventos: list[TraceEvent] = []

    def registrar_update(self, nodo: str, mensajes: list[BaseMessage]) -> None:
        for msg in mensajes:
            paso = len(self.eventos) + 1
            if isinstance(msg, AIMessage) and msg.tool_calls:
                if _texto(msg.content):
                    self.eventos.append(TraceEvent(paso, nodo, "razonamiento", _texto(msg.content)))
                    paso += 1
                for call in msg.tool_calls:
                    self.eventos.append(
                        TraceEvent(paso, nodo, "accion", f"{call['name']}({call['args']})",
                                   {"herramienta": call["name"], "argumentos": call["args"]})
                    )
                    paso += 1
            elif isinstance(msg, AIMessage):
                self.eventos.append(TraceEvent(paso, nodo, "respuesta_final", _texto(msg.content)))
            elif isinstance(msg, ToolMessage):
                self.eventos.append(
                    TraceEvent(paso, nodo, "observacion", _texto(msg.content),
                               {"herramienta": msg.name, "es_error": _texto(msg.content).startswith("ERROR")})
                )

    def guardar(self, carpeta: Path) -> tuple[Path, Path]:
        carpeta.mkdir(parents=True, exist_ok=True)
        sello = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = carpeta / f"traza_{self.thread_id}_{sello}"
        json_path, log_path = base.with_suffix(".json"), base.with_suffix(".log")
        json_path.write_text(
            json.dumps({"thread_id": self.thread_id, "pregunta": self.pregunta,
                        "eventos": [asdict(e) for e in self.eventos]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        lineas = [f"THREAD: {self.thread_id}", f"PREGUNTA: {self.pregunta}", ""]
        iconos = {"razonamiento": "PENSAR ", "accion": "ACTUAR ", "observacion": "OBSERVAR", "respuesta_final": "RESPUESTA"}
        lineas += [f"[{e.paso:02d}] {iconos[e.tipo]:<9} ({e.nodo}) {e.contenido}" for e in self.eventos]
        log_path.write_text("\n".join(lineas) + "\n", encoding="utf-8")
        return json_path, log_path
