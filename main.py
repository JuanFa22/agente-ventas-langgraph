"""CLI del agente de ventas.

Ejemplos:
    python main.py --demo                              # conversación de demostración (3 turnos)
    python main.py --thread-id ana --pregunta "..."    # un turno; reanuda la sesión 'ana'
"""
import argparse
import asyncio
from typing import Sequence

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError

import config
from graph import construir_grafo, crear_modelo
from registro_traza import TraceRecorder

DEMO_PREGUNTAS: tuple[str, ...] = (
    "¿Cuántas unidades vendimos de helado en la región NOA en 2026-03?",
    "¿Y cómo estuvo eso comparado con el mes anterior?",
    "¿Qué producto te acabo de consultar y en qué región?",
)


async def ejecutar_turno(app, thread_id: str, pregunta: str) -> str:
    """Ejecuta un turno, registra la traza ReAct y devuelve la respuesta final."""
    recorder = TraceRecorder(thread_id, pregunta)
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": config.RECURSION_LIMIT}
    respuesta = ""
    try:
        async for update in app.astream({"messages": [HumanMessage(content=pregunta)]}, cfg, stream_mode="updates"):
            for nodo, salida in update.items():
                mensajes = salida["messages"] if isinstance(salida["messages"], list) else [salida["messages"]]
                recorder.registrar_update(nodo, mensajes)
                if nodo == "agent" and not mensajes[-1].tool_calls:
                    contenido = mensajes[-1].content
                    respuesta = contenido if isinstance(contenido, str) else " ".join(
                        b.get("text", "") for b in contenido if isinstance(b, dict))
    except GraphRecursionError:
        respuesta = "Se alcanzó el límite de pasos sin llegar a una respuesta. Reformulá la pregunta."
    json_path, log_path = recorder.guardar(config.TRACES_DIR)
    print(f"\n> {pregunta}\n{respuesta}\n(traza: {json_path.name}, {log_path.name})")
    return respuesta


async def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--thread-id", default="demo", help="identificador de la sesión (memoria persistente)")
    parser.add_argument("--pregunta", help="pregunta a enviar al agente")
    parser.add_argument("--demo", action="store_true", help="corre una conversación de ejemplo de 3 turnos")
    args = parser.parse_args(argv)
    if not (args.demo or args.pregunta):
        parser.error("indicá --demo o --pregunta")

    config.CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(config.CHECKPOINT_DB)) as checkpointer:
        app = construir_grafo(crear_modelo(), checkpointer)
        preguntas = DEMO_PREGUNTAS if args.demo else (args.pregunta,)
        for pregunta in preguntas:
            await ejecutar_turno(app, args.thread_id, pregunta)


if __name__ == "__main__":
    asyncio.run(main())
