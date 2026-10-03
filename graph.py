"""Grafo ReAct con LangGraph: agente <-> herramientas, con estado persistente."""
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, trim_messages
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph  # noqa: F401  (END documenta el cierre del flujo)
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import config
from tools import TOOLS


class AgentState(TypedDict):
    # add_messages: reducer que ANEXA mensajes al historial en vez de sobrescribirlo.
    messages: Annotated[list[BaseMessage], add_messages]


def crear_modelo() -> Runnable:
    """Instancia Gemini con las herramientas vinculadas (import diferido: no exige la clave en los tests)."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not config.GOOGLE_API_KEY:
        raise SystemExit("Falta GOOGLE_API_KEY (ver .env.example).")
    llm = ChatGoogleGenerativeAI(model=config.GEMINI_MODEL, google_api_key=config.GOOGLE_API_KEY, temperature=0)
    return llm.bind_tools(TOOLS)


def construir_grafo(model: Runnable, checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """Arma el StateGraph: START -> agent -> (tools -> agent)* -> END."""

    async def call_model(state: AgentState) -> dict[str, list[BaseMessage]]:
        # Razonar: el LLM decide por sí mismo si responde o llama herramientas (sin if/else manual).
        contexto = trim_messages(
            state["messages"], max_tokens=config.MAX_CONTEXT_MESSAGES, strategy="last",
            token_counter=len, start_on="human", allow_partial=False,
        )
        respuesta = await model.ainvoke([SystemMessage(content=config.SYSTEM_PROMPT), *contexto])
        return {"messages": [respuesta]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(TOOLS))  # Actuar + Observar: ejecuta tool_calls y devuelve ToolMessage

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)  # tool_calls -> "tools"; si no -> END
    workflow.add_edge("tools", "agent")  # ciclo de retorno: la observación vuelve al agente

    return workflow.compile(checkpointer=checkpointer)
