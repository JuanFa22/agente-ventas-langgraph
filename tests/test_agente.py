"""Tests offline: usan un modelo simulado con tool_calls guionados (no requieren API key)."""
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from graph import construir_grafo
from tools import TOOLS, calcular_variacion, consultar_ventas


class ScriptedModel(GenericFakeChatModel):
    """Fake que acepta bind_tools y devuelve respuestas guionadas."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self


def ai(content: str = "", **call: Any) -> AIMessage:
    calls = [{"name": call["name"], "args": call["args"], "id": call.get("id", "c1")}] if call else []
    return AIMessage(content=content, tool_calls=calls)


async def test_tool_error_devuelve_opciones_validas() -> None:
    res = await consultar_ventas.ainvoke({"producto": "helado", "mes": "2026-03"})
    assert isinstance(res, str) and res.startswith("ERROR") and "Helado Familiar" in res


async def test_consulta_ok_y_variacion() -> None:
    res = await consultar_ventas.ainvoke({"producto": "Helado Familiar", "mes": "2026-03", "region": "NOA"})
    assert res["unidades"] == 5200
    var = await calcular_variacion.ainvoke({"valor_actual": 5200, "valor_anterior": 8000})
    assert var["variacion_pct"] == -35.0


async def test_ciclo_de_retorno_reintenta_tras_error(tmp_path) -> None:
    guion = iter([
        ai(name="consultar_ventas", args={"producto": "helado", "mes": "2026-03"}, id="a"),   # falla
        ai(name="consultar_ventas", args={"producto": "Helado Familiar", "mes": "2026-03"}, id="b"),  # corrige
        ai("Vendimos 13700 unidades."),
    ])
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "c.sqlite")) as saver:
        app = construir_grafo(ScriptedModel(messages=guion), saver)
        out = await app.ainvoke({"messages": [HumanMessage("ventas de helado marzo")]},
                                {"configurable": {"thread_id": "t"}, "recursion_limit": 12})
    tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs[0].content.startswith("ERROR")
    assert "unidades" in tool_msgs[1].content
    assert out["messages"][-1].content == "Vendimos 13700 unidades."


async def test_memoria_persistente_entre_sesiones(tmp_path) -> None:
    db = str(tmp_path / "c.sqlite")
    cfg = {"configurable": {"thread_id": "ana"}, "recursion_limit": 12}
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        app = construir_grafo(ScriptedModel(messages=iter([ai("Hola Ana")])), saver)
        await app.ainvoke({"messages": [HumanMessage("Soy Ana")]}, cfg)
    # "Nuevo proceso": saver y grafo nuevos sobre el mismo archivo SQLite.
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        app = construir_grafo(ScriptedModel(messages=iter([ai("Sos Ana")])), saver)
        out = await app.ainvoke({"messages": [HumanMessage("¿Quién soy?")]}, cfg)
    textos = [m.content for m in out["messages"]]
    assert "Soy Ana" in textos and len(out["messages"]) == 4


def test_herramientas_tienen_descripcion() -> None:
    assert all(t.description for t in TOOLS)
