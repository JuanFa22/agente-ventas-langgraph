"""Herramientas propias del agente (contrato: nombre, descripción, esquema de entrada)."""
import asyncio
import csv
from functools import lru_cache
from typing import Any

from langchain_core.tools import tool

import config

Row = dict[str, Any]


@lru_cache(maxsize=1)
def _cargar_ventas() -> tuple[Row, ...]:
    """Lee el CSV una sola vez y convierte los tipos numéricos."""
    with config.VENTAS_CSV.open(encoding="utf-8", newline="") as fh:
        return tuple(
            {**row, "unidades": int(row["unidades"]), "monto": float(row["monto"])}
            for row in csv.DictReader(fh)
        )


def _valores_unicos(campo: str) -> list[str]:
    return sorted({str(row[campo]) for row in _cargar_ventas()})


@tool
async def listar_catalogo() -> dict[str, list[str]]:
    """Lista los productos, regiones y meses disponibles en la base de ventas.

    Usala cuando el usuario no especifique un producto, región o mes, o cuando
    necesites conocer los nombres exactos válidos antes de consultar ventas.
    """
    return await asyncio.to_thread(
        lambda: {
            "productos": _valores_unicos("producto"),
            "regiones": _valores_unicos("region"),
            "meses": _valores_unicos("mes"),
        }
    )


@tool
async def consultar_ventas(producto: str, mes: str, region: str | None = None) -> dict[str, Any] | str:
    """Devuelve las unidades vendidas y el monto en pesos de un producto en un mes.

    Args:
        producto: nombre EXACTO del producto (por ejemplo 'Helado Familiar').
        mes: mes en formato AAAA-MM (por ejemplo '2026-03').
        region: región opcional (por ejemplo 'NOA'). Si se omite, suma todas las regiones.

    Si el producto, mes o región no existen, devuelve un mensaje de error con las
    opciones válidas para que puedas corregir y reintentar.
    """

    def _consultar() -> dict[str, Any] | str:
        filas = list(_cargar_ventas())
        for campo, valor in (("producto", producto), ("mes", mes), ("region", region)):
            if valor is None:
                continue
            validos = _valores_unicos(campo)
            if valor.strip().lower() not in {v.lower() for v in validos}:
                return f"ERROR: {campo} '{valor}' no existe. Valores válidos: {', '.join(validos)}."
            filas = [r for r in filas if str(r[campo]).lower() == valor.strip().lower()]
        if not filas:
            return "ERROR: no hay datos para esa combinación de producto, mes y región."
        return {
            "producto": filas[0]["producto"],
            "mes": filas[0]["mes"],
            "region": region or "todas",
            "unidades": sum(r["unidades"] for r in filas),
            "monto_ars": sum(r["monto"] for r in filas),
        }

    return await asyncio.to_thread(_consultar)


@tool
async def calcular_variacion(valor_actual: float, valor_anterior: float) -> dict[str, float] | str:
    """Calcula la variación porcentual entre dos valores (actual vs. anterior).

    Usala para comparar períodos, por ejemplo ventas de este mes contra el mes
    anterior. Devuelve la diferencia absoluta y el porcentaje de variación.
    """
    if valor_anterior == 0:
        return "ERROR: el valor anterior es 0; no se puede calcular una variación porcentual."
    return {
        "diferencia": round(valor_actual - valor_anterior, 2),
        "variacion_pct": round((valor_actual - valor_anterior) / valor_anterior * 100, 2),
    }


TOOLS = [listar_catalogo, consultar_ventas, calcular_variacion]
