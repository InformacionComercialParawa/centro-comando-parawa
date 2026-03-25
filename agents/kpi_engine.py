"""
KPI Engine — Módulo Pandas puro de KPIs estratégicos de Sell Out.
No depende de Gemini ni de BaseAgent. Todas las funciones son deterministas.

Convención de retorno por función temporal:
{
    "periodos": [str, ...],      # lista ordenada de todos los períodos
    "actual": float | None,       # valor del período más reciente
    "anterior": float | None,     # valor del período anterior (para delta)
    "variacion_pct": float | None,# (actual - anterior) / anterior * 100
    "serie": [{"periodo": str, "valor": float | None}, ...]
}
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


# Columnas canónicas (defensivas: se verifica existencia antes de usarlas)
_CV   = "Total de Unidades Vendidas (und)"
_CLI  = "Descripción Cliente"
_FECHA = "Fecha de Venta"
_POND  = "Es_Ponderado"
_DIST  = "Distribuidor"
_CAT   = "Categoria Parawa"
_MARCA = "Marca Parawa"
_LINEA = "Linea Parawa"

# SKU: primero código (más estable), luego descripción con/sin tilde
_SKU_OPCIONES = ["Código SKU Parawa", "Descripción SKU Parawa", "Descripcion SKU Parawa"]


# =================================================================
# HELPERS INTERNOS
# =================================================================

def _find_col(df: pd.DataFrame, opciones: List[str]) -> Optional[str]:
    for col in opciones:
        if col in df.columns:
            return col
    return None


def _agregar_periodo(df: pd.DataFrame, temporalidad: str) -> pd.DataFrame:
    """Agrega columna 'Periodo' según la temporalidad seleccionada."""
    df = df.copy()
    if "Anio" not in df.columns or "Mes" not in df.columns:
        df["Periodo"] = "Sin_Fecha"
        return df

    anio = df["Anio"].astype(str)
    mes  = pd.to_numeric(df["Mes"], errors="coerce").fillna(0).astype(int)

    if temporalidad == "Mensual":
        df["Periodo"] = anio + "-" + mes.apply(lambda m: str(m).zfill(2))

    elif temporalidad == "Bimestral":
        bimestre = ((mes - 1) // 2 + 1).clip(lower=1)
        df["Periodo"] = anio + "-B" + bimestre.astype(str)

    elif temporalidad == "Trimestral":
        trimestre = ((mes - 1) // 3 + 1).clip(lower=1)
        df["Periodo"] = anio + "-Q" + trimestre.astype(str)

    else:
        df["Periodo"] = anio + "-" + mes.apply(lambda m: str(m).zfill(2))

    return df


def _build_result(serie: List[Dict], periodos: List[str]) -> Dict:
    """Construye el dict de retorno estándar a partir de una serie."""
    valores = [e["valor"] for e in serie if e["valor"] is not None]
    actual   = serie[-1]["valor"] if serie else None
    anterior = serie[-2]["valor"] if len(serie) >= 2 else None

    var = None
    if actual is not None and anterior is not None and anterior != 0:
        var = round((actual - anterior) / anterior * 100, 2)

    return {
        "periodos": periodos,
        "actual": actual,
        "anterior": anterior,
        "variacion_pct": var,
        "serie": serie,
    }


def _empty_result() -> Dict:
    return {"periodos": [], "actual": None, "anterior": None, "variacion_pct": None, "serie": []}


# =================================================================
# 1. COBERTURA
# =================================================================

def calcular_cobertura(df: pd.DataFrame, temporalidad: str) -> Dict:
    """
    Retorna dos sub-métricas por período:
    - cobertura_ponderados: clientes únicos activos (ponderados) / total ponderados histórico
    - cobertura_total: clientes únicos activos / total clientes histórico único
    """
    if _CLI not in df.columns:
        vacio = _empty_result()
        return {"cobertura_ponderados": vacio, "cobertura_total": vacio}

    df = _agregar_periodo(df, temporalidad)
    df[_CLI] = df[_CLI].astype(str)

    # Denominadores fijos sobre todo el histórico disponible
    total_clientes  = df[_CLI].nunique()
    total_ponderados = (
        df[df[_POND] == True][_CLI].nunique()
        if _POND in df.columns else 0
    )

    periodos = sorted(df["Periodo"].dropna().unique().tolist())
    serie_total = []
    serie_pond  = []

    for periodo in periodos:
        df_p = df[df["Periodo"] == periodo]
        activos = df_p[_CLI].nunique()

        cob_t = round(activos / total_clientes * 100, 1) if total_clientes > 0 else 0.0

        if _POND in df_p.columns and total_ponderados > 0:
            activos_p = df_p[df_p[_POND] == True][_CLI].nunique()
            cob_p = round(activos_p / total_ponderados * 100, 1)
        else:
            cob_p = None

        serie_total.append({"periodo": periodo, "valor": cob_t})
        serie_pond.append({"periodo": periodo, "valor": cob_p})

    return {
        "cobertura_ponderados": _build_result(serie_pond, periodos),
        "cobertura_total":      _build_result(serie_total, periodos),
    }


# =================================================================
# 2. FRECUENCIA
# =================================================================

def calcular_frecuencia(df: pd.DataFrame, temporalidad: str) -> Dict:
    """
    Días entre Compra Promedio: promedio de días entre compras consecutivas por cliente
    activo en cada período, promediado sobre todos los clientes con ≥ 2 fechas distintas.
    Devuelve None si no hay suficientes datos.
    """
    if _FECHA not in df.columns or _CLI not in df.columns:
        return _empty_result()

    df = _agregar_periodo(df, temporalidad)
    df[_FECHA] = pd.to_datetime(df[_FECHA], errors="coerce")
    df[_CLI] = df[_CLI].astype(str)

    periodos = sorted(df["Periodo"].dropna().unique().tolist())
    serie = []

    for periodo in periodos:
        df_p = df[df["Periodo"] == periodo].dropna(subset=[_FECHA])
        frecuencias = []

        for cliente, grupo in df_p.groupby(_CLI):
            fechas = sorted(grupo[_FECHA].dropna().dt.normalize().unique())
            if len(fechas) >= 2:
                diffs = [(fechas[i + 1] - fechas[i]).days for i in range(len(fechas) - 1)]
                frecuencias.append(sum(diffs) / len(diffs))

        valor = round(sum(frecuencias) / len(frecuencias), 1) if frecuencias else None
        serie.append({"periodo": periodo, "valor": valor})

    return _build_result(serie, periodos)


def calcular_frecuencia_compra(df: pd.DataFrame, temporalidad: str) -> Dict:
    """
    Frecuencia de Compra = activaciones únicas / (clientes únicos × períodos únicos).
    Activación = combinación única (Descripción Cliente, Fecha de Venta) en el período.
    Equivale al KPI DAX: DIVIDE(activaciones, clientes_unicos * meses_unicos).
    """
    if _FECHA not in df.columns or _CLI not in df.columns:
        return _empty_result()

    df = _agregar_periodo(df, temporalidad)
    df[_FECHA] = pd.to_datetime(df[_FECHA], errors="coerce")
    df[_CLI] = df[_CLI].astype(str)

    periodos = sorted(df["Periodo"].dropna().unique().tolist())
    serie = []

    for periodo in periodos:
        df_p = df[df["Periodo"] == periodo].dropna(subset=[_FECHA])
        clientes_unicos = df_p[_CLI].nunique()
        if clientes_unicos == 0:
            serie.append({"periodo": periodo, "valor": None})
            continue
        activaciones = df_p.drop_duplicates(subset=[_CLI, _FECHA])
        total_activaciones = len(activaciones)
        # periodos_unicos dentro del slice siempre es 1; se mantiene explícito
        # para que la fórmula sea idéntica al DAX cuando se usa filtro por período
        periodos_unicos = df_p["Periodo"].nunique()
        divisor = clientes_unicos * periodos_unicos
        valor = round(total_activaciones / divisor, 2) if divisor > 0 else None
        serie.append({"periodo": periodo, "valor": valor})

    return _build_result(serie, periodos)


# =================================================================
# 3. AMPLITUD DE PORTAFOLIO
# =================================================================

def calcular_amplitud(df: pd.DataFrame, temporalidad: str) -> Dict:
    """
    Promedio de SKUs distintos comprados por cliente activo en cada período.
    """
    col_sku = _find_col(df, _SKU_OPCIONES)
    if _CLI not in df.columns or col_sku is None:
        return _empty_result()

    df = _agregar_periodo(df, temporalidad)
    df[_CLI] = df[_CLI].astype(str)

    periodos = sorted(df["Periodo"].dropna().unique().tolist())
    serie = []

    for periodo in periodos:
        df_p = df[df["Periodo"] == periodo]
        skus_por_cliente = df_p.groupby(_CLI)[col_sku].nunique()
        valor = round(float(skus_por_cliente.mean()), 2) if len(skus_por_cliente) > 0 else None
        serie.append({"periodo": periodo, "valor": valor})

    return _build_result(serie, periodos)


# =================================================================
# 4. VOLUMEN POR CLIENTE
# =================================================================

def calcular_volumen_por_cliente(df: pd.DataFrame, temporalidad: str) -> Dict:
    """
    Unidades totales vendidas / clientes activos por período.
    """
    if _CV not in df.columns or _CLI not in df.columns:
        return _empty_result()

    df = _agregar_periodo(df, temporalidad)
    df[_CV]  = pd.to_numeric(df[_CV], errors="coerce").fillna(0)
    df[_CLI] = df[_CLI].astype(str)

    periodos = sorted(df["Periodo"].dropna().unique().tolist())
    serie = []

    for periodo in periodos:
        df_p     = df[df["Periodo"] == periodo]
        total    = df_p[_CV].sum()
        clientes = df_p[df_p[_CV] > 0][_CLI].nunique()
        valor = round(float(total / clientes), 1) if clientes > 0 else None
        serie.append({"periodo": periodo, "valor": valor})

    return _build_result(serie, periodos)


# =================================================================
# 5. PARTICIPACIÓN DE PORTAFOLIO
# =================================================================

def calcular_participacion(df: pd.DataFrame) -> Dict:
    """
    Participación % por Categoría, Marca y Línea Parawa sobre unidades totales.
    Incluye participación por Distribuidor si hay más de uno.
    """
    if _CV not in df.columns:
        return {}

    df = df.copy()
    df[_CV] = pd.to_numeric(df[_CV], errors="coerce").fillna(0)
    total = df[_CV].sum()

    if total == 0:
        return {}

    result: Dict[str, Dict] = {}

    dims = {
        "categoria": _CAT,
        "marca":     _MARCA,
        "linea":     _LINEA,
    }
    for key, col in dims.items():
        if col in df.columns:
            dist = (
                df.groupby(col)[_CV].sum()
                .sort_values(ascending=False)
            )
            result[key] = {
                str(k): round(float(v) / total * 100, 2)
                for k, v in dist.items()
                if v > 0
            }

    if _DIST in df.columns and df[_DIST].nunique() > 1:
        dist_d = (
            df.groupby(_DIST)[_CV].sum()
            .sort_values(ascending=False)
        )
        result["distribuidor"] = {
            str(k): round(float(v) / total * 100, 2)
            for k, v in dist_d.items()
            if v > 0
        }

    return result
