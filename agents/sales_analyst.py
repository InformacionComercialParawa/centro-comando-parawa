import time
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
try:
    from agents.base_agent import BaseAgent, AgentResult
except ImportError:
    from .base_agent import BaseAgent, AgentResult


class SalesAnalystAgent(BaseAgent):
    """
    Agente 1: Analista de Ventas (Sell Out) - Volumen en Unidades
    Columnas mapeadas a los Parquet reales de Parawa.
    """

    # Umbrales de deteccion
    ANOMALY_DROP_THRESHOLD = -0.20
    ANOMALY_SPIKE_THRESHOLD = 0.50
    CONCENTRATION_RISK_THRESHOLD = 0.40

    def __init__(self, api_key: str):
        super().__init__(api_key=api_key, agent_name="Analista de Ventas (Sell Out)")

    def analyze(
        self,
        df: pd.DataFrame,
        top_n: int = 5,
        include_gemini: bool = True,
    ) -> AgentResult:
        start_time = time.time()
        self._log_event(f"Iniciando analisis con {len(df):,} registros...")

        if df.empty:
            return self._safe_result("El DataFrame esta vacio.", time.time() - start_time)

        # Columna principal de ventas (volumen)
        COL_VENTAS = "Total de Unidades Vendidas (und)"
        if COL_VENTAS not in df.columns:
            return self._safe_result(
                f"Columna '{COL_VENTAS}' no encontrada. Columnas: {df.columns.tolist()[:10]}",
                time.time() - start_time,
            )

        try:
            df = df.copy()

            # --- Preparar columnas numericas ---
            df[COL_VENTAS] = pd.to_numeric(df[COL_VENTAS], errors="coerce").fillna(0)

            # --- Extraer Año y Mes de "Fecha de Venta" ---
            if "Fecha de Venta" in df.columns:
                df["Fecha de Venta"] = pd.to_datetime(df["Fecha de Venta"], errors="coerce")
                df["Anio"] = df["Fecha de Venta"].dt.year.astype("Int64").astype(str)
                df["Mes"] = df["Fecha de Venta"].dt.month.astype("Int64").astype(str)
                self._log_event("Anio y Mes extraidos de Fecha de Venta")
            else:
                df["Anio"] = "N/A"
                df["Mes"] = "N/A"
                self._log_event("ADVERTENCIA: Columna 'Fecha de Venta' no encontrada")

            # --- Ejecutar modulos de analisis ---
            metrics = {}
            metrics["resumen_general"] = self._resumen_general(df, COL_VENTAS)
            self._log_event("Resumen general calculado")

            metrics["tendencia_mensual"] = self._tendencia_mensual(df, COL_VENTAS)
            self._log_event("Tendencia mensual calculada")

            metrics["top_bottom"] = self._top_bottom_performers(df, COL_VENTAS, top_n)
            self._log_event("Top/Bottom performers identificados")

            metrics["anomalias"] = self._detectar_anomalias(df, COL_VENTAS)
            self._log_event("Anomalias detectadas")

            metrics["concentracion"] = self._analisis_concentracion(df, COL_VENTAS)
            self._log_event("Concentracion analizada")

            metrics["oportunidades"] = self._oportunidades_cruzadas(df, COL_VENTAS, top_n)
            self._log_event("Oportunidades cruzadas identificadas")

            # --- Gemini interpreta ---
            if include_gemini:
                self._log_event("Enviando metricas a Gemini...")
                gemini_response = self._interpret_with_gemini(metrics)
                # Adaptar formato Quirúrgico → campos estándar de AgentResult
                culpables = gemini_response.get("culpables", [])
                acciones = gemini_response.get("acciones", [])
                diagnostico = gemini_response.get("diagnostico", "")

                if culpables:
                    insights = []
                    if diagnostico:
                        insights.append(f"🚨 {diagnostico}")
                    for c in culpables:
                        nombre = c.get("nombre", "?")
                        und = c.get("unidades_perdidas", 0)
                        pct = c.get("pct_del_total", 0)
                        razon = c.get("razon", "")
                        insights.append(f"🕵️ {nombre}: -{und:,.0f} und ({pct}% del impacto) — {razon}")
                else:
                    insights = gemini_response.get("insights", [])

                _urgencia_icon = {"inmediata": "🔴", "esta_semana": "🟡", "este_mes": "🟢"}
                if acciones:
                    recommendations = []
                    for a in acciones:
                        urgencia = a.get("urgencia", "")
                        icon = _urgencia_icon.get(urgencia, "•")
                        accion = a.get("accion", "")
                        impacto = a.get("impacto_esperado_unidades", "")
                        if impacto:
                            recommendations.append(f"{icon} [{urgencia.upper()}] {accion} (+{impacto:,} und esperadas)")
                        else:
                            recommendations.append(f"{icon} [{urgencia.upper()}] {accion}")
                else:
                    recommendations = gemini_response.get("recommendations", [])

                result = AgentResult(
                    agent_name=self.agent_name,
                    status="success",
                    metrics=metrics,
                    insights=insights,
                    recommendations=recommendations,
                    narrative=gemini_response.get("narrative", ""),
                    raw_analysis=gemini_response,
                    execution_time_seconds=time.time() - start_time,
                )
            else:
                result = AgentResult(
                    agent_name=self.agent_name,
                    status="success",
                    metrics=metrics,
                    insights=["Pre-analisis completado (Gemini deshabilitado)"],
                    execution_time_seconds=time.time() - start_time,
                )

            self._log_event(f"Completado en {result.execution_time_seconds:.1f}s")
            return result

        except Exception as e:
            return self._safe_result(f"Error: {str(e)}", time.time() - start_time)

    # =================================================================
    # MODULOS DE PRE-ANALISIS (Pandas puro)
    # =================================================================

    def _resumen_general(self, df: pd.DataFrame, cv: str) -> Dict:
        resumen = {
            "total_registros": int(len(df)),
            "unidades_totales": round(float(df[cv].sum()), 0),
            "unidades_promedio": round(float(df[cv].mean()), 2),
            "unidades_mediana": round(float(df[cv].median()), 2),
        }
        dims = {
            "distribuidores": "Distribuidor",
            "sucursales": "Sucursal Aliado",
            "segmentos": "Segmento Cliente",
            "rutas": "Descripcion Ruta Aliado",
            "productos": "Descripcion SKU Parawa",
            "marcas": "Marca Parawa",
            "categorias": "Categoria Parawa",
            "lineas": "Linea Parawa",
        }
        for label, col in dims.items():
            if col in df.columns:
                resumen[f"total_{label}"] = int(df[col].nunique())

        if "Anio" in df.columns:
            resumen["anios"] = sorted(df["Anio"].dropna().unique().tolist())
        return resumen

    def _tendencia_mensual(self, df: pd.DataFrame, cv: str) -> Dict:
        if "Anio" not in df.columns or "Mes" not in df.columns:
            return {"disponible": False, "razon": "Sin columnas de fecha"}

        agrupado = (
            df.groupby(["Anio", "Mes"])[cv]
            .sum()
            .reset_index()
            .sort_values(["Anio", "Mes"])
        )
        agrupado.columns = ["anio", "mes", "unidades"]
        agrupado["mom_pct"] = agrupado["unidades"].pct_change()

        tendencia = []
        for _, row in agrupado.iterrows():
            entry = {
                "periodo": f"{row['anio']}-{str(row['mes']).zfill(2)}",
                "unidades": round(float(row["unidades"]), 0),
            }
            if pd.notna(row["mom_pct"]):
                entry["mom_pct"] = round(float(row["mom_pct"]) * 100, 2)
            tendencia.append(entry)

        # YoY
        anios = sorted(agrupado["anio"].unique())
        yoy = []
        if len(anios) >= 2:
            for mes_val in agrupado["mes"].unique():
                datos_mes = agrupado[agrupado["mes"] == mes_val].sort_values("anio")
                if len(datos_mes) >= 2:
                    ultimo = datos_mes.iloc[-1]
                    penultimo = datos_mes.iloc[-2]
                    if penultimo["unidades"] > 0:
                        cambio = (ultimo["unidades"] - penultimo["unidades"]) / penultimo["unidades"]
                        yoy.append({
                            "mes": str(mes_val),
                            "anio_actual": str(ultimo["anio"]),
                            "anio_anterior": str(penultimo["anio"]),
                            "unidades_actual": round(float(ultimo["unidades"]), 0),
                            "unidades_anterior": round(float(penultimo["unidades"]), 0),
                            "yoy_pct": round(float(cambio) * 100, 2),
                        })

        # Direccion
        if len(tendencia) >= 3:
            ultimos = [t["unidades"] for t in tendencia[-3:]]
            direccion = "creciente" if ultimos[-1] > ultimos[0] else "decreciente" if ultimos[-1] < ultimos[0] else "estable"
        else:
            direccion = "datos_insuficientes"

        return {
            "disponible": True,
            "tendencia": tendencia,
            "yoy": yoy,
            "direccion": direccion,
        }

    def _top_bottom_performers(self, df: pd.DataFrame, cv: str, top_n: int) -> Dict:
        resultado = {}
        dims = {
            "segmento": "Segmento Cliente",
            "producto": "Descripcion SKU Parawa",
            "ruta": "Descripcion Ruta Aliado",
            "sucursal": "Sucursal Aliado",
            "categoria": "Categoria Parawa",
            "marca": "Marca Parawa",
            "linea": "Linea Parawa",
        }
        for dim_key, col in dims.items():
            if col not in df.columns:
                continue
            ranking = df.groupby(col)[cv].sum().sort_values(ascending=False).reset_index()
            ranking.columns = [dim_key, "unidades"]
            total = ranking["unidades"].sum()
            if total == 0:
                continue

            top = ranking.head(top_n).copy()
            top["pct"] = round((top["unidades"] / total) * 100, 2)

            bottom = ranking[ranking["unidades"] > 0].tail(top_n).copy()
            bottom["pct"] = round((bottom["unidades"] / total) * 100, 2)

            resultado[dim_key] = {
                "total_items": int(len(ranking)),
                "top": top.to_dict("records"),
                "bottom": bottom.to_dict("records"),
            }
        return resultado

    def _detectar_anomalias(self, df: pd.DataFrame, cv: str) -> Dict:
        if "Anio" not in df.columns or "Mes" not in df.columns:
            return {"disponible": False}

        anomalias = {"caidas": [], "picos": []}
        mensual = df.groupby(["Anio", "Mes"])[cv].sum().reset_index().sort_values(["Anio", "Mes"])
        mensual.columns = ["anio", "mes", "unidades"]
        mensual["mom_pct"] = mensual["unidades"].pct_change()

        for _, row in mensual.iterrows():
            if pd.isna(row["mom_pct"]):
                continue
            pct = float(row["mom_pct"])
            periodo = f"{row['anio']}-{str(row['mes']).zfill(2)}"
            entry = {"periodo": periodo, "variacion_pct": round(pct * 100, 2), "unidades": round(float(row["unidades"]), 0)}

            if pct <= self.ANOMALY_DROP_THRESHOLD:
                anomalias["caidas"].append(entry)
            elif pct >= self.ANOMALY_SPIKE_THRESHOLD:
                anomalias["picos"].append(entry)

        anomalias["caidas"].sort(key=lambda x: x["variacion_pct"])
        anomalias["picos"].sort(key=lambda x: x["variacion_pct"], reverse=True)

        return {
            "disponible": True,
            "total_caidas": len(anomalias["caidas"]),
            "total_picos": len(anomalias["picos"]),
            "caidas": anomalias["caidas"][:10],
            "picos": anomalias["picos"][:10],
        }

    def _analisis_concentracion(self, df: pd.DataFrame, cv: str) -> Dict:
        total = df[cv].sum()
        if total == 0:
            return {"disponible": False}

        alertas = []
        dims = {
            "ruta": "Descripcion Ruta Aliado",
            "sucursal": "Sucursal Aliado",
            "producto": "Descripcion SKU Parawa",
            "segmento": "Segmento Cliente",
        }
        for dim_key, col in dims.items():
            if col not in df.columns:
                continue
            dist = df.groupby(col)[cv].sum().sort_values(ascending=False)
            share = dist / total
            for item, pct in share.items():
                if pct >= self.CONCENTRATION_RISK_THRESHOLD:
                    alertas.append({
                        "dimension": dim_key,
                        "item": str(item),
                        "pct": round(float(pct) * 100, 2),
                        "riesgo": "alto" if pct >= 0.60 else "medio",
                    })

        return {"disponible": True, "alertas": alertas}

    def _oportunidades_cruzadas(self, df: pd.DataFrame, cv: str, top_n: int) -> Dict:
        col_seg = "Segmento Cliente"
        col_prod = "Categoria Parawa"

        if col_seg not in df.columns or col_prod not in df.columns:
            return {"disponible": False}

        pivot = pd.pivot_table(df, values=cv, index=col_prod, columns=col_seg, aggfunc="sum", fill_value=0)
        oportunidades = []

        for producto in pivot.index:
            ventas = pivot.loc[producto]
            if ventas.sum() == 0:
                continue
            max_venta = ventas.max()
            seg_lider = ventas.idxmax()

            for segmento in pivot.columns:
                if ventas[segmento] == 0 and max_venta > 0:
                    oportunidades.append({
                        "categoria": str(producto),
                        "segmento_potencial": str(segmento),
                        "segmento_lider": str(seg_lider),
                        "unidades_en_lider": round(float(max_venta), 0),
                    })

        oportunidades.sort(key=lambda x: x["unidades_en_lider"], reverse=True)
        return {"disponible": True, "oportunidades": oportunidades[:top_n]}

    # =================================================================
    # GEMINI INTERPRETA
    # =================================================================

    def _interpret_with_gemini(self, metrics: Dict) -> Dict:
        system_prompt = """Eres un Analista de Ventas Senior experto en consumo masivo en Venezuela.
Tu UNICO trabajo es DIAGNOSTICAR FUGAS DE VENTAS y dar PLANES DE ACCION HIPERSPECIFICOS.

ESTRUCTURA OBLIGATORIA (sin excepciones):

\U0001f6a8 EL DIAGNOSTICO
- Que KPI cayo exactamente? (Cobertura, Frecuencia, Amplitud, Volumen x Cliente, etc.)
- Cuantifica el impacto total en UNIDADES
- Periodo afectado vs periodo anterior
- Ej: "Cobertura Ponderada cayo de 72.3% a 68.1% (-4.2pp) = perdida de ~250 unidades"

\U0001f575\ufe0f LOS CULPABLES (nombres y numeros EXACTOS)
- CUALES distribuidores, clientes, o SKUs causaron la caida?
- Ordena por magnitud de impacto (mayor perdida primero)
- Cuantifica el aporte de CADA uno en UNIDADES
- Ej:
  * DISTRIBUIDOR_CARACAS: -150 und (caida de frecuencia en 8 clientes)
  * CLIENTE_0032: -80 und (pauso compras de linea X)
  * SKU_PREMIUM: -20 und (sustitucion por generico)

\U0001f3af LA ACCION (que hacer HOY)
- Accion 1 (INMEDIATA): Descripcion especifica + a quien contactar
- Accion 2 (ESTA SEMANA): Descripcion especifica + impacto esperado
- Accion 3 (ESTE MES): Descripcion especifica + como monitorear
- NUNCA acciones genericas - deben ser nombres, numeros y contactos

REGLAS CRITICAS:
1. NUNCA digas "bajaron las ventas" sin:
   - NOMBRE exacto (distribuidor, cliente, SKU)
   - NUMERO exacto (unidades perdidas)
   - RAZON especifica (si esta en los datos)

2. Si detectas anomalia de -20% en un periodo:
   - Busca en los datos cuales dimensiones (cliente, SKU, ruta) la explican
   - Lista TODAS las que contribuyeron > 5% de la caida
   - Cuantifica cada una en unidades

3. Si hay oportunidades cruzadas (ej: Producto X fuerte en Segmento Y, ausente en Z):
   - Conviertelas en ACCIONES INMEDIATAS
   - Nombre el segmento especifico y el contact point

4. FORMATO DE RESPUESTA (JSON):
{
    "diagnostico": "Texto 2-3 frases con KPI, caida en %, impacto en unidades",
    "culpables": [
        {
            "tipo": "distribuidor|cliente|sku",
            "nombre": "NOMBRE_EXACTO",
            "unidades_perdidas": 150,
            "pct_del_total": 45.5,
            "razon": "Caida de frecuencia en 8 clientes key"
        }
    ],
    "acciones": [
        {
            "numero": 1,
            "urgencia": "inmediata|esta_semana|este_mes",
            "accion": "Descripcion especifica con nombre exacto de quien contactar",
            "impacto_esperado_unidades": 150
        }
    ],
    "narrative": "Resumen ejecutivo 3-4 parrafos: que paso, quien fue responsable, que hacer"
}

RECUERDA: Tienes toda la informacion pre-calculada en los DATOS.
No inventes numeros. Solo interpreta lo que ves."""

        user_prompt = self._format_metrics(metrics)
        return self._call_gemini(system_prompt, user_prompt)

    def _format_metrics(self, metrics: Dict) -> str:
        parts = []

        r = metrics.get("resumen_general", {})
        if r:
            parts.append("=== RESUMEN ===")
            for k, v in r.items():
                parts.append(f"  {k}: {v}")

        t = metrics.get("tendencia_mensual", {})
        if t.get("disponible"):
            parts.append(f"\n=== TENDENCIA ({t['direccion']}) ===")
            for item in t.get("tendencia", [])[-6:]:
                mom = item.get("mom_pct", "N/A")
                parts.append(f"  {item['periodo']}: {item['unidades']:,.0f} und (MoM: {mom}%)")
            if t.get("yoy"):
                parts.append("  --- YoY ---")
                for y in t["yoy"][-6:]:
                    parts.append(f"  Mes {y['mes']}: {y['yoy_pct']}%")

        tb = metrics.get("top_bottom", {})
        if tb:
            parts.append("\n=== TOP/BOTTOM ===")
            for dim, data in tb.items():
                parts.append(f"  [{dim.upper()}]")
                for item in data.get("top", [])[:3]:
                    parts.append(f"    TOP: {item[dim]}: {item['unidades']:,.0f} und ({item['pct']}%)")
                for item in data.get("bottom", [])[:3]:
                    parts.append(f"    BOTTOM: {item[dim]}: {item['unidades']:,.0f} und ({item['pct']}%)")

        a = metrics.get("anomalias", {})
        if a.get("disponible"):
            parts.append(f"\n=== ANOMALIAS ({a['total_caidas']} caidas, {a['total_picos']} picos) ===")
            for item in a.get("caidas", [])[:5]:
                parts.append(f"  CAIDA: {item['periodo']}: {item['variacion_pct']}%")
            for item in a.get("picos", [])[:5]:
                parts.append(f"  PICO: {item['periodo']}: {item['variacion_pct']}%")

        c = metrics.get("concentracion", {})
        if c.get("disponible") and c.get("alertas"):
            parts.append(f"\n=== CONCENTRACION ===")
            for item in c["alertas"][:5]:
                parts.append(f"  [{item['riesgo'].upper()}] {item['dimension']}: {item['item']} = {item['pct']}%")

        o = metrics.get("oportunidades", {})
        if o.get("disponible") and o.get("oportunidades"):
            parts.append(f"\n=== OPORTUNIDADES ===")
            for item in o["oportunidades"][:5]:
                parts.append(f"  {item['categoria']}: fuerte en [{item['segmento_lider']}], ausente en [{item['segmento_potencial']}]")

        return "\n".join(parts)
