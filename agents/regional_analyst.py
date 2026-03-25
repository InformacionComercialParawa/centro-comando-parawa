import time
import pandas as pd
from typing import Dict, Any, List
from agents.base_agent import BaseAgent, AgentResult


class RegionalAnalystAgent(BaseAgent):
    """
    Analiza una region especifica de Sell Out.
    Recibe un DataFrame ya filtrado por region y el nombre de la region.
    """

    ANOMALY_DROP_THRESHOLD = -0.20

    def __init__(self, api_key: str):
        super().__init__(api_key=api_key, agent_name="Analista Regional")

    def analyze(
        self,
        df: pd.DataFrame,
        region_name: str,
        top_n: int = 5,
        include_gemini: bool = True,
    ) -> AgentResult:
        start_time = time.time()
        self.agent_name = f"Analista Regional [{region_name}]"
        self._log_event(f"Iniciando analisis de region '{region_name}' con {len(df):,} registros...")

        if df.empty:
            return self._safe_result(f"DataFrame vacio para region '{region_name}'.", time.time() - start_time)

        COL_VENTAS = "Total de Unidades Vendidas (und)"
        if COL_VENTAS not in df.columns:
            return self._safe_result(
                f"Columna '{COL_VENTAS}' no encontrada. Columnas: {df.columns.tolist()[:10]}",
                time.time() - start_time,
            )

        try:
            df = df.copy()
            df[COL_VENTAS] = pd.to_numeric(df[COL_VENTAS], errors="coerce").fillna(0)

            if "Fecha de Venta" in df.columns:
                df["Fecha de Venta"] = pd.to_datetime(df["Fecha de Venta"], errors="coerce")
                df["Anio"] = df["Fecha de Venta"].dt.year.astype("Int64").astype(str)
                df["Mes"] = df["Fecha de Venta"].dt.month.astype("Int64").astype(str)
            else:
                df["Anio"] = "N/A"
                df["Mes"] = "N/A"
                self._log_event("ADVERTENCIA: Columna 'Fecha de Venta' no encontrada")

            metrics = {"region": region_name}

            metrics["kpis_generales"] = self._kpis_generales(df, COL_VENTAS)
            self._log_event("KPIs generales calculados")

            metrics["tendencia_mensual"] = self._tendencia_mensual(df, COL_VENTAS)
            self._log_event("Tendencia mensual calculada")

            metrics["top_bottom_distribuidores"] = self._top_bottom_distribuidores(df, COL_VENTAS, top_n)
            self._log_event("Top/Bottom distribuidores identificados")

            metrics["anomalias"] = self._detectar_anomalias(df, COL_VENTAS)
            self._log_event("Anomalias detectadas")

            if include_gemini:
                self._log_event("Enviando metricas a Gemini...")
                gemini_response = self._interpret_with_gemini(metrics, region_name)
                result = AgentResult(
                    agent_name=self.agent_name,
                    status="success",
                    metrics=metrics,
                    insights=gemini_response.get("insights", []),
                    recommendations=gemini_response.get("recommendations", []),
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

    def _kpis_generales(self, df: pd.DataFrame, cv: str) -> Dict:
        kpis = {
            "total_registros": int(len(df)),
            "unidades_totales": round(float(df[cv].sum()), 0),
            "unidades_promedio": round(float(df[cv].mean()), 2),
        }
        if "Distribuidor" in df.columns:
            kpis["distribuidores_activos"] = int(df[df[cv] > 0]["Distribuidor"].nunique())
            kpis["total_distribuidores"] = int(df["Distribuidor"].nunique())
        if "Anio" in df.columns:
            kpis["anios"] = sorted(df["Anio"].dropna().unique().tolist())
        return kpis

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

        if len(tendencia) >= 3:
            ultimos = [t["unidades"] for t in tendencia[-3:]]
            direccion = (
                "creciente" if ultimos[-1] > ultimos[0]
                else "decreciente" if ultimos[-1] < ultimos[0]
                else "estable"
            )
        else:
            direccion = "datos_insuficientes"

        return {"disponible": True, "tendencia": tendencia, "direccion": direccion}

    def _top_bottom_distribuidores(self, df: pd.DataFrame, cv: str, top_n: int) -> Dict:
        col = "Distribuidor"
        if col not in df.columns:
            return {"disponible": False, "razon": "Columna 'Distribuidor' no encontrada"}

        ranking = df.groupby(col)[cv].sum().sort_values(ascending=False).reset_index()
        ranking.columns = ["distribuidor", "unidades"]
        total = ranking["unidades"].sum()

        if total == 0:
            return {"disponible": False, "razon": "Sin ventas registradas"}

        top = ranking.head(top_n).copy()
        top["pct"] = round((top["unidades"] / total) * 100, 2)

        bottom = ranking[ranking["unidades"] > 0].tail(top_n).copy()
        bottom["pct"] = round((bottom["unidades"] / total) * 100, 2)

        return {
            "disponible": True,
            "total_distribuidores": int(len(ranking)),
            "top": top.to_dict("records"),
            "bottom": bottom.to_dict("records"),
        }

    def _detectar_anomalias(self, df: pd.DataFrame, cv: str) -> Dict:
        if "Anio" not in df.columns or "Mes" not in df.columns:
            return {"disponible": False}

        # Anomalias agregadas de la region
        mensual = (
            df.groupby(["Anio", "Mes"])[cv]
            .sum()
            .reset_index()
            .sort_values(["Anio", "Mes"])
        )
        mensual.columns = ["anio", "mes", "unidades"]
        mensual["mom_pct"] = mensual["unidades"].pct_change()

        caidas = []
        for _, row in mensual.iterrows():
            if pd.isna(row["mom_pct"]):
                continue
            pct = float(row["mom_pct"])
            if pct <= self.ANOMALY_DROP_THRESHOLD:
                caidas.append({
                    "periodo": f"{row['anio']}-{str(row['mes']).zfill(2)}",
                    "variacion_pct": round(pct * 100, 2),
                    "unidades": round(float(row["unidades"]), 0),
                })

        caidas.sort(key=lambda x: x["variacion_pct"])

        # Anomalias por distribuidor en el ultimo periodo disponible
        caidas_distribuidores = []
        if "Distribuidor" in df.columns and len(mensual) >= 2:
            ultimo_anio = mensual.iloc[-1]["anio"]
            ultimo_mes = mensual.iloc[-1]["mes"]
            penultimo_anio = mensual.iloc[-2]["anio"]
            penultimo_mes = mensual.iloc[-2]["mes"]

            ultimo = df[(df["Anio"] == ultimo_anio) & (df["Mes"] == ultimo_mes)]
            penultimo = df[(df["Anio"] == penultimo_anio) & (df["Mes"] == penultimo_mes)]

            vol_ultimo = ultimo.groupby("Distribuidor")[cv].sum()
            vol_penultimo = penultimo.groupby("Distribuidor")[cv].sum()

            for dist in vol_ultimo.index:
                if dist in vol_penultimo.index and vol_penultimo[dist] > 0:
                    cambio = (vol_ultimo[dist] - vol_penultimo[dist]) / vol_penultimo[dist]
                    if cambio <= self.ANOMALY_DROP_THRESHOLD:
                        caidas_distribuidores.append({
                            "distribuidor": str(dist),
                            "periodo": f"{ultimo_anio}-{str(ultimo_mes).zfill(2)}",
                            "variacion_pct": round(float(cambio) * 100, 2),
                            "unidades_actual": round(float(vol_ultimo[dist]), 0),
                            "unidades_anterior": round(float(vol_penultimo[dist]), 0),
                        })

            caidas_distribuidores.sort(key=lambda x: x["variacion_pct"])

        return {
            "disponible": True,
            "total_caidas_region": len(caidas),
            "caidas_region": caidas[:10],
            "caidas_distribuidores_ultimo_periodo": caidas_distribuidores[:5],
        }

    # =================================================================
    # GEMINI INTERPRETA
    # =================================================================

    def _interpret_with_gemini(self, metrics: Dict, region_name: str) -> Dict:
        system_prompt = f"""Eres un Analista Senior de Sell Out para la region '{region_name}' de una empresa de consumo masivo en Venezuela.
Tu trabajo es INTERPRETAR metricas de volumen de ventas (en UNIDADES) ya calculadas con Pandas.

REGLAS:
1. NUNCA inventes numeros. Solo usa los datos recibidos.
2. Habla en espanol ejecutivo, conciso y directo.
3. Enfocate en la dinamica de distribuidores: quienes lideran, quienes caen, alertas.
4. Cada insight debe ser ACCIONABLE para el gerente regional.
5. La metrica es UNIDADES VENDIDAS (volumen), no dinero.

RESPONDE en este formato JSON:
{{
    "narrative": "Resumen ejecutivo de la region en 3-4 parrafos",
    "insights": ["hallazgo 1", "hallazgo 2", ...],
    "recommendations": ["accion 1", "accion 2", ...],
    "risk_level": "bajo|medio|alto|critico",
    "priority_actions": [
        {{"action": "descripcion", "urgency": "inmediata|esta_semana|este_mes", "impact": "alto|medio|bajo"}}
    ]
}}"""

        user_prompt = self._format_metrics(metrics, region_name)
        return self._call_gemini(system_prompt, user_prompt)

    def _format_metrics(self, metrics: Dict, region_name: str) -> str:
        parts = [f"=== REGION: {region_name} ==="]

        k = metrics.get("kpis_generales", {})
        if k:
            parts.append("\n--- KPIs GENERALES ---")
            parts.append(f"  Unidades totales: {k.get('unidades_totales', 0):,.0f}")
            parts.append(f"  Total registros: {k.get('total_registros', 0):,}")
            parts.append(f"  Distribuidores activos: {k.get('distribuidores_activos', 'N/A')}")
            parts.append(f"  Total distribuidores: {k.get('total_distribuidores', 'N/A')}")

        t = metrics.get("tendencia_mensual", {})
        if t.get("disponible"):
            parts.append(f"\n--- TENDENCIA ({t['direccion']}) ---")
            for item in t.get("tendencia", [])[-6:]:
                mom = item.get("mom_pct", "N/A")
                parts.append(f"  {item['periodo']}: {item['unidades']:,.0f} und (MoM: {mom}%)")

        tb = metrics.get("top_bottom_distribuidores", {})
        if tb.get("disponible"):
            parts.append(f"\n--- TOP {len(tb.get('top', []))} DISTRIBUIDORES ---")
            for item in tb.get("top", []):
                parts.append(f"  {item['distribuidor']}: {item['unidades']:,.0f} und ({item['pct']}%)")
            parts.append(f"\n--- BOTTOM {len(tb.get('bottom', []))} DISTRIBUIDORES ---")
            for item in tb.get("bottom", []):
                parts.append(f"  {item['distribuidor']}: {item['unidades']:,.0f} und ({item['pct']}%)")

        a = metrics.get("anomalias", {})
        if a.get("disponible"):
            parts.append(f"\n--- ANOMALIAS ({a['total_caidas_region']} caidas en region) ---")
            for item in a.get("caidas_region", [])[:5]:
                parts.append(f"  CAIDA REGION: {item['periodo']}: {item['variacion_pct']}%")
            for item in a.get("caidas_distribuidores_ultimo_periodo", []):
                parts.append(
                    f"  CAIDA DIST: {item['distribuidor']} en {item['periodo']}: "
                    f"{item['variacion_pct']}% ({item['unidades_anterior']:,.0f} → {item['unidades_actual']:,.0f})"
                )

        return "\n".join(parts)
