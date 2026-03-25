import time
import pandas as pd
from typing import Dict, Any, List
from agents.base_agent import BaseAgent, AgentResult


class NationalAnalystAgent(BaseAgent):
    """
    Super Agente Nacional de Sell Out.
    Sintetiza los AgentResult de multiples agentes regionales y genera
    una narrativa ejecutiva nacional comparando regiones.
    """

    def __init__(self, api_key: str):
        super().__init__(api_key=api_key, agent_name="Analista Nacional")

    def analyze(
        self,
        regional_results: List[AgentResult],
        include_gemini: bool = True,
    ) -> AgentResult:
        start_time = time.time()
        self._log_event(f"Iniciando sintesis nacional con {len(regional_results)} regiones...")

        if not regional_results:
            return self._safe_result("No se recibieron resultados regionales.", time.time() - start_time)

        try:
            exitosos = [r for r in regional_results if r.status == "success"]
            fallidos = [r for r in regional_results if r.status != "success"]

            if not exitosos:
                return self._safe_result(
                    f"Todos los agentes regionales fallaron. Errores: {[r.error_message for r in fallidos]}",
                    time.time() - start_time,
                )

            if fallidos:
                self._log_event(f"ADVERTENCIA: {len(fallidos)} region(es) con error: {[r.agent_name for r in fallidos]}")

            metrics = {}
            metrics["resumen_cobertura"] = {
                "regiones_analizadas": len(exitosos),
                "regiones_con_error": len(fallidos),
                "regiones_exitosas": [r.metrics.get("region", r.agent_name) for r in exitosos],
                "regiones_fallidas": [r.agent_name for r in fallidos],
            }

            metrics["comparativa_regional"] = self._comparar_regiones(exitosos)
            self._log_event("Comparativa regional calculada")

            metrics["ranking_regiones"] = self._ranking_regiones(exitosos)
            self._log_event("Ranking de regiones calculado")

            metrics["consolidado_tendencias"] = self._consolidar_tendencias(exitosos)
            self._log_event("Tendencias consolidadas")

            metrics["alertas_nacionales"] = self._alertas_nacionales(exitosos)
            self._log_event("Alertas nacionales identificadas")

            if include_gemini:
                self._log_event("Enviando sintesis a Gemini para narrativa nacional...")
                gemini_response = self._interpret_with_gemini(metrics, exitosos)
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
                    insights=["Sintesis nacional completada (Gemini deshabilitado)"],
                    execution_time_seconds=time.time() - start_time,
                )

            self._log_event(f"Completado en {result.execution_time_seconds:.1f}s")
            return result

        except Exception as e:
            return self._safe_result(f"Error en sintesis nacional: {str(e)}", time.time() - start_time)

    # =================================================================
    # MODULOS DE SINTESIS (Pandas puro)
    # =================================================================

    def _comparar_regiones(self, resultados: List[AgentResult]) -> List[Dict]:
        filas = []
        for r in resultados:
            region = r.metrics.get("region", r.agent_name)
            k = r.metrics.get("kpis_generales", {})
            t = r.metrics.get("tendencia_mensual", {})

            fila = {
                "region": region,
                "unidades_totales": k.get("unidades_totales", 0),
                "total_registros": k.get("total_registros", 0),
                "distribuidores_activos": k.get("distribuidores_activos", 0),
                "tendencia": t.get("direccion", "sin_datos"),
            }

            # Variacion MoM del ultimo periodo disponible
            tendencia_lista = t.get("tendencia", [])
            if len(tendencia_lista) >= 2:
                ultimo = tendencia_lista[-1]
                fila["ultimo_periodo"] = ultimo.get("periodo", "N/A")
                fila["unidades_ultimo_periodo"] = ultimo.get("unidades", 0)
                fila["mom_pct_ultimo"] = ultimo.get("mom_pct", None)
            else:
                fila["ultimo_periodo"] = "N/A"
                fila["unidades_ultimo_periodo"] = 0
                fila["mom_pct_ultimo"] = None

            filas.append(fila)

        # Ordenar por unidades totales descendente
        filas.sort(key=lambda x: x["unidades_totales"], reverse=True)

        # Calcular participacion porcentual de cada region
        total_nacional = sum(f["unidades_totales"] for f in filas)
        for fila in filas:
            fila["pct_nacional"] = (
                round((fila["unidades_totales"] / total_nacional) * 100, 2)
                if total_nacional > 0 else 0.0
            )

        return filas

    def _ranking_regiones(self, resultados: List[AgentResult]) -> Dict:
        comparativa = self._comparar_regiones(resultados)
        if not comparativa:
            return {}

        mejor = comparativa[0]
        peor = comparativa[-1]

        # Region con mayor caida MoM
        con_mom = [r for r in comparativa if r.get("mom_pct_ultimo") is not None]
        mayor_caida = min(con_mom, key=lambda x: x["mom_pct_ultimo"]) if con_mom else None
        mayor_crecimiento = max(con_mom, key=lambda x: x["mom_pct_ultimo"]) if con_mom else None

        return {
            "mejor_volumen": {"region": mejor["region"], "unidades": mejor["unidades_totales"], "pct_nacional": mejor["pct_nacional"]},
            "peor_volumen": {"region": peor["region"], "unidades": peor["unidades_totales"], "pct_nacional": peor["pct_nacional"]},
            "mayor_caida_mom": {
                "region": mayor_caida["region"],
                "mom_pct": mayor_caida["mom_pct_ultimo"],
            } if mayor_caida else None,
            "mayor_crecimiento_mom": {
                "region": mayor_crecimiento["region"],
                "mom_pct": mayor_crecimiento["mom_pct_ultimo"],
            } if mayor_crecimiento else None,
        }

    def _consolidar_tendencias(self, resultados: List[AgentResult]) -> Dict:
        regiones_crecientes = []
        regiones_decrecientes = []
        regiones_estables = []

        for r in resultados:
            region = r.metrics.get("region", r.agent_name)
            t = r.metrics.get("tendencia_mensual", {})
            direccion = t.get("direccion", "sin_datos")

            if direccion == "creciente":
                regiones_crecientes.append(region)
            elif direccion == "decreciente":
                regiones_decrecientes.append(region)
            elif direccion == "estable":
                regiones_estables.append(region)

        total = len(resultados)
        return {
            "crecientes": regiones_crecientes,
            "decrecientes": regiones_decrecientes,
            "estables": regiones_estables,
            "pct_crecientes": round(len(regiones_crecientes) / total * 100, 1) if total > 0 else 0,
            "pct_decrecientes": round(len(regiones_decrecientes) / total * 100, 1) if total > 0 else 0,
        }

    def _alertas_nacionales(self, resultados: List[AgentResult]) -> Dict:
        regiones_con_caidas = []
        total_caidas_acumuladas = 0

        for r in resultados:
            region = r.metrics.get("region", r.agent_name)
            a = r.metrics.get("anomalias", {})
            if not a.get("disponible"):
                continue

            n_caidas = a.get("total_caidas_region", 0)
            total_caidas_acumuladas += n_caidas

            if n_caidas > 0:
                peor_caida = a.get("caidas_region", [{}])[0] if a.get("caidas_region") else {}
                regiones_con_caidas.append({
                    "region": region,
                    "total_caidas": n_caidas,
                    "peor_caida_pct": peor_caida.get("variacion_pct"),
                    "peor_caida_periodo": peor_caida.get("periodo"),
                })

        regiones_con_caidas.sort(key=lambda x: x.get("peor_caida_pct", 0))

        return {
            "regiones_con_caidas": regiones_con_caidas,
            "total_eventos_caida": total_caidas_acumuladas,
            "regiones_alertadas": len(regiones_con_caidas),
        }

    # =================================================================
    # GEMINI INTERPRETA
    # =================================================================

    def _interpret_with_gemini(self, metrics: Dict, resultados: List[AgentResult]) -> Dict:
        system_prompt = """Eres el Director de Analitica de Ventas de una empresa de consumo masivo en Venezuela.
Tu trabajo es sintetizar el desempeno NACIONAL de Sell Out (volumen en UNIDADES) comparando regiones.

REGLAS:
1. NUNCA inventes numeros. Solo usa los datos recibidos.
2. Habla en espanol ejecutivo para presentar a la Direccion General.
3. Compara regiones: identifica la de mejor y peor desempeno.
4. Genera exactamente 3 recomendaciones nacionales priorizadas.
5. Detecta patrones transversales que afecten a multiples regiones.
6. La metrica es UNIDADES VENDIDAS (volumen), no dinero.

RESPONDE en este formato JSON:
{
    "narrative": "Narrativa ejecutiva nacional en 4-5 parrafos",
    "insights": ["hallazgo nacional 1", "hallazgo nacional 2", ...],
    "recommendations": ["recomendacion nacional 1", "recomendacion nacional 2", "recomendacion nacional 3"],
    "mejor_region": "nombre de la region lider",
    "peor_region": "nombre de la region con menor desempeno",
    "risk_level": "bajo|medio|alto|critico",
    "priority_actions": [
        {"action": "descripcion", "region_objetivo": "todas|nombre_region", "urgency": "inmediata|esta_semana|este_mes", "impact": "alto|medio|bajo"}
    ]
}"""

        user_prompt = self._format_sintesis(metrics, resultados)
        return self._call_gemini(system_prompt, user_prompt)

    def _format_sintesis(self, metrics: Dict, resultados: List[AgentResult]) -> str:
        parts = ["=== SINTESIS NACIONAL DE SELL OUT ==="]

        cob = metrics.get("resumen_cobertura", {})
        parts.append(f"\n--- COBERTURA ---")
        parts.append(f"  Regiones analizadas: {cob.get('regiones_analizadas', 0)}")
        if cob.get("regiones_con_error", 0) > 0:
            parts.append(f"  Regiones con error: {cob.get('regiones_con_error')} ({cob.get('regiones_fallidas')})")

        comp = metrics.get("comparativa_regional", [])
        if comp:
            parts.append("\n--- COMPARATIVA REGIONAL (mayor a menor volumen) ---")
            for fila in comp:
                mom_str = f"MoM: {fila['mom_pct_ultimo']}%" if fila.get("mom_pct_ultimo") is not None else "MoM: N/A"
                parts.append(
                    f"  {fila['region']}: {fila['unidades_totales']:,.0f} und "
                    f"({fila['pct_nacional']}% nacional) | "
                    f"{fila['tendencia']} | {mom_str} | "
                    f"Dist. activos: {fila['distribuidores_activos']}"
                )

        rank = metrics.get("ranking_regiones", {})
        if rank:
            parts.append("\n--- RANKING DESTACADO ---")
            if rank.get("mejor_volumen"):
                m = rank["mejor_volumen"]
                parts.append(f"  LIDER EN VOLUMEN: {m['region']} ({m['unidades']:,.0f} und, {m['pct_nacional']}%)")
            if rank.get("peor_volumen"):
                p = rank["peor_volumen"]
                parts.append(f"  MENOR VOLUMEN: {p['region']} ({p['unidades']:,.0f} und, {p['pct_nacional']}%)")
            if rank.get("mayor_crecimiento_mom"):
                c = rank["mayor_crecimiento_mom"]
                parts.append(f"  MAYOR CRECIMIENTO MoM: {c['region']} (+{c['mom_pct']}%)")
            if rank.get("mayor_caida_mom"):
                ca = rank["mayor_caida_mom"]
                parts.append(f"  MAYOR CAIDA MoM: {ca['region']} ({ca['mom_pct']}%)")

        tend = metrics.get("consolidado_tendencias", {})
        if tend:
            parts.append("\n--- ESTADO DE TENDENCIAS ---")
            if tend.get("crecientes"):
                parts.append(f"  Crecientes ({tend['pct_crecientes']}%): {', '.join(tend['crecientes'])}")
            if tend.get("decrecientes"):
                parts.append(f"  Decrecientes ({tend['pct_decrecientes']}%): {', '.join(tend['decrecientes'])}")
            if tend.get("estables"):
                parts.append(f"  Estables: {', '.join(tend['estables'])}")

        alertas = metrics.get("alertas_nacionales", {})
        if alertas.get("regiones_alertadas", 0) > 0:
            parts.append(f"\n--- ALERTAS DE CAIDAS ({alertas['total_eventos_caida']} eventos totales) ---")
            for alerta in alertas.get("regiones_con_caidas", []):
                parts.append(
                    f"  {alerta['region']}: {alerta['total_caidas']} caidas, "
                    f"peor: {alerta['peor_caida_pct']}% en {alerta['peor_caida_periodo']}"
                )

        # Incluir narrativas regionales como contexto adicional
        partes_narrativas = []
        for r in resultados:
            if r.narrative:
                region = r.metrics.get("region", r.agent_name)
                partes_narrativas.append(f"\n[{region}]: {r.narrative[:400]}...")

        if partes_narrativas:
            parts.append("\n--- CONTEXTO: NARRATIVAS REGIONALES (resumen) ---")
            parts.extend(partes_narrativas)

        return "\n".join(parts)
