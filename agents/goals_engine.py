"""
GoalsEngine — 3 escenarios de metas por distribuidor, Pandas puro.
Cálculo masivo: todas las series se agregan con groupby sobre el df completo
antes de iterar por distribuidor (sin filtrar el df entero N veces).
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any


class GoalsEngine:

    def __init__(self, api_key: str = ""):
        pass  # api_key mantenido por compatibilidad con el caller

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _mom_promedio(serie: pd.Series, n_ultimos: int = 3) -> float:
        """Promedio de variación MoM (%) de los últimos n_ultimos cambios de una serie ordenada."""
        vals = serie.values
        mom = [
            (vals[i] - vals[i - 1]) / vals[i - 1] * 100
            for i in range(1, len(vals))
            if vals[i - 1] > 0
        ]
        return float(np.mean(mom[-n_ultimos:])) if mom else 0.0

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------

    def calculate_proposals(
        self,
        df: pd.DataFrame,
        periodo: str,
        periodicidad: str,
    ) -> List[Dict[str, Any]]:
        """
        Calcula 3 escenarios de meta para cada distribuidor único en df.
        Cálculo masivo: groupby una sola vez sobre el df completo.

        Returns lista de dicts con claves:
          distribuidor, region,
          promedio_6m, tendencia_mom_ventas,
          clientes_activos, tendencia_mom_activacion,
          promedio_por_cliente, amplitud_actual,
          meta_escenario1, just_escenario1,
          meta_escenario2, just_escenario2,
          meta_escenario3, just_escenario3,
          nivel_confianza
        """
        CV   = "Total de Unidades Vendidas (und)"
        CLI  = "Descripción Cliente"
        FECHA = "Fecha de Venta"
        DIST  = "Distribuidor"

        if df.empty or DIST not in df.columns:
            return []
        if "Anio" not in df.columns or "Mes" not in df.columns:
            return []

        # ── 1. Columna de periodo mensual ─────────────────────────────
        df = df.copy()
        df["_anio_int"] = pd.to_numeric(df["Anio"], errors="coerce")
        df["_mes_int"]  = pd.to_numeric(df["Mes"],  errors="coerce")
        df = df.dropna(subset=["_anio_int", "_mes_int"])
        df["_periodo"] = (
            df["_anio_int"].astype(int).astype(str)
            + "-"
            + df["_mes_int"].astype(int).apply(lambda m: str(m).zfill(2))
        )

        # ── 2. Serie ventas: distribuidor × periodo ───────────────────
        ventas_dp = (
            df.groupby([DIST, "_periodo"])[CV]
            .sum()
            .reset_index()
            .rename(columns={CV: "ventas"})
        )

        # ── 3. Serie clientes activos: distribuidor × periodo ─────────
        df_act = df[df[CV] > 0]
        clientes_dp = (
            df_act.groupby([DIST, "_periodo"])[CLI]
            .nunique()
            .reset_index()
            .rename(columns={CLI: "n_clientes"})
        )

        # ── 4. Amplitud media por distribuidor ────────────────────────
        sku_col = next(
            (c for c in ["Código SKU Parawa", "Descripción SKU Parawa", "Descripcion SKU Parawa"]
             if c in df.columns),
            None,
        )
        if sku_col:
            amp_por_dist = (
                df_act.groupby([DIST, CLI])[sku_col]
                .nunique()
                .reset_index()
                .rename(columns={sku_col: "n_skus"})
                .groupby(DIST)["n_skus"]
                .mean()
            )
        else:
            amp_por_dist = pd.Series(dtype=float)

        # ── 5. Región por distribuidor ────────────────────────────────
        if "Region_Distribuidor" in df.columns:
            region_map = df.groupby(DIST)["Region_Distribuidor"].first()
        else:
            region_map = pd.Series(dtype=str)

        # ── 6. Escenarios por distribuidor ────────────────────────────
        distribuidores = sorted(df[DIST].dropna().unique().tolist())
        resultados = []

        for dist in distribuidores:

            # Ventas serie (ordenada)
            v = (
                ventas_dp[ventas_dp[DIST] == dist]
                .sort_values("_periodo")
                .set_index("_periodo")["ventas"]
            )
            if v.empty:
                continue

            n_meses     = len(v)
            promedio_6m = float(v.tail(6).mean())
            tendencia_mom_ventas = self._mom_promedio(v.tail(4))

            # Clientes activos serie (ordenada)
            c = (
                clientes_dp[clientes_dp[DIST] == dist]
                .sort_values("_periodo")
                .set_index("_periodo")["n_clientes"]
            )
            clientes_activos         = int(c.iloc[-1]) if not c.empty else 0
            tendencia_mom_activacion = self._mom_promedio(c.tail(4)) if not c.empty else 0.0

            # Promedio por cliente (del último período promedio)
            promedio_por_cliente = (promedio_6m / clientes_activos) if clientes_activos > 0 else 0.0

            # Amplitud
            amplitud_actual = float(amp_por_dist.get(dist, 0.0))

            # Región
            region = str(region_map.get(dist, "Sin Región"))

            # ── Escenario 1 — Volumen ─────────────────────────────────
            t1 = max(tendencia_mom_ventas, -10.0)
            meta_esc1 = int(round(promedio_6m * (1 + t1 / 100)))
            just_esc1 = (
                f"Volumen: {promedio_6m:,.0f} und × tendencia {tendencia_mom_ventas:+.1f}%"
            )

            # ── Escenario 2 — Activación ──────────────────────────────
            t2 = max(tendencia_mom_activacion, -10.0)
            act_proyectada = clientes_activos * (1 + t2 / 100)
            meta_esc2 = int(round(act_proyectada * promedio_por_cliente))
            just_esc2 = (
                f"Activación: {clientes_activos} → {act_proyectada:.0f} clientes"
                f" × {promedio_por_cliente:,.0f} und/cliente"
            )

            # ── Escenario 3 — Amplitud ────────────────────────────────
            if amplitud_actual > 0:
                base     = clientes_activos * promedio_por_cliente
                amp_p1   = base * ((amplitud_actual + 1.0) / amplitud_actual)
                amp_p15  = base * ((amplitud_actual + 1.5) / amplitud_actual)
                meta_esc3 = int(round((amp_p1 + amp_p15) / 2))
                just_esc3 = (
                    f"Amplitud: {amplitud_actual:.1f} SKUs → "
                    f"+1 SKU={amp_p1:,.0f} | +1.5 SKU={amp_p15:,.0f}"
                )
            else:
                meta_esc3 = meta_esc1
                just_esc3 = "Amplitud sin datos — usando escenario Volumen como fallback"

            # ── Confianza ─────────────────────────────────────────────
            if n_meses >= 6:
                nivel_confianza = "alto"
            elif n_meses >= 3:
                nivel_confianza = "medio"
            else:
                nivel_confianza = "bajo"

            resultados.append({
                "distribuidor":             dist,
                "region":                   region,
                "promedio_6m":              round(promedio_6m, 1),
                "tendencia_mom_ventas":     round(tendencia_mom_ventas, 2),
                "clientes_activos":         clientes_activos,
                "tendencia_mom_activacion": round(tendencia_mom_activacion, 2),
                "promedio_por_cliente":     round(promedio_por_cliente, 1),
                "amplitud_actual":          round(amplitud_actual, 2),
                "meta_escenario1":          meta_esc1,
                "just_escenario1":          just_esc1,
                "meta_escenario2":          meta_esc2,
                "just_escenario2":          just_esc2,
                "meta_escenario3":          meta_esc3,
                "just_escenario3":          just_esc3,
                "nivel_confianza":          nivel_confianza,
            })

        return resultados
