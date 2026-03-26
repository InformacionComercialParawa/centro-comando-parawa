import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import json
import requests
import base64
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path

from agents.base_agent import AgentResult
from agents.sales_analyst import SalesAnalystAgent
from agents.regional_analyst import RegionalAnalystAgent
from agents.national_analyst import NationalAnalystAgent
from agents.goals_engine import GoalsEngine
from agents.kpi_engine import (
    calcular_cobertura,
    calcular_frecuencia,
    calcular_frecuencia_compra,
    calcular_amplitud,
    calcular_volumen_por_cliente,
    calcular_participacion,
)

# =================================================================
# CONFIGURACIÓN POWER BI
# =================================================================

POWER_BI_CONFIG_PATH = Path(__file__).parent / "data" / "power_bi_reports.json"


def cargar_reportes_power_bi():
    """Carga la lista de reportes desde el archivo JSON."""
    if POWER_BI_CONFIG_PATH.exists():
        with open(POWER_BI_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("reportes", [])
    return []


def agregar_reporte(titulo, descripcion, link):
    """Agrega un nuevo reporte a la librería."""
    config = {"reportes": []}
    if POWER_BI_CONFIG_PATH.exists():
        with open(POWER_BI_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    nuevo_id = "reporte_" + str(len(config["reportes"]) + 1)
    config["reportes"].append({"id": nuevo_id, "titulo": titulo, "descripcion": descripcion, "link": link})
    POWER_BI_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(POWER_BI_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# =================================================================
# CONFIGURACION DE PAGINA
# =================================================================
st.set_page_config(
    page_title="Centro de Comando | Parawa",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =================================================================
# CARGA DE LOGO
# =================================================================

def load_logo_b64(filename: str) -> str:
    assets_dir = Path(__file__).parent / "assets"
    ruta = assets_dir / filename
    if ruta.exists():
        with open(ruta, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

LOGO_SQUARE_B64 = load_logo_b64("Logo_Parawa.png")

# =================================================================
# CSS MINIMO — solo lo esencial
# =================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito+Sans:wght@400;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito Sans', sans-serif !important;
}
.stApp { background-color: #F4F6F8 !important; }
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar siempre visible — ocultar botón de colapso */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] button,
button[aria-label="Collapse sidebar"],
button[title="Collapse sidebar"] {
    display: none !important;
}
.block-container { padding-top: 0.5rem !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: white !important;
    border-radius: 8px !important;
    padding: 6px !important;
    border: 1px solid #E8ECF0 !important;
    gap: 6px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px !important;
    font-weight: 900 !important;
    font-size: 18px !important;
    padding: 8px 18px !important;
    color: #374151 !important;
}
.stTabs [aria-selected="true"] {
    background: #00ACC1 !important;
    color: white !important;
    font-weight: 900 !important;
    font-size: 18px !important;
}

/* Botones primarios */
.stButton button[kind="primary"],
[data-testid="stFormSubmitButton"] button {
    background: #00ACC1 !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 6px rgba(0,172,193,0.3) !important;
}
.stButton button[kind="primary"]:hover,
[data-testid="stFormSubmitButton"] button:hover {
    background: #0097A7 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E8ECF0 !important;
}
[data-testid="stSidebar"] * { color: #374151 !important; }
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #1A1A2E !important;
    font-size: 11px !important;
    font-weight: 900 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-bottom: 2px solid #00ACC1 !important;
    padding-bottom: 6px !important;
}
[data-testid="stSidebar"] .stMultiSelect > label,
[data-testid="stSidebar"] .stRadio > label,
[data-testid="stSidebar"] .stSelectbox > label {
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #6B7280 !important;
}

/* Métricas nativas (usadas en proyección y regiones) */
[data-testid="stMetricValue"] {
    font-weight: 900 !important;
    font-size: 1.8rem !important;
}
</style>
""", unsafe_allow_html=True)


# =================================================================
# HELPERS DE GRÁFICOS — Plotly estilo Parawa
# =================================================================

_PARAWA_COLORS = ["#00ACC1", "#F57C00", "#1A237E", "#26A69A"]


def _plotly_linea(df_serie, x_col, y_cols, titulo, colores=None):
    """Gráfico de línea estilo Parawa: marcadores circulares, valores encima, fondo blanco."""
    if colores is None:
        colores = _PARAWA_COLORS
    fig = go.Figure()
    for i, col in enumerate(y_cols):
        color = colores[i % len(colores)]
        y_vals = df_serie[col].tolist()
        text_vals = [
            "{:,.0f}".format(v) if v is not None and not pd.isna(v) else ""
            for v in y_vals
        ]
        fig.add_trace(go.Scatter(
            x=df_serie[x_col],
            y=y_vals,
            name=col,
            mode="lines+markers+text",
            line=dict(color=color, width=2.5),
            marker=dict(size=7, color=color),
            text=text_vals,
            textposition="top center",
            textfont=dict(size=18, family="Arial"),
            connectgaps=False,
        ))
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=21, family="Arial", color="#212121"), x=0),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=18, color="#212121", family="Arial Black"), title=dict(font=dict(size=19, color="#212121", family="Arial Black"))),
        yaxis=dict(showgrid=True, gridcolor="#F0F0F0", zeroline=False, tickfont=dict(size=18, color="#212121", family="Arial Black"), title=dict(font=dict(size=19, color="#212121", family="Arial Black"))),
        font=dict(size=21, family="Arial"),
    )
    return fig


def _plotly_barras_linea(df_serie, x_col, bar_col, line_col, titulo):
    """Combo chart barras grises (fondo) + línea naranja con marcadores y valores."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_serie[x_col],
        y=df_serie[bar_col],
        name=bar_col,
        marker_color="#E0E0E0",
        opacity=0.85,
    ))
    line_vals = df_serie[line_col].tolist()
    text_vals = [
        "{:.1f}%".format(v) if v is not None and not pd.isna(v) else ""
        for v in line_vals
    ]
    fig.add_trace(go.Scatter(
        x=df_serie[x_col],
        y=line_vals,
        name=line_col,
        mode="lines+markers+text",
        line=dict(color="#F57C00", width=2.5),
        marker=dict(size=7, color="#F57C00"),
        text=text_vals,
        textposition="top center",
        textfont=dict(size=12, family="Arial"),
        yaxis="y2",
    ))
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=14, family="Arial", color="#212121"), x=0),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(family="Arial")),
        yaxis=dict(showgrid=True, gridcolor="#F0F0F0", zeroline=False, tickfont=dict(family="Arial")),
        yaxis2=dict(overlaying="y", side="right", showgrid=False, zeroline=False),
        font=dict(family="Arial"),
    )
    return fig


# =================================================================
# HELPERS VISUALES — Cards, títulos y header
# =================================================================

def render_section_title(titulo, subtitulo=""):
    """Título de sección uppercase bold con subtítulo opcional."""
    html = (
        "<div style='margin:20px 0 12px 0;'>"
        "<div style='font-size:13px;font-weight:900;color:#1A1A2E;"
        "text-transform:uppercase;letter-spacing:0.05em;'>"
        + titulo +
        "</div>"
    )
    if subtitulo:
        html += (
            "<div style='font-size:11px;color:#8A94A6;margin-top:2px;'>"
            + subtitulo +
            "</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _indicator_card_html(label, valor, color, emoji=""):
    """Card de indicador general con borde izquierdo de color y emoji opcional."""
    emoji_html = (
        "<div style='font-size:22px;margin-bottom:6px;'>" + emoji + "</div>"
        if emoji else ""
    )
    return (
        "<div style='background:white;border-radius:12px;padding:18px 22px;"
        "border-left:4px solid " + color + ";"
        "box-shadow:0 1px 4px rgba(0,0,0,0.06);'>"
        + emoji_html +
        "<div style='font-size:14px;font-weight:700;color:#8A94A6;"
        "text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;'>"
        + label +
        "</div>"
        "<div style='font-size:38px;font-weight:900;color:" + color + ";line-height:1;'>"
        + valor +
        "</div>"
        "</div>"
    )


def _kpi_card_html(label, valor, delta_str, inverse=False, emoji="", color="#00ACC1"):
    """Card KPI estratégico — mismo estilo que Indicadores Generales con delta."""
    emoji_html = (
        "<div style='font-size:22px;margin-bottom:6px;'>" + emoji + "</div>"
        if emoji else ""
    )
    delta_html = ""
    if delta_str:
        is_pos = delta_str.startswith("+")
        if inverse:
            is_pos = not is_pos
        dc = "#43A047" if is_pos else "#E53935"
        delta_html = (
            "<div style='font-size:16px;font-weight:700;margin-top:6px;color:"
            + dc + ";'>" + delta_str + "</div>"
        )
    return (
        "<div style='background:white;border-radius:12px;padding:18px 22px;"
        "border-left:4px solid " + color + ";"
        "box-shadow:0 1px 4px rgba(0,0,0,0.06);'>"
        + emoji_html +
        "<div style='font-size:14px;font-weight:700;color:#8A94A6;"
        "text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;'>"
        + label +
        "</div>"
        "<div style='font-size:38px;font-weight:900;color:" + color + ";line-height:1;'>"
        + valor +
        "</div>"
        + delta_html +
        "</div>"
    )


def render_page_header(title, subtitle, username, region):
    """Header corporativo: título, subtítulo y badges de usuario y región."""
    user_badge = (
        "<div style='background:#E0F7FA;color:#00838F;font-size:11px;"
        "font-weight:700;padding:4px 12px;border-radius:20px;'>"
        + username + "</div>"
    )
    region_badge = (
        "<div style='background:#E8EAF6;color:#3949AB;font-size:11px;"
        "font-weight:700;padding:4px 12px;border-radius:20px;'>"
        + region + "</div>"
    )

    html = (
        "<div style='background:white;border-bottom:1px solid #E8ECF0;"
        "padding:14px 24px;display:flex;align-items:center;"
        "justify-content:space-between;margin-bottom:16px;"
        "box-shadow:0 1px 3px rgba(0,0,0,0.04);'>"
        "<div>"
        "<div style='font-size:32px;font-weight:900;color:#1A1A2E;"
        "letter-spacing:-0.5px;line-height:1.1;'>" + title + "</div>"
        "<div style='font-size:13px;color:#8A94A6;font-weight:600;"
        "margin-top:4px;'>" + subtitle + "</div>"
        "</div>"
        "<div style='display:flex;gap:8px;align-items:center;'>"
        + user_badge + region_badge +
        "</div></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


# =================================================================
# MAPEO REGION → PREFIJOS
# =================================================================
REGION_PREFIJOS = {
    "Capital"          : ["CARACAS"],
    "Centro Occidente" : ["CENTRO OCCIDENTE"],
    "Centro"           : ["CENTRO"],
    "Los Andes"        : ["LOS ANDES"],
    "Occidente"        : ["OCCIDENTE"],
    "Oriente"          : ["ORIENTE"],
}


def extract_region_from_distributor(distribuidor: str) -> str:
    nombre_upper = str(distribuidor).upper()
    for region, prefijos in REGION_PREFIJOS.items():
        for prefijo in prefijos:
            if nombre_upper.startswith(prefijo):
                return region
    return "Sin Región"


# =================================================================
# LOGIN + RBAC
# =================================================================

def check_login() -> bool:
    if st.session_state.get("authenticated"):
        return True

    col_l, col_c, col_r = st.columns([1, 1.2, 1])
    with col_c:
        st.markdown("<br><br>", unsafe_allow_html=True)

        with st.container(border=False):
            st.markdown("<br>", unsafe_allow_html=True)

            if LOGO_SQUARE_B64:
                st.markdown(
                    "<div style='text-align:center;margin-bottom:1rem;'>"
                    "<div style='background:linear-gradient(135deg,#006064,#00ACC1);"
                    "width:90px;height:90px;border-radius:20px;"
                    "display:inline-flex;align-items:center;justify-content:center;"
                    "box-shadow:0 4px 16px rgba(0,131,143,0.3);padding:8px;'>"
                    "<img src='data:image/png;base64," + LOGO_SQUARE_B64 + "' height='70' />"
                    "</div></div>",
                    unsafe_allow_html=True
                )

            st.markdown("### Centro de Comando")
            st.caption("Plataforma de Inteligencia Comercial · Parawa")
            st.divider()

            with st.form("login_form"):
                username  = st.text_input("**Usuario**", placeholder="Ingresa tu usuario")
                password  = st.text_input("**Contraseña**", type="password", placeholder="Ingresa tu contraseña")
                submitted = st.form_submit_button("Iniciar Sesión", use_container_width=True, type="primary")

                if submitted:
                    passwords = st.secrets.get("passwords", {})
                    if username in passwords and passwords[username] == password:
                        st.session_state["authenticated"] = True
                        st.session_state["username"] = username
                        st.rerun()
                    else:
                        st.error("Usuario o contraseña incorrectos.")

            st.markdown("<br>", unsafe_allow_html=True)

        st.caption("© 2026 Parawa · Acceso restringido")

    return False


def get_user_region(username: str) -> str:
    try:
        access = st.secrets.get("access", {}).get(username, {})
        region = access.get("region", "")
        if region:
            return region
        distribuidores = access.get("distribuidores", [])
        if distribuidores:
            return f"Asesor ({len(distribuidores)} dist.)"
        return "Sin Región"
    except Exception:
        return "Sin Región"


def filter_by_rbac(df: pd.DataFrame, username: str) -> pd.DataFrame:
    try:
        access = st.secrets.get("access", {}).get(username, {})
    except Exception:
        return pd.DataFrame()

    region = access.get("region", "")
    if region == "TODOS":
        return df
    if region:
        if "Region_Distribuidor" not in df.columns:
            return df
        if isinstance(region, list):
            df_filtrado = df[df["Region_Distribuidor"].isin(region)]
        else:
            df_filtrado = df[df["Region_Distribuidor"] == region]
        if df_filtrado.empty:
            st.warning(f"No se encontraron distribuidores para la región '{region}'.")
        return df_filtrado

    distribuidores = access.get("distribuidores", [])
    if distribuidores:
        df_filtrado = df[df["Distribuidor"].isin(distribuidores)]
        if df_filtrado.empty:
            st.warning("No se encontraron datos para los distribuidores asignados.")
        return df_filtrado

    st.warning(f"El usuario '{username}' no tiene acceso configurado.")
    return pd.DataFrame()


# =================================================================
# CARGA DE DATOS
# =================================================================

@st.cache_data(ttl=600, show_spinner=False)
def load_all_parquets(folder_path: str) -> pd.DataFrame:
    all_dfs = []
    if not os.path.exists(folder_path):
        st.error(f"La carpeta no existe: {folder_path}")
        return pd.DataFrame()

    for filename in os.listdir(folder_path):
        if not filename.endswith(".parquet"):
            continue
        filepath = os.path.join(folder_path, filename)
        try:
            df = pd.read_parquet(filepath)
            name_clean = filename.replace("Ventas_","").replace("_Consolidado.parquet","")
            name_clean = name_clean.replace("_-_"," - ").replace("_"," ")
            name_clean = re.sub(r'\s+',' ',name_clean).strip()
            df["Distribuidor"] = name_clean
            all_dfs.append(df)
        except Exception as e:
            st.warning(f"Error leyendo {filename}: {e}")

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["Region_Distribuidor"] = combined["Distribuidor"].apply(extract_region_from_distributor)

    if "Fecha de Venta" in combined.columns:
        combined["Fecha de Venta"] = pd.to_datetime(combined["Fecha de Venta"], errors="coerce")
        combined["Anio"] = combined["Fecha de Venta"].dt.year.astype("Int64").astype(str)
        combined["Mes"]  = combined["Fecha de Venta"].dt.month.astype("Int64").astype(str)
        meses_nombre = {
            "1":"01-Ene","2":"02-Feb","3":"03-Mar","4":"04-Abr",
            "5":"05-May","6":"06-Jun","7":"07-Jul","8":"08-Ago",
            "9":"09-Sep","10":"10-Oct","11":"11-Nov","12":"12-Dic",
        }
        combined["Mes_Nombre"] = combined["Mes"].map(meses_nombre).fillna(combined["Mes"])

    if "Total de Unidades Vendidas (und)" in combined.columns:
        combined["Total de Unidades Vendidas (und)"] = pd.to_numeric(
            combined["Total de Unidades Vendidas (und)"], errors="coerce"
        ).fillna(0)

    return combined


@st.cache_data(ttl=600, show_spinner=False)
def load_maestro(maestros_folder: str) -> pd.DataFrame:
    ruta = Path(maestros_folder) / "maestro_clientes.parquet"
    if not ruta.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(ruta)
    except Exception as e:
        st.warning(f"No se pudo cargar el maestro: {e}")
        return pd.DataFrame()


def normalizar_nombre(texto) -> str:
    if pd.isna(texto):
        return ""
    return (
        str(texto).upper().strip()
        .replace("\t","").replace(".","").replace(",","")
        .replace("-"," ").replace("  "," ")
    )


def enriquecer_con_maestro(df: pd.DataFrame, maestro: pd.DataFrame) -> pd.DataFrame:
    if maestro.empty:
        df["Segmento Parawa"] = None
        df["Canal Parawa"]    = None
        df["Regional Parawa"] = None
        df["Es_Ponderado"]    = False
        return df

    df["_Nombre_Norm"]      = df["Descripción Cliente"].apply(normalizar_nombre)
    maestro["_Nombre_Norm"] = maestro["Nombre_Parquet"].apply(normalizar_nombre)

    maestro_slim = maestro[[
        "_Nombre_Norm","Perfil del PDV","Canal del PDV","Regional - Obligatorio",
    ]].drop_duplicates(subset=["_Nombre_Norm"]).rename(columns={
        "Perfil del PDV"         : "Segmento Parawa",
        "Canal del PDV"          : "Canal Parawa",
        "Regional - Obligatorio" : "Regional Parawa",
    })

    df = df.merge(maestro_slim, on="_Nombre_Norm", how="left")
    df["Es_Ponderado"] = df["Segmento Parawa"].notna()
    df.drop(columns=["_Nombre_Norm"], inplace=True)
    return df


@st.cache_data(ttl=600, show_spinner=False)
def load_maestro_skus(maestros_folder: str) -> pd.DataFrame:
    ruta = Path(maestros_folder) / "maestro_skus.parquet"
    if not ruta.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(ruta)
    except Exception as e:
        st.warning(f"No se pudo cargar el maestro de SKUs: {e}")
        return pd.DataFrame()


def build_drive_service():
    """
    Busca el JSON de cuenta de servicio en .streamlit/*.json.
    Retorna el servicio de Drive v3, o None si no hay credenciales o librería.
    """
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build as _gdrive_build
    except ImportError:
        return None

    streamlit_dir = Path(__file__).parent / ".streamlit"
    json_files = list(streamlit_dir.glob("*.json"))
    if not json_files:
        return None

    try:
        creds = service_account.Credentials.from_service_account_file(
            str(json_files[0]),
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
        return _gdrive_build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_consolidated(procesada_folder: str):
    """
    Lee el Parquet consolidado. Intenta Google Drive primero;
    fallback a disco local si Drive no está disponible o falla.
    Retorna (df, fuente) donde fuente es un string de caption o None.
    En modo demo lee sellout_demo_consolidado.parquet y retorna marca especial.
    """
    modo = st.secrets.get("app", {}).get("modo", "produccion")

    # ── Modo Demo ────────────────────────────────────────────────────────────
    if modo == "demo":
        ruta = Path(procesada_folder) / "sellout_demo_consolidado.parquet"
        if not ruta.exists():
            return pd.DataFrame(), None
        try:
            return pd.read_parquet(ruta), "⚠️ MODO DEMO — Data de ejemplo ficticia"
        except Exception:
            return pd.DataFrame(), None

    # ── Intento 1: Google Drive ──────────────────────────────────────────────
    try:
        service = build_drive_service()
        if service is not None:
            folder_id = st.secrets.get("drive", {}).get("folder_id", "")
            if folder_id:
                from agents.drive_loader import load_parquet_from_drive
                df = load_parquet_from_drive(
                    service, "sellout_consolidado.parquet", folder_id
                )
                return df, "📡 Datos cargados desde Google Drive"
    except Exception:
        pass  # sin credenciales o Drive inaccesible → fallback local

    # ── Fallback: disco local ────────────────────────────────────────────────
    ruta = Path(procesada_folder) / "sellout_consolidado.parquet"
    if not ruta.exists():
        return pd.DataFrame(), None
    try:
        return pd.read_parquet(ruta), "💾 Datos cargados desde disco local"
    except Exception:
        return pd.DataFrame(), None


# =================================================================
# GESTIÓN DE METAS
# =================================================================

def _get_metas_path() -> Path:
    return Path(__file__).parent / "data" / "metas.json"


def load_metas() -> dict:
    path = _get_metas_path()
    if not path.exists():
        return {}
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return {}


def save_metas(metas: dict):
    path = _get_metas_path()
    path.parent.mkdir(exist_ok=True)
    json.dump(metas, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def enriquecer_con_maestro_skus(df: pd.DataFrame, maestro_skus: pd.DataFrame) -> pd.DataFrame:
    SKU_COLS = ["Nombre_SKU", "Categoria_Parawa", "Linea_Parawa", "Marca_Parawa", "Agrupacion_Parawa", "Status_SKU"]
    join_key_df = "Código SKU Parawa"

    if maestro_skus.empty or join_key_df not in df.columns or "Codigo_SKU" not in maestro_skus.columns:
        for col in SKU_COLS:
            if col not in df.columns:
                df[col] = None
        return df

    cols_to_merge = [c for c in SKU_COLS if c not in df.columns]
    if not cols_to_merge:
        return df

    maestro_slim = (
        maestro_skus[["Codigo_SKU"] + cols_to_merge]
        .drop_duplicates(subset=["Codigo_SKU"])
    )

    df = df.merge(
        maestro_slim,
        left_on=join_key_df,
        right_on="Codigo_SKU",
        how="left",
    )
    df.drop(columns=["Codigo_SKU"], inplace=True, errors="ignore")
    return df


# =================================================================
# CONTEXTO DEL USUARIO
# =================================================================

def build_user_context(username: str, df: pd.DataFrame, anonimizar: bool = False) -> str:
    CV = "Total de Unidades Vendidas (und)"

    # Helper: reemplaza nombres reales por códigos en texto de session_state
    _smap = st.session_state.get("sanitize_map", {}) if anonimizar else {}

    def _anon(texto: str) -> str:
        if not _smap:
            return texto
        for pool in _smap.values():
            for real in sorted(pool.keys(), key=len, reverse=True):
                if real in texto:
                    texto = texto.replace(real, pool[real])
        return texto

    lines = ["=== CONTEXTO DEL USUARIO ===", f"Usuario: {username}"]
    region = get_user_region(username)
    lines.append(f"Región asignada: {_anon(region)}")
    lines.append(f"Distribuidores visibles: {df['Distribuidor'].nunique()}")
    if "Region_Distribuidor" in df.columns:
        lines.append(f"Regiones en vista: {', '.join(sorted(df['Region_Distribuidor'].dropna().unique().tolist()))}")
    lines += ["","=== DATOS ACTUALES ==="]
    lines.append(f"Registros: {len(df):,}")
    lines.append(f"Unidades totales: {df[CV].sum():,.0f}")
    if "Anio" in df.columns:
        lines.append(f"Años: {', '.join(sorted(df['Anio'].dropna().unique().tolist()))}")
    if "Es_Ponderado" in df.columns:
        tv = df[CV].sum()
        pv = df[df["Es_Ponderado"]][CV].sum()
        lines.append(f"Cobertura ponderados: {(pv/tv*100) if tv>0 else 0:.1f}%")

    agent_result = st.session_state.get("agent_result")
    if agent_result and agent_result.get("status") != "error":
        lines += ["","=== ULTIMO ANALISIS AGENTE 1 ==="]
        for ins in agent_result.get("insights",[])[:5]:
            lines.append(f"  - {_anon(ins)}")
        tendencia = agent_result.get("metrics",{}).get("tendencia_mensual",{})
        if tendencia.get("disponible") and tendencia.get("tendencia"):
            for t in tendencia["tendencia"][-3:]:
                mom = f" ({t.get('mom_pct',0):+.1f}% MoM)" if t.get("mom_pct") is not None else ""
                lines.append(f"  - {t['periodo']}: {t['unidades']:,.0f} und{mom}")
    else:
        lines += ["","=== ULTIMO ANALISIS AGENTE 1 ===","Aún no ejecutado."]

    regional_result = st.session_state.get("regional_result")
    if regional_result and regional_result.get("status") == "success":
        lines += ["","=== ANALISIS REGIONAL ==="]
        region_name = _anon(regional_result.get("metrics", {}).get("region", "N/A"))
        lines.append(f"Región: {region_name}")
        k = regional_result.get("metrics", {}).get("kpis_generales", {})
        if k:
            lines.append(f"  Unidades totales región: {k.get('unidades_totales', 0):,.0f}")
            lines.append(f"  Distribuidores activos: {k.get('distribuidores_activos', 'N/A')}")
        t_reg = regional_result.get("metrics", {}).get("tendencia_mensual", {})
        if t_reg.get("disponible"):
            lines.append(f"  Tendencia regional: {t_reg.get('direccion', 'N/A')}")
        for ins in regional_result.get("insights", [])[:3]:
            lines.append(f"  - {_anon(ins)}")

    contexto_manual = st.session_state.get("contexto_regional","").strip()
    if contexto_manual:
        lines += ["","=== CONTEXTO ADICIONAL ===", contexto_manual,
                  "IMPORTANTE: Considera este contexto en el análisis."]

    # KPIs estratégicos calculados en tiempo real (últimos 12 meses)
    try:
        temporalidad = st.session_state.get("temporalidad", "Mensual")
        df_kpi, kpi_corte, kpi_max = _filtrar_ultimos_12_meses(df)
        periodo_label = f" | {kpi_corte} → {kpi_max}" if kpi_corte else ""
        lines += ["", "=== KPIs ESTRATÉGICOS ACTUALES ===",
                  f"Temporalidad: {temporalidad}{periodo_label}"]

        def _fmt_kpi(nombre, res, sufijo="", decimales=1):
            actual = res.get("actual")
            anterior = res.get("anterior")
            var = res.get("variacion_pct")
            if actual is None:
                return None
            s = f"  {nombre}: {actual:,.{decimales}f}{sufijo}"
            if anterior is not None and var is not None:
                s += f" (vs anterior: {anterior:,.{decimales}f}{sufijo}, {var:+.1f}pp)"
            return s

        try:
            cob = calcular_cobertura(df_kpi, temporalidad)
            pond = cob.get("cobertura_ponderados", {})
            total = cob.get("cobertura_total", {})
            l = _fmt_kpi("Cobertura Ponderados", pond, "%")
            if l: lines.append(l)
            l = _fmt_kpi("Cobertura Total", total, "%")
            if l: lines.append(l)
        except Exception:
            pass

        try:
            fc = calcular_frecuencia_compra(df_kpi, temporalidad)
            l = _fmt_kpi("Frecuencia de Compra", fc, " compras/cliente")
            if l: lines.append(l)
        except Exception:
            pass

        try:
            frec = calcular_frecuencia(df_kpi, temporalidad)
            l = _fmt_kpi("Días entre Compras", frec, " días")
            if l: lines.append(l)
        except Exception:
            pass

        try:
            amp = calcular_amplitud(df_kpi, temporalidad)
            l = _fmt_kpi("Amplitud", amp, " SKUs/cliente", decimales=2)
            if l: lines.append(l)
        except Exception:
            pass

        try:
            vol = calcular_volumen_por_cliente(df_kpi, temporalidad)
            l = _fmt_kpi("Volumen por Cliente", vol, " und/cliente")
            if l: lines.append(l)
        except Exception:
            pass

    except Exception:
        pass

    lines.append("=== FIN CONTEXTO ===")
    return "\n".join(lines)


# =================================================================
# FILTROS EN CASCADA — Sidebar
# =================================================================

def render_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    _opts_temp = ["Mensual", "Bimestral", "Trimestral"]
    _default_temp = st.session_state.get("temporalidad", "Mensual")
    _idx_temp = _opts_temp.index(_default_temp) if _default_temp in _opts_temp else 0
    temporalidad = st.sidebar.radio(
        "⏱️ Temporalidad",
        _opts_temp,
        index=_idx_temp,
        horizontal=True,
    )
    st.session_state["temporalidad"] = temporalidad
    st.sidebar.divider()

    # Los filtros SIEMPRE se muestran, incluso en chat
    # Permite cambiar región/distribuidor y re-preguntar sin perder contexto
    st.session_state.pop("preserve_filters", None)

    st.sidebar.markdown("### 🎯 Filtros")

    # ── Distribuidor — aplica inmediatamente ──
    distribuidores = sorted(df["Distribuidor"].dropna().unique().tolist())
    sel_dist = st.sidebar.multiselect("Distribuidor", distribuidores, default=[], key="filt_dist")
    filtered_dist = df[df["Distribuidor"].isin(sel_dist)].copy() if sel_dist else df.copy()

    # ── Filtros pendientes — opciones en cascada, solo se aplican al presionar el botón ──
    filtered_opts = filtered_dist.copy()

    if "Anio" in filtered_opts.columns:
        anios = sorted([a for a in filtered_opts["Anio"].dropna().unique().tolist() if a != "<NA>" and str(a) != "nan"])
        st.sidebar.multiselect("Año", anios, default=[], key="pending_anio")
        _pending_anio = [a for a in st.session_state.get("pending_anio", []) if a in anios]
        if _pending_anio:
            filtered_opts = filtered_opts[filtered_opts["Anio"].isin(_pending_anio)]

    if "Mes_Nombre" in filtered_opts.columns:
        meses = sorted(filtered_opts["Mes_Nombre"].dropna().unique().tolist())
        st.sidebar.multiselect("Mes", meses, default=[], key="pending_mes")
        _pending_mes = [m for m in st.session_state.get("pending_mes", []) if m in meses]
        if _pending_mes:
            filtered_opts = filtered_opts[filtered_opts["Mes_Nombre"].isin(_pending_mes)]

    if "Sucursal Aliado" in filtered_opts.columns:
        sucursales = sorted(filtered_opts["Sucursal Aliado"].dropna().unique().tolist())
        if sucursales:
            st.sidebar.multiselect("Sucursal", sucursales, default=[], key="pending_suc")
            _pending_suc = [s for s in st.session_state.get("pending_suc", []) if s in sucursales]
            if _pending_suc:
                filtered_opts = filtered_opts[filtered_opts["Sucursal Aliado"].isin(_pending_suc)]

    st.sidebar.divider()
    fuente_segmento = st.sidebar.radio(
        "Segmentación",
        ["Distribuidor (todos)", "Parawa (ponderados)"],
        help="Distribuidor: segmento original | Parawa: segmento correcto"
    )
    st.session_state["fuente_segmento"] = fuente_segmento

    if fuente_segmento == "Parawa (ponderados)":
        _opts_seg_df = filtered_opts[filtered_opts["Es_Ponderado"] == True] if "Es_Ponderado" in filtered_opts.columns else filtered_opts
        if "Segmento Parawa" in _opts_seg_df.columns:
            segmentos = sorted(_opts_seg_df["Segmento Parawa"].dropna().unique().tolist())
            st.sidebar.multiselect("Segmento Parawa", segmentos, default=[], key="pending_seg")
    else:
        if "Segmento Cliente" in filtered_opts.columns:
            segmentos = sorted(filtered_opts["Segmento Cliente"].dropna().unique().tolist())
            st.sidebar.multiselect("Segmento", segmentos, default=[], key="pending_seg")

    # Aplicar segmentación a opts para que los SKU filters sean consistentes
    if fuente_segmento == "Parawa (ponderados)" and "Es_Ponderado" in filtered_opts.columns:
        filtered_opts = filtered_opts[filtered_opts["Es_Ponderado"] == True]

    st.sidebar.divider()

    for _col, _label, _pk in [
        ("Marca_Parawa",      "Marca",      "pending_marca"),
        ("Categoria_Parawa",  "Categoría",  "pending_cat"),
        ("Linea_Parawa",      "Línea",      "pending_linea"),
        ("Agrupacion_Parawa", "Agrupación", "pending_agrup"),
        ("Nombre_SKU",        "SKU",        "pending_sku"),
    ]:
        if _col in filtered_opts.columns:
            _opts = sorted(filtered_opts[_col].dropna().unique().tolist())
            if _opts:
                st.sidebar.multiselect(_label, _opts, default=[], key=_pk)
                _pending_val = [v for v in st.session_state.get(_pk, []) if v in _opts]
                if _pending_val:
                    filtered_opts = filtered_opts[filtered_opts[_col].isin(_pending_val)]

    # ── Botón Aplicar Filtros ──
    st.sidebar.divider()
    if st.sidebar.button("🔍 Aplicar Filtros", type="primary", use_container_width=True):
        st.session_state["tab_activo"] = "dashboard"
        st.session_state["applied_filters"] = {
            "anio":            list(st.session_state.get("pending_anio", [])),
            "mes":             list(st.session_state.get("pending_mes", [])),
            "suc":             list(st.session_state.get("pending_suc", [])),
            "seg":             list(st.session_state.get("pending_seg", [])),
            "fuente_segmento": fuente_segmento,
            "marca":           list(st.session_state.get("pending_marca", [])),
            "cat":             list(st.session_state.get("pending_cat", [])),
            "linea":           list(st.session_state.get("pending_linea", [])),
            "agrup":           list(st.session_state.get("pending_agrup", [])),
            "sku":             list(st.session_state.get("pending_sku", [])),
        }
        st.session_state["_apply_just_pressed"] = True

    # ── Construir df resultado con filtros aplicados ──
    applied = st.session_state.get("applied_filters", {})
    result = filtered_dist.copy()

    _fuente_app = applied.get("fuente_segmento", fuente_segmento)
    if _fuente_app == "Parawa (ponderados)" and "Es_Ponderado" in result.columns:
        result = result[result["Es_Ponderado"] == True]

    if "Anio" in result.columns and applied.get("anio"):
        result = result[result["Anio"].isin(applied["anio"])]

    if "Mes_Nombre" in result.columns and applied.get("mes"):
        result = result[result["Mes_Nombre"].isin(applied["mes"])]

    if "Sucursal Aliado" in result.columns and applied.get("suc"):
        result = result[result["Sucursal Aliado"].isin(applied["suc"])]

    if applied.get("seg"):
        _seg_col = "Segmento Parawa" if _fuente_app == "Parawa (ponderados)" else "Segmento Cliente"
        if _seg_col in result.columns:
            result = result[result[_seg_col].isin(applied["seg"])]

    for _col, _key in [
        ("Marca_Parawa",      "marca"),
        ("Categoria_Parawa",  "cat"),
        ("Linea_Parawa",      "linea"),
        ("Agrupacion_Parawa", "agrup"),
        ("Nombre_SKU",        "sku"),
    ]:
        if _col in result.columns and applied.get(_key):
            result = result[result[_col].isin(applied[_key])]

    st.sidebar.divider()
    st.sidebar.metric("Registros filtrados", f"{len(result):,}")

    if "Es_Ponderado" in result.columns:
        CV = "Total de Unidades Vendidas (und)"
        tv = result[CV].sum()
        pv = result[result["Es_Ponderado"]][CV].sum()
        cobertura = (pv / tv * 100) if tv > 0 else 0
        st.sidebar.caption(f"📊 Cobertura ponderados: **{cobertura:.1f}%**")

    st.session_state["df_filtered_cache"] = result
    return result


# =================================================================
# KPIs ESTRATÉGICOS
# =================================================================

def _render_kpis_estrategicos(df: pd.DataFrame):
    # ── helpers sin anotación de retorno (compatible Python 3.9) ──
    def _delta(kpi, fmt="{:+.1f}%"):
        v = kpi.get("variacion_pct")
        return fmt.format(v) if v is not None else None

    def _val(kpi, fmt="{:.1f}"):
        v = kpi.get("actual")
        return fmt.format(v) if v is not None else "—"

    try:
        temporalidad = st.session_state.get("temporalidad", "Mensual")
        st.markdown(
            "<h2 style='color:#1A1A2E;font-weight:900;font-size:20px;margin:20px 0 4px 0;'>"
            "📈 KPIs Estratégicos</h2>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='color:#666;font-size:13px;font-weight:600;margin-bottom:12px;'>"
            "Temporalidad: " + temporalidad + " · período actual vs anterior</p>",
            unsafe_allow_html=True
        )

        # ── Cache manual de KPIs ──
        _CV_KPI = "Total de Unidades Vendidas (und)"
        _hash = f"{len(df)}_{df[_CV_KPI].sum():.0f}_{df['Distribuidor'].nunique()}_{temporalidad}"
        if st.session_state.get("kpis_cache_key") != _hash:
            kpi_cob         = calcular_cobertura(df, temporalidad)
            kpi_frec        = calcular_frecuencia(df, temporalidad)
            kpi_frec_compra = calcular_frecuencia_compra(df, temporalidad)
            kpi_amp         = calcular_amplitud(df, temporalidad)
            kpi_vol         = calcular_volumen_por_cliente(df, temporalidad)
            kpi_part        = calcular_participacion(df)
            st.session_state["kpis_cache_key"]    = _hash
            st.session_state["kpis_cache_result"] = {
                "cob": kpi_cob, "frec": kpi_frec, "frec_compra": kpi_frec_compra,
                "amp": kpi_amp, "vol": kpi_vol,   "part": kpi_part,
            }
        else:
            _c              = st.session_state["kpis_cache_result"]
            kpi_cob         = _c["cob"]
            kpi_frec        = _c["frec"]
            kpi_frec_compra = _c["frec_compra"]
            kpi_amp         = _c["amp"]
            kpi_vol         = _c["vol"]
            kpi_part        = _c["part"]

        # --- Fila 1: Cobertura Ponderados / Total / Vol por cliente / Días entre Compra / Frecuencia de Compra ---
        c1, c2, c3, c4, c5 = st.columns(5)

        cob_p = kpi_cob.get("cobertura_ponderados", {})
        c1.markdown(_kpi_card_html(
            "Cobertura Ponderados",
            _val(cob_p) + "%",
            _delta(cob_p, "{:+.1f}pp"),
            emoji="🎯", color="#00ACC1",
        ), unsafe_allow_html=True)

        cob_t = kpi_cob.get("cobertura_total", {})
        c2.markdown(_kpi_card_html(
            "Cobertura Total",
            _val(cob_t) + "%",
            _delta(cob_t, "{:+.1f}pp"),
            emoji="👥", color="#3949AB",
        ), unsafe_allow_html=True)

        c3.markdown(_kpi_card_html(
            "Vol. por Cliente",
            _val(kpi_vol, "{:,.1f}"),
            _delta(kpi_vol),
            emoji="📦", color="#00897B",
        ), unsafe_allow_html=True)

        frec_val   = kpi_frec.get("actual")
        frec_delta = _delta(kpi_frec)
        c4.markdown(_kpi_card_html(
            "Dias entre Compras",
            "{:.1f}".format(frec_val) if frec_val is not None else "—",
            frec_delta,
            inverse=True,
            emoji="📅", color="#F57C00",
        ), unsafe_allow_html=True)

        frec_c_val   = kpi_frec_compra.get("actual")
        frec_c_delta = _delta(kpi_frec_compra)
        c5.markdown(_kpi_card_html(
            "Frecuencia de Compra",
            "{:.2f}".format(frec_c_val) if frec_c_val is not None else "—",
            frec_c_delta,
            emoji="🔄", color="#8E24AA",
        ), unsafe_allow_html=True)

        # --- Fila 2: Amplitud ---
        c1_2, _c2, _c3, _c4 = st.columns(4)
        amp_val = kpi_amp.get("actual")
        c1_2.markdown(_kpi_card_html(
            "Amplitud (SKUs/cliente)",
            "{:.2f}".format(amp_val) if amp_val is not None else "—",
            _delta(kpi_amp),
            emoji="🛒", color="#E53935",
        ), unsafe_allow_html=True)

        # --- Gráfico participación por Categoría ---
        cat_part = kpi_part.get("categoria", {})
        if cat_part:
            st.markdown(
                "<h2 style='color:#1A1A2E;font-weight:900;font-size:20px;margin:20px 0 4px 0;'>"
                "📊 Participación por Categoría</h2>",
                unsafe_allow_html=True
            )
            df_cat = pd.DataFrame(
                list(cat_part.items()), columns=["Categoría", "% Participación"]
            ).sort_values("% Participación", ascending=True)
            fig_cat = go.Figure(go.Bar(
                x=df_cat["% Participación"],
                y=df_cat["Categoría"],
                orientation="h",
                marker_color="#00ACC1",
                text=["{:.1f}%".format(v) for v in df_cat["% Participación"]],
                textposition="outside",
                textfont=dict(size=17, family="Arial"),
            ))
            fig_cat.update_layout(
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                height=max(200, len(df_cat) * 38 + 60),
                margin=dict(l=150, r=50, t=20, b=50),
                font=dict(size=17),
                title=dict(text=""),
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=15, family="Arial Black", color="#8A94A6"), tickcolor="#8A94A6", title=dict(text="", font=dict(size=16, family="Arial Black"))),
                yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=16, family="Arial Black", color="#8A94A6"), tickcolor="#8A94A6", title=dict(text="")),
            )
            st.plotly_chart(fig_cat, use_container_width=True)

    except Exception as _e:
        st.error(f"⚠️ Error en KPIs Estratégicos: {type(_e).__name__}: {_e}")


# =================================================================
# PROYECCIÓN ANUAL
# =================================================================

def _render_proyeccion_anual(df: pd.DataFrame):
    """Gráfico de ventas reales vs proyección para el año en curso."""
    try:
        CV = "Total de Unidades Vendidas (und)"
        anio_actual = str(datetime.now().year)

        if "Anio" not in df.columns or "Mes" not in df.columns:
            return

        df_anio = df[df["Anio"].astype(str) == anio_actual].copy()
        if df_anio.empty:
            st.info(f"Sin datos disponibles para {anio_actual}.")
            return

        df_anio["_mes_int"] = pd.to_numeric(df_anio["Mes"], errors="coerce")
        df_anio = df_anio.dropna(subset=["_mes_int"])
        df_anio["_mes_int"] = df_anio["_mes_int"].astype(int)

        # Ventas reales por mes
        reales = (
            df_anio.groupby("_mes_int")[CV]
            .sum()
            .reindex(range(1, 13), fill_value=0)
        )
        meses_con_data = [m for m in range(1, 13) if reales[m] > 0]
        if not meses_con_data:
            st.info(f"Sin ventas registradas para {anio_actual}.")
            return

        ultimo_mes_real = max(meses_con_data)

        # Tendencia MoM de los últimos 3 meses con data
        vals_recientes = [reales[m] for m in sorted(meses_con_data)[-4:]]
        mom_vals = [
            (vals_recientes[i] - vals_recientes[i - 1]) / vals_recientes[i - 1] * 100
            for i in range(1, len(vals_recientes))
            if vals_recientes[i - 1] > 0
        ]
        tendencia_mom = float(np.mean(mom_vals[-3:])) if mom_vals else 0.0
        tendencia_aplicada = max(min(tendencia_mom, 20.0), -10.0)  # clip razonable

        # Construir serie completa: reales + proyección
        nombres_meses = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
        ventas_reales  = []
        proyeccion     = []
        valor_proy     = float(reales[ultimo_mes_real])

        for m in range(1, 13):
            etiqueta = nombres_meses[m - 1]
            if m in meses_con_data:
                ventas_reales.append(float(reales[m]))
                proyeccion.append(None)
            elif m == ultimo_mes_real + 1:
                # Primer mes proyectado conecta visualmente con el último real
                ventas_reales.append(None)
                valor_proy = float(reales[ultimo_mes_real]) * (1 + tendencia_aplicada / 100)
                proyeccion.append(round(valor_proy))
            else:
                ventas_reales.append(None)
                valor_proy = valor_proy * (1 + tendencia_aplicada / 100)
                proyeccion.append(round(valor_proy))

        df_graf = pd.DataFrame({
            "Mes":            nombres_meses,
            "Ventas Reales":  ventas_reales,
            "Proyección":     proyeccion,
        }).set_index("Mes")

        # Métricas de resumen
        total_real      = sum(r for r in ventas_reales if r is not None)
        total_proyectado = total_real + sum(p for p in proyeccion if p is not None)
        avance_pct      = (total_real / total_proyectado * 100) if total_proyectado > 0 else 0.0

        st.markdown(
            "<h2 style='color:#1A1A2E;font-weight:900;font-size:20px;margin:20px 0 4px 0;'>"
            "📊 Proyección de Ventas — " + anio_actual + "</h2>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='color:#666;font-size:13px;font-weight:600;margin-bottom:12px;'>"
            "Tendencia MoM aplicada: " + "{:+.1f}".format(tendencia_mom)
            + "% (rango entre -10% y +20%)</p>",
            unsafe_allow_html=True
        )
        df_graf_plot = df_graf.reset_index()
        fig_proy = _plotly_linea(
            df_graf_plot, "Mes",
            ["Ventas Reales", "Proyección"],
            "Ventas Reales vs Proyección — " + anio_actual,
            colores=["#00ACC1", "#F57C00"],
        )
        fig_proy.update_layout(
            height=300,
            font=dict(size=17, family="Arial"),
            title=dict(text=""),
            xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=15, color="#8A94A6", family="Arial Black"), tickcolor="#8A94A6", title=dict(font=dict(size=16, family="Arial Black"))),
            yaxis=dict(showticklabels=False, showgrid=True, gridcolor="#E8ECF0", zeroline=False, title=None),
        )
        st.plotly_chart(fig_proy, use_container_width=True)

        mc1, mc2, mc3 = st.columns(3)
        mc1.markdown(_indicator_card_html("Acumulado Real", "{:,.0f} und".format(int(total_real)), "#00ACC1", "✅"), unsafe_allow_html=True)
        mc2.markdown(_indicator_card_html("Proyección Año Completo", "{:,.0f} und".format(int(total_proyectado)), "#3949AB", "🎯"), unsafe_allow_html=True)
        mc3.markdown(_indicator_card_html("Avance vs Proyección", "{:.1f}%".format(avance_pct), "#00897B", "📊"), unsafe_allow_html=True)

    except Exception as _e:
        st.caption(f"⚠️ No se pudo calcular la proyección anual: {_e}")


# =================================================================
# DASHBOARD
# =================================================================

def render_dashboard(df: pd.DataFrame, df_user: pd.DataFrame = None, username: str = ""):
    CV = "Total de Unidades Vendidas (und)"

    st.markdown(
        "<h2 style='color:#1A1A2E;font-weight:900;font-size:20px;margin:20px 0 4px 0;'>"
        "📊 Indicadores Generales</h2>",
        unsafe_allow_html=True
    )
    c1, c2, c3, c4 = st.columns(4)
    _df_activos = df[df[CV] > 0]
    _clientes_activos = _df_activos["Descripción Cliente"].nunique()

    _v_und  = "{:,.0f}".format(df[CV].sum())
    _v_cli  = "{:,}".format(_clientes_activos)
    _v_dist = str(df['Distribuidor'].nunique())
    _v_prod = (
        str(df['Descripción SKU Parawa'].nunique())
        if 'Descripción SKU Parawa' in df.columns else "—"
    )
    c1.markdown(_indicator_card_html("Unidades Totales", _v_und,  "#00ACC1", "📦"), unsafe_allow_html=True)
    c2.markdown(_indicator_card_html("Clientes",         _v_cli,  "#3949AB", "👥"), unsafe_allow_html=True)
    c3.markdown(_indicator_card_html("Distribuidores",   _v_dist, "#00897B", "🏢"), unsafe_allow_html=True)
    c4.markdown(_indicator_card_html("Productos",        _v_prod, "#F57C00", "🛒"), unsafe_allow_html=True)

    fuente = st.session_state.get("fuente_segmento","Distribuidor (todos)")
    if fuente == "Parawa (ponderados)":
        st.info("📊 Mostrando solo clientes **ponderados** con segmentación Parawa.")
    elif "Es_Ponderado" in df.columns:
        tv = df[CV].sum()
        pv = df[df["Es_Ponderado"]][CV].sum()
        cobertura = (pv/tv*100) if tv>0 else 0
        st.caption(f"ℹ️ Vista completa — ponderados representan el **{cobertura:.1f}%** del volumen.")

    _render_kpis_estrategicos(df)
    _render_proyeccion_anual(df)
    st.markdown(
        "<h2 style='color:#1A1A2E;font-weight:900;font-size:20px;margin:20px 0 4px 0;'>"
        "🤖 Agente: Analista de Ventas</h2>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='color:#666;font-size:13px;font-weight:600;margin-bottom:12px;'>"
        "Pandas calcula las métricas · Gemini las interpreta</p>",
        unsafe_allow_html=True
    )

    col_btn, col_mode = st.columns([1,2])
    with col_btn:
        run = st.button("▶ Ejecutar Análisis", type="primary", use_container_width=True)
    with col_mode:
        use_gemini = st.toggle("Incluir interpretación de Gemini", value=True)

    if run:
        try:
            api_key = st.secrets["gemini"]["api_key"]
        except (KeyError, FileNotFoundError):
            st.error("No se encontró la API Key de Gemini.")
            return
        with st.spinner("Analizando..."):
            agent  = SalesAnalystAgent(api_key=api_key)
            result = agent.analyze(df=df, top_n=5, include_gemini=use_gemini)
        st.session_state["agent_result"]    = result.to_dict()
        st.session_state["agent_log"]       = agent.get_log()
        st.session_state["agent_exec_time"] = result.execution_time_seconds
        if result.status == "error":
            st.error(f"Error: {result.error_message}")
            return

    # Crear tabs siempre, independientemente de si el Agente 1 se ejecutó
    is_admin = st.secrets.get("access", {}).get(username, {}).get("region") == "TODOS"
    _tab_labels = ["📋 Resumen Ejecutivo", "📊 Métricas Detalladas", "🎯 Acciones", "🗺️ Análisis Regional", "🔧 Debug"]
    if is_admin:
        _tab_labels.append("🌎 Nacional")
    _all_tabs = st.tabs(_tab_labels)
    tab1, tab2, tab3, tab_regional, tab_debug = _all_tabs[:5]

    # Contenido del Agente 1 (Resumen, Métricas, Acciones, Debug)
    if "agent_result" not in st.session_state:
        with tab1:
            st.info("Presiona **Ejecutar Análisis** para analizar los datos filtrados.")
    else:
        saved = st.session_state["agent_result"]
        if saved.get("status") == "error":
            with tab1:
                st.error(f"Error: {saved.get('error_message','')}")
        else:
            st.success(f"✅ Análisis completado en {st.session_state.get('agent_exec_time',0):.1f} segundos")

            with tab1:
                if saved.get("narrative"):
                    st.markdown(saved["narrative"])
                if saved.get("insights"):
                    st.subheader("💡 Hallazgos Clave")
                    for i, ins in enumerate(saved["insights"], 1):
                        st.markdown(f"**{i}.** {ins}")

            with tab2:
                _render_metrics_tab_from_dict(saved.get("metrics", {}))

            with tab3:
                if saved.get("recommendations"):
                    st.subheader("🎯 Recomendaciones")
                    for i, rec in enumerate(saved["recommendations"], 1):
                        st.markdown(f"**{i}.** {rec}")
                actions = saved.get("raw_analysis", {}).get("priority_actions", [])
                if actions:
                    st.subheader("⚡ Plan de Acción")
                    for a in actions:
                        urgency_map = {"inmediata": "🔴", "esta_semana": "🟡", "este_mes": "🟢"}
                        emoji = urgency_map.get(a.get("urgency", ""), "⚪")
                        st.markdown(f"{emoji} **{a.get('action','')}** — {a.get('urgency','N/A')} | {a.get('impact','N/A')}")

            with tab_debug:
                for entry in st.session_state.get("agent_log", []):
                    st.text(entry)
                with st.expander("JSON completo"):
                    st.json(saved)

    # Tab Análisis Regional (siempre visible)
    with tab_regional:
        _render_regional_tab(st.session_state.get("regional_result"), df_user, username)

    # Tab Nacional (solo admin)
    if is_admin:
        with _all_tabs[5]:
            try:
                _api_key_nac = st.secrets["gemini"]["api_key"]
                _render_nacional_tab(df_user if df_user is not None else df, _api_key_nac)
            except (KeyError, FileNotFoundError):
                st.error("No se encontró la API Key de Gemini en secrets.toml.")


def _render_metrics_tab_from_dict(m: dict):
    tendencia = m.get("tendencia_mensual",{})
    if tendencia.get("disponible") and tendencia.get("tendencia"):
        st.subheader("📈 Tendencia Mensual")
        df_t = pd.DataFrame(tendencia["tendencia"])
        fig_tend = _plotly_linea(df_t, "periodo", ["unidades"], "Unidades por Período")
        fig_tend.update_layout(height=280)
        st.plotly_chart(fig_tend, use_container_width=True)
        df_t_display = df_t.copy()
        df_t_display["unidades"] = df_t_display["unidades"].apply(lambda x: f"{x:,.0f}")
        if "mom_pct" in df_t_display.columns:
            df_t_display["mom_pct"] = df_t_display["mom_pct"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "—")
        df_t_display = df_t_display.rename(columns={"periodo":"Periodo","unidades":"Unidades","mom_pct":"Variación MoM"})
        st.dataframe(df_t_display, use_container_width=True, hide_index=True)

        if tendencia.get("yoy"):
            st.subheader("📅 Comparación Año vs Año")
            df_yoy = pd.DataFrame(tendencia["yoy"])
            df_yoy["unidades_actual"]   = df_yoy["unidades_actual"].apply(lambda x: f"{x:,.0f}")
            df_yoy["unidades_anterior"] = df_yoy["unidades_anterior"].apply(lambda x: f"{x:,.0f}")
            df_yoy["yoy_pct"]           = df_yoy["yoy_pct"].apply(lambda x: f"{x:+.2f}%")
            df_yoy = df_yoy.rename(columns={"mes":"Mes","anio_actual":"Año Actual","anio_anterior":"Año Anterior","unidades_actual":"Und. Actual","unidades_anterior":"Und. Anterior","yoy_pct":"Variación YoY"})
            st.dataframe(df_yoy, use_container_width=True, hide_index=True)

    tb = m.get("top_bottom",{})
    if tb:
        st.subheader("🏆 Top / Bottom Performers")
        dim_labels = {"segmento":"Segmento","producto":"Producto","ruta":"Ruta","sucursal":"Sucursal","categoria":"Categoría","marca":"Marca","linea":"Línea"}
        dim_options = list(tb.keys())
        sel_idx = st.selectbox("Dimensión", range(len(dim_options)), format_func=lambda i: dim_labels.get(dim_options[i],dim_options[i]), key="tb_dim")
        sel_dim = dim_options[sel_idx]

        def _fmt_tb(data, dim_key):
            df_tb = pd.DataFrame(data)
            if df_tb.empty:
                return df_tb
            df_tb = df_tb.rename(columns={dim_key: dim_labels.get(dim_key,dim_key.title())})
            df_tb["unidades"] = df_tb["unidades"].apply(lambda x: f"{x:,.0f}")
            df_tb = df_tb.rename(columns={"unidades":"Unidades"})
            df_tb["pct"] = df_tb["pct"].apply(lambda x: f"{x:.2f}%")
            df_tb = df_tb.rename(columns={"pct":"Participación %"})
            return df_tb

        col_t, col_b = st.columns(2)
        with col_t:
            st.markdown("**🏆 Top**")
            st.dataframe(_fmt_tb(tb[sel_dim]["top"],sel_dim), use_container_width=True, hide_index=True)
        with col_b:
            st.markdown("**⚠️ Bottom**")
            st.dataframe(_fmt_tb(tb[sel_dim]["bottom"],sel_dim), use_container_width=True, hide_index=True)

    anom = m.get("anomalias",{})
    if anom.get("disponible"):
        st.subheader(f"⚠️ Anomalías — {anom.get('total_caidas',0)} caídas · {anom.get('total_picos',0)} picos")
        def _fmt_anom(data):
            df_a = pd.DataFrame(data)
            if df_a.empty:
                return df_a
            df_a["unidades"]      = df_a["unidades"].apply(lambda x: f"{x:,.0f}")
            df_a["variacion_pct"] = df_a["variacion_pct"].apply(lambda x: f"{x:+.2f}%")
            return df_a.rename(columns={"periodo":"Periodo","variacion_pct":"Variación %","unidades":"Unidades"})
        ca, cp = st.columns(2)
        with ca:
            if anom.get("caidas"):
                st.markdown("**🔴 Caídas**")
                st.dataframe(_fmt_anom(anom["caidas"]), use_container_width=True, hide_index=True)
        with cp:
            if anom.get("picos"):
                st.markdown("**🟢 Picos**")
                st.dataframe(_fmt_anom(anom["picos"]), use_container_width=True, hide_index=True)

    conc = m.get("concentracion",{})
    if conc.get("disponible") and conc.get("alertas"):
        st.subheader("🎯 Riesgo de Concentración")
        for a in conc["alertas"]:
            emoji = "🔴" if a["riesgo"]=="alto" else "🟡"
            st.markdown(f"{emoji} **{a['dimension'].title()}:** {a['item']} = **{a['pct']}%**")


# =================================================================
# TAB: ANÁLISIS REGIONAL
# =================================================================

def _render_regional_tab(regional_result: dict, df_user: pd.DataFrame = None, username: str = ""):
    col_btn, col_mode = st.columns([1, 2])
    with col_btn:
        run_reg = st.button("▶ Ejecutar Análisis Regional", type="primary", use_container_width=True)
    with col_mode:
        use_gemini_reg = st.toggle("Incluir Gemini", value=True, key="reg_gemini_toggle")

    if run_reg:
        try:
            api_key = st.secrets["gemini"]["api_key"]
        except (KeyError, FileNotFoundError):
            st.error("No se encontró la API Key de Gemini.")
        else:
            region_name = get_user_region(username)
            df_to_analyze = df_user if df_user is not None else pd.DataFrame()
            if df_to_analyze.empty:
                st.error("Sin datos para analizar.")
            else:
                with st.spinner("Analizando región..."):
                    try:
                        _agent_reg = RegionalAnalystAgent(api_key=api_key)
                        _result_reg = _agent_reg.analyze(
                            df=df_to_analyze, region_name=region_name, include_gemini=use_gemini_reg
                        )
                        st.session_state["regional_result"] = _result_reg.to_dict()
                        st.rerun()
                    except Exception as _e:
                        st.error(f"Error: {_e}")

    if not regional_result:
        st.info("Presiona **Ejecutar Análisis Regional** para analizar tu región.")
        return
    if regional_result.get("status") == "error":
        st.error(f"Error en análisis regional: {regional_result.get('error_message','')}")
        return

    region = regional_result.get("metrics", {}).get("region", "N/A")
    exec_time = regional_result.get("execution_time_seconds", 0)
    st.success(f"✅ Análisis de **{region}** completado en {exec_time:.1f}s")

    if regional_result.get("narrative"):
        st.markdown(regional_result["narrative"])

    metrics = regional_result.get("metrics", {})

    k = metrics.get("kpis_generales", {})
    if k:
        st.subheader("📊 KPIs Regionales")
        c1, c2, c3 = st.columns(3)
        c1.metric("📦 Unidades Totales", f"{k.get('unidades_totales', 0):,.0f}")
        c2.metric("🏢 Distribuidores Activos", str(k.get("distribuidores_activos", "—")))
        c3.metric("📋 Registros", f"{k.get('total_registros', 0):,}")

    tb = metrics.get("top_bottom_distribuidores", {})
    if tb.get("disponible"):
        st.subheader("🏆 Top / Bottom Distribuidores")
        col_t, col_b = st.columns(2)
        with col_t:
            st.markdown("**🏆 Top**")
            df_top = pd.DataFrame(tb.get("top", []))
            if not df_top.empty:
                df_top["unidades"] = df_top["unidades"].apply(lambda x: f"{x:,.0f}")
                df_top["pct"] = df_top["pct"].apply(lambda x: f"{x:.2f}%")
                df_top = df_top.rename(columns={"distribuidor": "Distribuidor", "unidades": "Unidades", "pct": "Participación %"})
                st.dataframe(df_top, use_container_width=True, hide_index=True)
        with col_b:
            st.markdown("**⚠️ Bottom**")
            df_bot = pd.DataFrame(tb.get("bottom", []))
            if not df_bot.empty:
                df_bot["unidades"] = df_bot["unidades"].apply(lambda x: f"{x:,.0f}")
                df_bot["pct"] = df_bot["pct"].apply(lambda x: f"{x:.2f}%")
                df_bot = df_bot.rename(columns={"distribuidor": "Distribuidor", "unidades": "Unidades", "pct": "Participación %"})
                st.dataframe(df_bot, use_container_width=True, hide_index=True)

    t = metrics.get("tendencia_mensual", {})
    if t.get("disponible") and t.get("tendencia"):
        st.subheader(f"📈 Tendencia Regional ({t.get('direccion', '')})")
        df_t = pd.DataFrame(t["tendencia"])
        fig_reg = _plotly_linea(
            df_t, "periodo", ["unidades"],
            "Tendencia Regional — " + t.get("direccion", ""),
        )
        fig_reg.update_layout(height=280)
        st.plotly_chart(fig_reg, use_container_width=True)
        df_t_disp = df_t.copy()
        df_t_disp["unidades"] = df_t_disp["unidades"].apply(lambda x: f"{x:,.0f}")
        if "mom_pct" in df_t_disp.columns:
            df_t_disp["mom_pct"] = df_t_disp["mom_pct"].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "—")
        df_t_disp = df_t_disp.rename(columns={"periodo": "Periodo", "unidades": "Unidades", "mom_pct": "Variación MoM"})
        st.dataframe(df_t_disp, use_container_width=True, hide_index=True)

    a = metrics.get("anomalias", {})
    if a.get("disponible") and a.get("total_caidas_region", 0) > 0:
        st.subheader(f"⚠️ Anomalías — {a['total_caidas_region']} caída(s) detectada(s)")
        caidas_reg = a.get("caidas_region", [])
        caidas_dist = a.get("caidas_distribuidores_ultimo_periodo", [])
        if caidas_reg:
            st.markdown("**Caídas de la región (mensual)**")
            df_cr = pd.DataFrame(caidas_reg)
            df_cr["unidades"] = df_cr["unidades"].apply(lambda x: f"{x:,.0f}")
            df_cr["variacion_pct"] = df_cr["variacion_pct"].apply(lambda x: f"{x:+.2f}%")
            df_cr = df_cr.rename(columns={"periodo": "Periodo", "variacion_pct": "Variación %", "unidades": "Unidades"})
            st.dataframe(df_cr, use_container_width=True, hide_index=True)
        if caidas_dist:
            st.markdown("**Distribuidores con caída en último período**")
            df_cd = pd.DataFrame(caidas_dist)
            df_cd["unidades_actual"] = df_cd["unidades_actual"].apply(lambda x: f"{x:,.0f}")
            df_cd["unidades_anterior"] = df_cd["unidades_anterior"].apply(lambda x: f"{x:,.0f}")
            df_cd["variacion_pct"] = df_cd["variacion_pct"].apply(lambda x: f"{x:+.2f}%")
            df_cd = df_cd.rename(columns={
                "distribuidor": "Distribuidor", "periodo": "Período",
                "variacion_pct": "Variación %", "unidades_actual": "Und. Actual",
                "unidades_anterior": "Und. Anterior",
            })
            st.dataframe(df_cd, use_container_width=True, hide_index=True)

    if regional_result.get("insights"):
        st.subheader("💡 Hallazgos Regionales")
        for i, ins in enumerate(regional_result["insights"], 1):
            st.markdown(f"**{i}.** {ins}")

    if regional_result.get("recommendations"):
        st.subheader("🎯 Recomendaciones Regionales")
        for i, rec in enumerate(regional_result["recommendations"], 1):
            st.markdown(f"**{i}.** {rec}")


# =================================================================
# TAB: ANÁLISIS NACIONAL (solo admin)
# =================================================================

def _render_nacional_tab(df_user: pd.DataFrame, api_key: str):
    st.subheader("🌎 Análisis Nacional")
    st.caption("Consolida los 6 agentes regionales y genera una narrativa ejecutiva comparativa")

    if st.button("▶ Ejecutar Análisis Nacional", type="primary"):
        regiones = list(REGION_PREFIJOS.keys())
        resultados_regionales = []
        progress = st.progress(0, text="Iniciando análisis regionales...")

        for i, region_name in enumerate(regiones):
            progress.progress(i / len(regiones), text=f"Analizando {region_name}...")
            df_region = df_user[df_user["Region_Distribuidor"] == region_name]
            if df_region.empty:
                continue
            try:
                agent_reg = RegionalAnalystAgent(api_key=api_key)
                result_reg = agent_reg.analyze(df=df_region, region_name=region_name, include_gemini=True)
                resultados_regionales.append(result_reg)
            except Exception as e:
                st.warning(f"Error en región {region_name}: {e}")

        if resultados_regionales:
            progress.progress(0.9, text="Generando narrativa nacional con Gemini...")
            try:
                agent_nac = NationalAnalystAgent(api_key=api_key)
                result_nac = agent_nac.analyze(regional_results=resultados_regionales, include_gemini=True)
                st.session_state["nacional_result"] = result_nac.to_dict()
                st.session_state["nacional_regional_results"] = [r.to_dict() for r in resultados_regionales]
            except Exception as e:
                st.error(f"Error en análisis nacional: {e}")
        else:
            st.error("No se obtuvieron resultados de ninguna región.")

        progress.progress(1.0, text="¡Completado!")
        progress.empty()

    nacional = st.session_state.get("nacional_result")
    if not nacional:
        st.info("Presiona **Ejecutar Análisis Nacional** para consolidar todas las regiones.")
        return
    if nacional.get("status") == "error":
        st.error(f"Error: {nacional.get('error_message','')}")
        return

    exec_time = nacional.get("execution_time_seconds", 0)
    st.success(f"✅ Análisis nacional completado en {exec_time:.1f}s")

    if nacional.get("narrative"):
        st.markdown(nacional["narrative"])

    comp = nacional.get("metrics", {}).get("comparativa_regional", [])
    if comp:
        st.subheader("📊 Comparativa Regional")
        df_comp = pd.DataFrame(comp)
        cols_show = ["region", "unidades_totales", "pct_nacional", "distribuidores_activos", "tendencia", "mom_pct_ultimo"]
        df_comp = df_comp[[c for c in cols_show if c in df_comp.columns]].copy()
        if "unidades_totales" in df_comp.columns:
            df_comp["unidades_totales"] = df_comp["unidades_totales"].apply(lambda x: f"{x:,.0f}")
        if "pct_nacional" in df_comp.columns:
            df_comp["pct_nacional"] = df_comp["pct_nacional"].apply(lambda x: f"{x:.2f}%")
        if "mom_pct_ultimo" in df_comp.columns:
            df_comp["mom_pct_ultimo"] = df_comp["mom_pct_ultimo"].apply(
                lambda x: f"{x:+.2f}%" if pd.notna(x) and x is not None else "—"
            )
        df_comp = df_comp.rename(columns={
            "region": "Región", "unidades_totales": "Unidades Totales",
            "pct_nacional": "% Nacional", "distribuidores_activos": "Dist. Activos",
            "tendencia": "Tendencia", "mom_pct_ultimo": "MoM Último",
        })
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

    if nacional.get("insights"):
        st.subheader("💡 Hallazgos Nacionales")
        for i, ins in enumerate(nacional["insights"], 1):
            st.markdown(f"**{i}.** {ins}")

    if nacional.get("recommendations"):
        st.subheader("🎯 Recomendaciones Nacionales")
        for i, rec in enumerate(nacional["recommendations"], 1):
            st.markdown(f"**{i}.** {rec}")

    actions = nacional.get("raw_analysis", {}).get("priority_actions", [])
    if actions:
        st.subheader("⚡ Plan de Acción Nacional")
        urgency_map = {"inmediata": "🔴", "esta_semana": "🟡", "este_mes": "🟢"}
        for a in actions:
            emoji = urgency_map.get(a.get("urgency", ""), "⚪")
            region_obj = a.get("region_objetivo", "todas")
            st.markdown(f"{emoji} **{a.get('action','')}** · Región: {region_obj} · {a.get('urgency','N/A')} | {a.get('impact','N/A')}")

    alertas = nacional.get("metrics", {}).get("alertas_nacionales", {})
    if alertas.get("regiones_alertadas", 0) > 0:
        st.subheader(f"⚠️ Alertas — {alertas['regiones_alertadas']} región(es) con caídas")
        for alerta in alertas.get("regiones_con_caidas", []):
            st.markdown(
                f"🔴 **{alerta['region']}**: {alerta['total_caidas']} caída(s) · "
                f"peor: {alerta.get('peor_caida_pct')}% en {alerta.get('peor_caida_periodo')}"
            )

    resultados_reg = st.session_state.get("nacional_regional_results", [])
    if resultados_reg:
        with st.expander("🔍 Ver detalle por región"):
            for r_dict in resultados_reg:
                region_name = r_dict.get("metrics", {}).get("region", r_dict.get("agent_name", ""))
                st.markdown(f"**{region_name}**")
                if r_dict.get("narrative"):
                    st.caption(r_dict["narrative"][:300] + "...")


# =================================================================
# CHAT INTELIGENTE
# =================================================================

def render_chat(df: pd.DataFrame, username: str):
    st.subheader("💬 Chat Inteligente con IA")
    st.caption("Conoce tu perfil, tus datos y el último análisis ejecutado")

    with st.expander("📝 Agregar contexto al análisis", expanded=False):
        st.caption("Eventos recientes, condiciones del mercado, objetivos del período, etc.")
        contexto_actual = st.session_state.get("contexto_regional","")
        nuevo_contexto  = st.text_area(
            "Contexto",
            value=contexto_actual, height=100,
            placeholder="Ej: En febrero hubo un paro de transporte. El objetivo Q1 es +15%...",
            key="input_contexto_regional",
        )
        col_save, col_clear = st.columns(2)
        with col_save:
            if st.button("💾 Guardar", use_container_width=True):
                st.session_state["contexto_regional"] = nuevo_contexto
                st.success("✅ Contexto guardado.")
        with col_clear:
            if st.button("🗑️ Limpiar", use_container_width=True):
                st.session_state["contexto_regional"] = ""
                st.rerun()
        if st.session_state.get("contexto_regional"):
            texto = st.session_state["contexto_regional"]
            st.info(f"📌 Activo: *{texto[:100]}{'...' if len(texto)>100 else ''}*")

    st.divider()

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if not st.session_state["chat_history"]:
        st.info(f"👋 Hola **{username}**! Pregúntame sobre ventas, tendencias, productos o clientes de tu región.")

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Escribe tu pregunta sobre los datos...")

    if user_input:
        st.session_state["tab_activo"] = "chat"
        st.session_state["chat_history"].append({"role":"user","content":user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            api_key = st.secrets["gemini"]["api_key"]
        except (KeyError, FileNotFoundError):
            st.error("No se encontró la API Key de Gemini.")
            return

        with st.chat_message("assistant"):
            with st.spinner("Analizando tus datos..."):
                try:
                    df_chat, fecha_corte, fecha_max = _filtrar_ultimos_12_meses(df)
                    df_anonimo, mapa_inverso = _sanitizar_df(df_chat)
                    periodo_ctx = ""
                    if fecha_corte and fecha_max:
                        periodo_ctx = f"\nPERÍODO DE ANÁLISIS: últimos 12 meses (desde {fecha_corte} hasta {fecha_max})"
                    user_context   = build_user_context(username, df_anonimo, anonimizar=True) + periodo_ctx
                    code           = _generate_pandas_code(api_key, df_anonimo, user_input, user_context)
                    result_data    = _execute_pandas_code(code, df_anonimo)
                    final_response = _interpret_result(api_key, user_input, result_data, code, user_context)
                    final_response = _desanonimizar_respuesta(final_response, mapa_inverso)
                    st.markdown(final_response)
                    st.session_state["chat_history"].append({"role":"assistant","content":final_response})
                except Exception as e:
                    error_msg = f"Hubo un error al procesar tu pregunta: {str(e)}"
                    st.error(error_msg)
                    st.session_state["chat_history"].append({"role":"assistant","content":error_msg})


def _filtrar_ultimos_12_meses(df: pd.DataFrame):
    """Filtra df a los últimos 12 meses desde la fecha máxima disponible.
    Retorna (df_filtrado, fecha_corte_str, fecha_max_str).
    Si falla, retorna (df_original, None, None).
    """
    try:
        fechas = pd.to_datetime(df["Fecha de Venta"], errors="coerce").dropna()
        if fechas.empty:
            return df, None, None
        fecha_max  = fechas.max()
        fecha_corte = fecha_max - pd.Timedelta(days=365)
        df_f = df[pd.to_datetime(df["Fecha de Venta"], errors="coerce") >= fecha_corte]
        return df_f, fecha_corte.strftime("%Y-%m-%d"), fecha_max.strftime("%Y-%m-%d")
    except Exception:
        return df, None, None


def _get_df_schema(df: pd.DataFrame) -> str:
    CV = "Total de Unidades Vendidas (und)"
    lines = [f"COLUMNAS EN 'df': {len(df):,} filas",""]
    for col in df.columns:
        sample_vals = df[col].dropna().unique()[:5].tolist()
        lines.append(f"  - '{col}' ({df[col].dtype}) — {df[col].nunique()} únicos — Ej: {', '.join([str(v) for v in sample_vals])}")
    lines += [
        "",f"VENTAS: '{CV}' (UNIDADES, no dólares)",
        "NOTA: 'Anio' y 'Mes' son string.",
        "NOTA: 'Region_Distribuidor' = región del distribuidor.",
        "NOTA: 'Segmento Parawa' = segmento correcto. 'Segmento Cliente' = original.",
        "NOTA: 'Es_Ponderado' = True si está en maestro Parawa.",
    ]
    return "\n".join(lines)


def _gemini_request_with_retry(api_key: str, payload: dict, max_retries: int = 3) -> dict:
    import time as _time
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    for attempt in range(1, max_retries+1):
        try:
            response = requests.post(url, json=payload, headers={"Content-Type":"application/json"}, timeout=120)
            if response.status_code == 200:
                return response.json()
            if response.status_code in [429,500,502,503]:
                if attempt < max_retries:
                    _time.sleep(3*attempt)
                    continue
                raise RuntimeError(f"Gemini no respondió tras {max_retries} intentos.")
            raise RuntimeError(f"Error HTTP {response.status_code}")
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                _time.sleep(3*attempt)
                continue
            raise RuntimeError("Gemini tardó demasiado.")
        except requests.exceptions.ConnectionError:
            if attempt < max_retries:
                _time.sleep(3*attempt)
                continue
            raise RuntimeError("No se pudo conectar con Gemini.")
    raise RuntimeError("Error inesperado.")


def _generate_pandas_code(api_key: str, df: pd.DataFrame, question: str, user_context: str = "") -> str:
    schema  = _get_df_schema(df)
    history = ""
    if st.session_state.get("chat_history"):
        recent  = st.session_state["chat_history"][-4:]
        h_parts = [f"{'USUARIO' if m['role']=='user' else 'ASISTENTE'}: {m['content'][:200]}" for m in recent]
        history = "CONVERSACION RECIENTE:\n" + "\n".join(h_parts) + "\n\n"

    system_prompt = """Eres un experto en Pandas para análisis de ventas (Sell Out).
Tu ÚNICO trabajo es GENERAR CÓDIGO PANDAS que calcule exactamente lo que pregunta el usuario.

COLUMNAS DISPONIBLES:
- Total de Unidades Vendidas (und) -> volumen
- Descripción Cliente -> nombre del cliente
- Distribuidor -> nombre del distribuidor
- Es_Ponderado -> True si cliente está en maestro, False si no
- Fecha de Venta, Anio, Mes -> temporalidad
- Categoria_Parawa, Marca_Parawa, Linea_Parawa -> jerarquía de producto
- Código SKU Parawa -> código SKU

FÓRMULAS ESPECÍFICAS (NUNCA INVENTES OTRAS):

1. Cobertura Ponderada = (clientes ponderados activos) / 2389 * 100
   - Clientes ponderados activos = df[df["Es_Ponderado"] == True]["Descripción Cliente"].nunique()
   - 2389 es el universo de clientes ponderados (CONSTANTE, no calcules)
   - Resultado SIEMPRE será un % entre 0 y 100, NUNCA 100% a menos que se alcance el universo

2. Cobertura Total = (TODOS los clientes con ventas > 0) / (universo total del df)
   - df[df["Total de Unidades Vendidas (und)"] > 0]["Descripción Cliente"].nunique() / df["Descripción Cliente"].nunique()

3. Amplitud = promedio de SKUs distintos por cliente activo
   - df.groupby("Descripción Cliente")["Código SKU Parawa"].nunique().mean()

4. Volumen por Cliente = unidades totales / cantidad de clientes con ventas > 0
   - df[df["Total de Unidades Vendidas (und)"] > 0]["Total de Unidades Vendidas (und)"].sum() / df[df["Total de Unidades Vendidas (und)"] > 0]["Descripción Cliente"].nunique()

5. Frecuencia de Compra = activaciones únicas / (clientes únicos * períodos únicos)
   - activaciones = df.drop_duplicates(subset=["Descripción Cliente", "Fecha de Venta"]).shape[0]
   - clientes = df["Descripción Cliente"].nunique()
   - periodos = df["Anio"].nunique() * 12
   - resultado = activaciones / (clientes * periodos)

REGLAS CRÍTICAS - NUNCA VIOLES:
1. NO uses import (ni import pandas, nada)
2. NO uses pd.to_datetime (fechas ya vienen parseadas)
3. NO uses unicode (usa >= en lugar de >=, no uses caracteres especiales)
4. NO uses variables externas (solo df está disponible)
5. SIEMPRE valida que el resultado tenga sentido (cobertura 0-100%, volumen > 0, etc.)
6. Si denominador es cero, retorna 0
7. Máximo 15 líneas de código
8. Resultado SIEMPRE en variable llamada `resultado`

FORMATO DE RESPUESTA - SOLO CÓDIGO:
```python
# línea 1 del código
resultado = ...
```

SIN MARKDOWN EXTRA, SIN EXPLICACIONES, SIN NADA MÁS."""

    # Contenido dinámico en el mensaje de usuario
    user_message = (
        schema + "\n\n"
        + user_context + "\n\n"
        + history
        + "PREGUNTA: " + question + "\n\nCódigo:"
    )

    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_message}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1024},
    }
    data = _gemini_request_with_retry(api_key, payload)
    code = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()

    # Limpieza robusta de fences markdown (con o sin salto de línea)
    import re as _re
    code = _re.sub(r"^```(?:python)?\s*\n?", "", code)
    code = _re.sub(r"\n?```\s*$", "", code)
    return code.strip()


def _execute_pandas_code(code: str, df: pd.DataFrame) -> str:
    forbidden = [
        "import os","import sys","import subprocess","os.system","exec(","eval(",
        "open(","__import__","shutil","pathlib","glob","import socket",
        "requests.","urllib","http.","subprocess","os.remove","os.path",
        "st.secrets","api_key","secret","password","token",
    ]
    for f in forbidden:
        if f.lower() in code.lower():
            return f"BLOQUEADO: código intentó usar '{f}'."

    local_vars = {"df": df.copy(), "pd": pd}
    try:
        exec(code, {"__builtins__":{
            "len":len,"str":str,"int":int,"float":float,"round":round,
            "sorted":sorted,"list":list,"dict":dict,"tuple":tuple,"set":set,
            "min":min,"max":max,"sum":sum,"abs":abs,"range":range,
            "enumerate":enumerate,"zip":zip,"map":map,"filter":filter,
            "True":True,"False":False,"None":None,"print":lambda *a,**k:None
        }}, local_vars)
    except Exception as e:
        return f"ERROR AL EJECUTAR: {str(e)}\n\nCÓDIGO:\n{code}"

    resultado = local_vars.get("resultado","No se generó resultado.")
    if isinstance(resultado, pd.DataFrame):
        if len(resultado) > 50:
            return resultado.head(50).to_string(index=False) + f"\n\n... ({len(resultado)} filas)"
        return resultado.to_string(index=False)
    elif isinstance(resultado, pd.Series):
        return resultado.to_string()
    elif isinstance(resultado, (dict,list)):
        return json.dumps(resultado, ensure_ascii=False, indent=2, default=str)
    else:
        return str(resultado)


def _interpret_result(api_key: str, question: str, result_data: str, code: str, user_context: str = "") -> str:
    if result_data.startswith("BLOQUEADO:") or result_data.startswith("ERROR AL EJECUTAR:"):
        return f"⚠️ {result_data}"

    system_prompt = (
        "ROL: Eres un Analista Senior de Ventas de Parawa especializado en consumo masivo Venezuela.\n"
        + user_context + "\n\n"
        "FORMATO DE RESPUESTA SEGÚN TIPO:\n"
        "- Si es un KPI puntual: dar el número, compararlo vs período anterior (disponible en el contexto de KPIs), "
        "calificar si es bueno/malo/regular según el contexto del negocio.\n"
        "- Si es un listado: presentar los top 5-10 más relevantes en texto, mencionar el total de registros.\n"
        "- Si es recomendación: dar máximo 3 acciones concretas priorizadas por impacto, "
        "con el grupo específico (distribuidor, segmento o categoría) a atacar.\n\n"
        "REGLAS: 1) Organizado y legible. 2) Usa emojis. 3) Miles con coma (1,234,567). "
        "4) Usa el contexto del usuario. 5) NUNCA menciones código ni DataFrames. "
        "6) Si hay error, sugiere cómo reformular la pregunta. 7) Ventas siempre en UNIDADES. "
        "8) Dirígete al usuario por nombre cuando sea natural. "
        "9) NUNCA inventes números ni digas 'según los datos'.\n\n"
        "AL FINAL: si el resultado tiene más de 5 filas, agregar exactamente: "
        "'📥 Puedes descargar este listado completo en la pestaña **Explorar Datos** aplicando los mismos filtros.'"
    )

    payload = {
        "contents":[{"role":"user","parts":[{"text": system_prompt + "\n\nPREGUNTA: " + question + "\n\nRESULTADO:\n" + result_data[:3000] + "\n\nResponde:"}]}],
        "generationConfig":{"temperature":0.4,"maxOutputTokens":4096},
    }
    try:
        data = _gemini_request_with_retry(api_key, payload)
        text = data.get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text","")
        return text if text else "No pude interpretar los resultados."
    except RuntimeError as e:
        return f"⚠️ {str(e)}"


# =================================================================
# SANITIZACIÓN DE DATOS PARA EL CHAT
# =================================================================

def _sanitizar_df(df: pd.DataFrame):
    """
    Anonimiza columnas con datos sensibles antes de enviar a Gemini.
    Los códigos son estables en toda la sesión (session_state['sanitize_map']).
    Retorna (df_anonimo, mapa_inverso).

    Columnas anonimizadas:
      Descripción Cliente, Nombre del PDV → CLIENTE_0001, CLIENTE_0002, ...
      Distribuidor                        → DIST_01, DIST_02, ...
      Sucursal Aliado                     → SUCURSAL_01, SUCURSAL_02, ...
      Nombre_SKU                          → SKU_0001, SKU_0002, ...
      Region_Distribuidor                 → REGION_A, REGION_B, ...
    Columnas con "direc" o "ciudad" en el nombre → eliminadas.
    """
    import string as _string

    df = df.copy()

    smap = st.session_state.get("sanitize_map")
    if not smap:
        smap = {
            "clientes":       {},
            "distribuidores": {},
            "sucursales":     {},
            "skus":           {},
            "regiones":       {},
        }

    def _encode(pool_key, value, prefix, pad):
        """Asigna código anónimo al valor; reutiliza el existente si ya fue visto."""
        if pd.isna(value) or str(value).strip() in ("", "None", "nan"):
            return value
        val = str(value).strip()
        pool = smap[pool_key]
        if val not in pool:
            n = len(pool) + 1
            if pad == 0:
                pool[val] = f"{prefix}{_string.ascii_uppercase[min(n - 1, 25)]}"
            else:
                pool[val] = f"{prefix}{str(n).zfill(pad)}"
        return pool[val]

    # Descripción Cliente + Nombre del PDV → CLIENTE_XXXX (mismo pool)
    for col in ["Descripción Cliente", "Nombre del PDV"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: _encode("clientes", v, "CLIENTE_", 4))

    # Distribuidor → DIST_XX
    if "Distribuidor" in df.columns:
        df["Distribuidor"] = df["Distribuidor"].apply(
            lambda v: _encode("distribuidores", v, "DIST_", 2)
        )

    # Sucursal Aliado → SUCURSAL_XX
    if "Sucursal Aliado" in df.columns:
        df["Sucursal Aliado"] = df["Sucursal Aliado"].apply(
            lambda v: _encode("sucursales", v, "SUCURSAL_", 2)
        )

    # Nombre_SKU → SKU_XXXX
    if "Nombre_SKU" in df.columns:
        df["Nombre_SKU"] = df["Nombre_SKU"].apply(
            lambda v: _encode("skus", v, "SKU_", 4)
        )

    # Region_Distribuidor → REGION_A, REGION_B, ...
    if "Region_Distribuidor" in df.columns:
        df["Region_Distribuidor"] = df["Region_Distribuidor"].apply(
            lambda v: _encode("regiones", v, "REGION_", 0)
        )

    # Eliminar columnas de dirección / ciudad
    drop_cols = [
        c for c in df.columns
        if any(kw in c.lower() for kw in ["direc", "ciudad"])
    ]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    st.session_state["sanitize_map"] = smap

    # Mapa inverso: código → nombre real (para restaurar en la respuesta)
    mapa_inverso = {}
    for pool in smap.values():
        for real, code in pool.items():
            mapa_inverso[code] = real

    return df, mapa_inverso


def _desanonimizar_respuesta(texto: str, mapa_inverso: dict) -> str:
    """Reemplaza códigos anónimos en la respuesta de Gemini por los nombres reales."""
    if not mapa_inverso or not texto:
        return texto
    # Ordenar por longitud descendente para evitar reemplazos parciales
    for code in sorted(mapa_inverso.keys(), key=len, reverse=True):
        texto = texto.replace(code, mapa_inverso[code])
    return texto


# =================================================================
# AGENTE REGIONAL AUTOMÁTICO
# =================================================================

def _run_regional_agent_if_needed(df_user: pd.DataFrame, username: str):
    """Ejecuta el RegionalAnalystAgent una vez por sesión y guarda en session_state."""
    if "regional_result" in st.session_state:
        return
    try:
        api_key = st.secrets["gemini"]["api_key"]
    except (KeyError, FileNotFoundError):
        return

    region_name = get_user_region(username)
    if region_name in ("Sin Región",):
        return

    try:
        agent = RegionalAnalystAgent(api_key=api_key)
        result = agent.analyze(df=df_user, region_name=region_name, include_gemini=True)
        st.session_state["regional_result"] = result.to_dict()
    except Exception:
        pass  # No bloquear el dashboard si el agente falla


# =================================================================
# TAB: METAS INTELIGENTES
# =================================================================

def _detectar_rol(username: str):
    """
    Retorna el rol del usuario y su config de acceso.
    Roles: 'admin', 'divisional', 'gerente', 'asesor'.
    Admin se detecta por region == "TODOS" (soporta demo_admin y futuros admins).
    """
    try:
        access = st.secrets.get("access", {}).get(username, {})
    except Exception:
        access = {}

    region = access.get("region", None)

    if region == "TODOS":
        return "admin", access

    if region is not None:
        if isinstance(region, list):
            return "divisional", access
        return "gerente", access

    if access.get("distribuidores"):
        return "asesor", access

    return "asesor", access


def _metas_key(distribuidor: str, periodo: str, periodicidad: str) -> str:
    """Clave única para identificar una meta en metas.json."""
    return f"{distribuidor}|{periodo}|{periodicidad}"


def render_metas_tab(df_user: pd.DataFrame, df_all: pd.DataFrame, username: str):
    st.subheader("🎯 Metas Inteligentes")

    rol, access = _detectar_rol(username)

    if rol == "admin":
        _render_metas_admin(df_all, username)
    elif rol == "divisional":
        _render_metas_divisional(df_user, access, username)
    elif rol == "gerente":
        _render_metas_gerente(df_user, access, username)
    else:
        _render_metas_asesor(df_user, access, username)


# ── DESCARGA METAS OFICIALES (helper reutilizable) ───────────────

def _render_descarga_metas(metas_filtradas: dict):
    """Muestra botón de descarga Excel con las metas oficiales del dict recibido."""
    import io
    oficiales = {k: v for k, v in metas_filtradas.items() if v.get("estado") == "oficial"}
    st.divider()
    if not oficiales:
        st.caption("Sin metas oficiales disponibles para descargar aún.")
        return

    df_dl = pd.DataFrame([
        {
            "Distribuidor":  v["distribuidor"],
            "Región":        v.get("region", ""),
            "Período":       v["periodo"],
            "Periodicidad":  v.get("periodicidad", ""),
            "Meta Oficial":  v.get("meta_oficial", v.get("meta_preliminar", "")),
            "Aprobado Por":  v.get("meta_oficial_usuario", ""),
        }
        for v in oficiales.values()
    ]).sort_values(["Región", "Distribuidor", "Período"])

    try:
        buf = io.BytesIO()
        df_dl.to_excel(buf, index=False, engine="openpyxl")
        buf.seek(0)
        st.download_button(
            f"⬇️ Descargar Metas Oficiales ({len(oficiales)}) — Excel",
            data=buf,
            file_name=f"metas_oficiales_{datetime.now().strftime('%Y%m')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except ImportError:
        st.warning("Instala openpyxl para habilitar la descarga: `pip install openpyxl`")


# ── VISTA ADMIN ──────────────────────────────────────────────────

def _render_metas_admin(df_all: pd.DataFrame, username: str):
    st.caption("Calcula escenarios de metas por distribuidor, edítalos y guárdalos como metas preliminares.")

    # Selector de período — próximos 6 meses futuros a partir de hoy
    from datetime import date as _date
    _hoy = _date.today()
    _year, _month = _hoy.year, _hoy.month
    periodos_disponibles = []
    for _ in range(6):
        _month += 1
        if _month > 12:
            _month = 1
            _year += 1
        periodos_disponibles.append(f"{_year}-{str(_month).zfill(2)}")

    col_per, col_temp, col_reg = st.columns(3)
    with col_per:
        periodo_sel = st.selectbox(
            "Período objetivo",
            periodos_disponibles,
            help="Mes para el cual se generarán las metas",
        )
    with col_temp:
        periodicidad_sel = st.selectbox(
            "Periodicidad",
            ["Mensual", "Bimestral", "Trimestral"],
            key="metas_periodicidad",
        )
    with col_reg:
        regiones_disponibles = ["Todas"]
        if "Region_Distribuidor" in df_all.columns:
            regiones_disponibles += sorted(df_all["Region_Distribuidor"].dropna().unique().tolist())
        region_filtro = st.selectbox("Región", regiones_disponibles, key="metas_region_filtro")

    df_para_metas = (
        df_all if region_filtro == "Todas"
        else df_all[df_all["Region_Distribuidor"] == region_filtro]
    )
    n_dist = df_para_metas["Distribuidor"].nunique() if "Distribuidor" in df_para_metas.columns else 0
    st.caption(f"Se calcularán escenarios para **{n_dist}** distribuidor(es).")

    if st.button("▶ Calcular Escenarios de Metas", type="primary"):
        st.session_state["tab_activo"] = "metas"
        st.session_state["preserve_filters"] = True
        with st.spinner("Calculando escenarios..."):
            try:
                engine = GoalsEngine()
                propuestas = engine.calculate_proposals(df_para_metas, periodo_sel, periodicidad_sel)
                st.session_state["metas_propuestas"] = propuestas
                st.session_state["metas_periodo_sel"] = periodo_sel
                st.session_state["metas_periodicidad_sel"] = periodicidad_sel
                st.success(f"✅ {len(propuestas)} escenarios calculados.")
            except Exception as e:
                st.error(f"Error al calcular escenarios: {e}")
                return

    propuestas = st.session_state.get("metas_propuestas", [])
    if not propuestas:
        st.info("Presiona **Calcular Escenarios de Metas** para comenzar.")
        return

    periodo_activo     = st.session_state.get("metas_periodo_sel", periodo_sel)
    periodicidad_activa = st.session_state.get("metas_periodicidad_sel", periodicidad_sel)

    # Índice para lookup al guardar
    prop_idx = {p["distribuidor"]: p for p in propuestas}

    # Escenario por defecto = el de mayor valor
    _ESC_OPTS = [
        "Escenario 1 - Volumen",
        "Escenario 2 - Activación",
        "Escenario 3 - Amplitud",
        "Manual",
    ]

    def _default_escenario(p):
        vals = [p["meta_escenario1"], p["meta_escenario2"], p["meta_escenario3"]]
        idx  = vals.index(max(vals))
        return _ESC_OPTS[idx]

    def _meta_para_escenario(p, esc):
        return {
            "Escenario 1 - Volumen":    p["meta_escenario1"],
            "Escenario 2 - Activación": p["meta_escenario2"],
            "Escenario 3 - Amplitud":   p["meta_escenario3"],
        }.get(esc, max(p["meta_escenario1"], p["meta_escenario2"], p["meta_escenario3"]))

    # Construir DataFrame editable
    df_edit = pd.DataFrame([
        {
            "Región":              p["region"],
            "Distribuidor":        p["distribuidor"],
            "Promedio Ventas":     p["promedio_6m"],
            "Activación":          p["clientes_activos"],
            "Prom/Cliente":        p["promedio_por_cliente"],
            "Amplitud":            p["amplitud_actual"],
            "Tend. Ventas %":      p["tendencia_mom_ventas"],
            "Tend. Activación %":  p["tendencia_mom_activacion"],
            "Meta Esc. 1":         p["meta_escenario1"],
            "Meta Esc. 2":         p["meta_escenario2"],
            "Meta Esc. 3":         p["meta_escenario3"],
            "Escenario":           _default_escenario(p),
            "Meta Final":          _meta_para_escenario(p, _default_escenario(p)),
            "Confianza":           p["nivel_confianza"],
        }
        for p in propuestas
    ])

    st.markdown("**Selecciona el escenario por distribuidor y ajusta 'Meta Final' si lo deseas:**")
    edited = st.data_editor(
        df_edit,
        column_config={
            "Región":             st.column_config.TextColumn(disabled=True),
            "Distribuidor":       st.column_config.TextColumn(disabled=True),
            "Promedio Ventas":    st.column_config.NumberColumn(disabled=True, format="%.1f"),
            "Activación":         st.column_config.NumberColumn(disabled=True, format="%d"),
            "Prom/Cliente":       st.column_config.NumberColumn(disabled=True, format="%.1f"),
            "Amplitud":           st.column_config.NumberColumn(disabled=True, format="%.2f"),
            "Tend. Ventas %":     st.column_config.NumberColumn(disabled=True, format="%.2f%%"),
            "Tend. Activación %": st.column_config.NumberColumn(disabled=True, format="%.2f%%"),
            "Meta Esc. 1":        st.column_config.NumberColumn(disabled=True),
            "Meta Esc. 2":        st.column_config.NumberColumn(disabled=True),
            "Meta Esc. 3":        st.column_config.NumberColumn(disabled=True),
            "Escenario":          st.column_config.SelectboxColumn(options=_ESC_OPTS, required=True),
            "Meta Final":         st.column_config.NumberColumn(min_value=0, step=1),
            "Confianza":          st.column_config.TextColumn(disabled=True),
        },
        use_container_width=True,
        hide_index=True,
        key="metas_editor_admin",
    )

    # Expander con justificaciones de escenarios
    with st.expander("📝 Ver justificaciones por escenario"):
        for p in propuestas:
            st.markdown(
                f"**{p['distribuidor']}**  \n"
                f"Esc. 1: {p['just_escenario1']}  \n"
                f"Esc. 2: {p['just_escenario2']}  \n"
                f"Esc. 3: {p['just_escenario3']}"
            )
            st.divider()

    if st.button("💾 Guardar como Metas Preliminares"):
        st.session_state["tab_activo"] = "metas"
        st.session_state["preserve_filters"] = True
        metas = load_metas()
        fecha_creacion = datetime.now().strftime("%Y-%m")
        _ESTADOS_PROTEGIDOS = {"pendiente_aprobacion", "oficial"}
        n_guardadas = 0
        n_bloqueadas = 0
        for _, row in edited.iterrows():
            dist  = row["Distribuidor"]
            p     = prop_idx.get(dist, {})
            key   = _metas_key(dist, periodo_activo, periodicidad_activa)
            existente = metas.get(key, {})
            estado_existente = existente.get("estado", "")
            if estado_existente in _ESTADOS_PROTEGIDOS:
                st.warning(f"⚠️ {dist}: ya tiene estado **'{estado_existente}'** — no se sobreescribió")
                n_bloqueadas += 1
                continue
            metas[key] = {
                "distribuidor":      dist,
                "region":            row["Región"],
                "periodo":           periodo_activo,
                "periodicidad":      periodicidad_activa,
                "fecha_creacion":    fecha_creacion,
                "meta_esc1":         int(p.get("meta_escenario1", 0)),
                "meta_esc2":         int(p.get("meta_escenario2", 0)),
                "meta_esc3":         int(p.get("meta_escenario3", 0)),
                "meta_preliminar":   int(row["Meta Final"]),
                "escenario_elegido": row.get("Escenario", "Manual"),
                "nivel_confianza":   row["Confianza"],
                "estado":            "preliminar",
            }
            n_guardadas += 1
        save_metas(metas)
        if n_guardadas:
            st.success(f"✅ {n_guardadas} metas guardadas/actualizadas como **Preliminares** para {periodo_activo}.")
        if n_bloqueadas:
            st.info(f"⚠️ {n_bloqueadas} metas NO modificadas (estado: pendiente aprobación u oficial).")

    _render_descarga_metas(load_metas())


# ── VISTA GERENTE ─────────────────────────────────────────────────

def _render_metas_gerente(df_user: pd.DataFrame, access: dict, username: str):
    st.caption("Revisa las metas preliminares de tu región y envía tu sugerencia.")

    region_gerente = access.get("region", "")
    metas = load_metas()

    # Filtrar metas de mi región
    mis_metas = {
        k: v for k, v in metas.items()
        if v.get("region") == region_gerente
    }

    if not mis_metas:
        st.info("No hay metas preliminares disponibles para tu región aún. El Admin debe generarlas primero.")
        return

    # Agrupar por período disponible
    periodos = sorted(set(v["periodo"] for v in mis_metas.values()), reverse=True)
    periodo_sel = st.selectbox("Período", periodos, key="metas_gerente_periodo")

    metas_periodo = {k: v for k, v in mis_metas.items() if v["periodo"] == periodo_sel}

    if not metas_periodo:
        st.info("Sin metas para este período.")
        return

    st.markdown(f"**{len(metas_periodo)} distribuidores · Período: {periodo_sel}**")

    for key, meta in metas_periodo.items():
        dist = meta["distribuidor"]
        estado = meta.get("estado", "preliminar")

        with st.container():
            c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
            c1.markdown(f"**{dist}**")
            c2.metric("Meta Preliminar", f"{meta.get('meta_preliminar', '—'):,}")

            if estado in ("pendiente_aprobacion", "oficial"):
                sug = meta.get("sugerencia_gerente", "—")
                c3.metric("Tu Sugerencia", f"{sug:,}" if isinstance(sug, (int, float)) else sug)
                c4.markdown("📤 **Enviada**")
            else:
                sug_key = f"sug_{key}"
                com_key = f"com_{key}"
                with c3:
                    sug_val = st.number_input(
                        "Tu Sugerencia",
                        min_value=0,
                        value=int(meta.get("meta_preliminar", 0)),
                        step=1,
                        key=sug_key,
                    )
                with c4:
                    comentario = st.text_input("Comentario", key=com_key, placeholder="Opcional...")

                if st.button(f"📤 Enviar Sugerencia — {dist}", key=f"btn_sug_{key}"):
                    metas[key]["sugerencia_gerente"] = int(sug_val)
                    metas[key]["sugerencia_gerente_usuario"] = username
                    metas[key]["comentario_gerente"] = comentario
                    metas[key]["estado"] = "pendiente_aprobacion"
                    save_metas(metas)
                    st.success(f"✅ Sugerencia enviada para **{dist}**.")
                    st.session_state["preserve_filters"] = True
                    st.rerun()

            st.divider()

    _render_descarga_metas(mis_metas)


# ── VISTA DIVISIONAL ──────────────────────────────────────────────

def _render_metas_divisional(df_user: pd.DataFrame, access: dict, username: str):
    st.caption("Aprueba, rechaza o ajusta las sugerencias de los gerentes.")

    regiones_divisional = access.get("region", [])
    if isinstance(regiones_divisional, str):
        regiones_divisional = [regiones_divisional]

    metas = load_metas()

    mis_metas = {
        k: v for k, v in metas.items()
        if v.get("region") in regiones_divisional
    }

    if not mis_metas:
        st.info("No hay metas disponibles para tus regiones aún.")
        return

    periodos = sorted(set(v["periodo"] for v in mis_metas.values()), reverse=True)
    periodo_sel = st.selectbox("Período", periodos, key="metas_div_periodo")

    metas_periodo = {k: v for k, v in mis_metas.items() if v["periodo"] == periodo_sel}

    if not metas_periodo:
        st.info("Sin metas para este período.")
        return

    pendientes = {k: v for k, v in metas_periodo.items() if v.get("estado") == "pendiente_aprobacion"}
    otros = {k: v for k, v in metas_periodo.items() if v.get("estado") != "pendiente_aprobacion"}

    if pendientes:
        st.markdown(f"### Pendientes de aprobación ({len(pendientes)})")

    decisiones = {}
    valores_propios = {}

    for key, meta in pendientes.items():
        dist = meta["distribuidor"]
        c1, c2, c3, c4 = st.columns([3, 2, 2, 3])
        c1.markdown(f"**{dist}** · {meta.get('region','')}")
        c2.metric("Meta Preliminar", f"{meta.get('meta_preliminar', 0):,}")
        sug = meta.get("sugerencia_gerente")
        c3.metric(
            "Sugerencia Gerente",
            f"{sug:,}" if isinstance(sug, (int, float)) else "—",
            help=meta.get("comentario_gerente", ""),
        )
        with c4:
            dec = st.radio(
                "Decisión",
                ["✅ Aprobar", "❌ Rechazar", "✏️ Valor Propio"],
                horizontal=True,
                key=f"dec_{key}",
            )
            decisiones[key] = dec
            if dec == "✏️ Valor Propio":
                valores_propios[key] = st.number_input(
                    "Meta propia",
                    min_value=0,
                    value=int(meta.get("meta_preliminar", 0)),
                    step=1,
                    key=f"vp_{key}",
                )
        st.divider()

    if pendientes and st.button("✅ Confirmar Metas Oficiales", type="primary"):
        st.session_state["tab_activo"] = "metas"
        for key, dec in decisiones.items():
            meta = metas[key]
            if dec == "✅ Aprobar":
                meta["meta_oficial"] = meta.get("sugerencia_gerente", meta["meta_preliminar"])
            elif dec == "❌ Rechazar":
                meta["meta_oficial"] = meta["meta_preliminar"]
            else:
                meta["meta_oficial"] = int(valores_propios.get(key, meta["meta_preliminar"]))
            meta["meta_oficial_usuario"] = username
            meta["estado"] = "oficial"
            metas[key] = meta
        save_metas(metas)
        st.success(f"✅ {len(decisiones)} metas actualizadas como **Oficiales**.")
        st.session_state["preserve_filters"] = True
        st.rerun()

    # Descarga Excel metas oficiales
    oficiales = {k: v for k, v in metas_periodo.items() if v.get("estado") == "oficial"}
    if oficiales:
        st.markdown("---")
        st.markdown(f"### Metas Oficiales ({len(oficiales)})")
        df_of = pd.DataFrame([
            {
                "Distribuidor":   v["distribuidor"],
                "Región":         v.get("region", ""),
                "Período":        v["periodo"],
                "Periodicidad":   v.get("periodicidad", ""),
                "Meta Oficial":   v.get("meta_oficial", v.get("meta_preliminar", "")),
                "Aprobado Por":   v.get("meta_oficial_usuario", ""),
            }
            for v in oficiales.values()
        ])
        st.dataframe(df_of, use_container_width=True, hide_index=True)

        try:
            import io
            buf = io.BytesIO()
            df_of.to_excel(buf, index=False, engine="openpyxl")
            buf.seek(0)
            st.download_button(
                "⬇️ Descargar Excel Metas Oficiales",
                data=buf,
                file_name=f"metas_oficiales_{periodo_sel}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except ImportError:
            st.warning("Instala openpyxl para habilitar la descarga de Excel: `pip install openpyxl`")

    if otros:
        with st.expander(f"Ver otros estados ({len(otros)})"):
            for key, meta in otros.items():
                st.markdown(
                    f"**{meta['distribuidor']}** — Estado: `{meta.get('estado','?')}` "
                    f"| Meta Preliminar: {meta.get('meta_preliminar', '—'):,}"
                )

    _render_descarga_metas(mis_metas)


# ── VISTA ASESOR ──────────────────────────────────────────────────

def _render_metas_asesor(df_user: pd.DataFrame, access: dict, username: str):
    st.caption("Consulta las metas asignadas a tus distribuidores.")

    distribuidores_asesor = access.get("distribuidores", [])
    metas = load_metas()

    mis_metas = {
        k: v for k, v in metas.items()
        if v.get("distribuidor") in distribuidores_asesor
    }

    if not mis_metas:
        st.info("Aún no hay metas asignadas para tus distribuidores. Consulta con tu gerente o divisional.")
        return

    periodos = sorted(set(v["periodo"] for v in mis_metas.values()), reverse=True)
    periodo_sel = st.selectbox("Período", periodos, key="metas_asesor_periodo")

    metas_periodo = {k: v for k, v in mis_metas.items() if v["periodo"] == periodo_sel}

    rows = []
    for v in metas_periodo.values():
        estado = v.get("estado", "preliminar")
        if estado == "oficial":
            meta_val = v.get("meta_oficial", v.get("meta_preliminar", "—"))
            estado_label = "✅ Oficial"
        else:
            meta_val = v.get("meta_preliminar", "—")
            estado_label = "🔄 Preliminar"
        rows.append({
            "Distribuidor": v["distribuidor"],
            "Período":      v["periodo"],
            "Meta":         meta_val,
            "Estado":       estado_label,
        })

    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Sin metas para este período.")

    _render_descarga_metas(mis_metas)


# =================================================================
# MAIN
# =================================================================

def main():
    if not check_login():
        return

    username = st.session_state["username"]
    region   = get_user_region(username)

    # Sidebar — info + cerrar sesión
    st.sidebar.markdown(f"**{username}** · {region}")
    if st.sidebar.button("🚪 Cerrar Sesión", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    st.sidebar.divider()

    try:
        procesada_folder = st.secrets["data"]["procesada_folder"]
    except (KeyError, FileNotFoundError):
        st.error("Verifica secrets.toml: falta [data] procesada_folder.")
        return

    with st.spinner("Cargando datos consolidados..."):
        df_all, _data_source = load_consolidated(procesada_folder)

    if _data_source:
        if _data_source.startswith("⚠️"):
            st.warning(_data_source)
        else:
            st.caption(_data_source)

    if df_all.empty:
        st.warning(
            "No se encontró el archivo consolidado. "
            "Ejecuta `05_Scripts/consolidar_data.py` primero y vuelve a cargar."
        )
        return

    df_user = filter_by_rbac(df_all, username)

    if df_user.empty:
        st.warning(f"No hay datos disponibles para el usuario '{username}'.")
        return

    if "tab_activo" not in st.session_state:
        st.session_state["tab_activo"] = "dashboard"

    df_filtered_sidebar = render_sidebar_filters(df_user)

    # Si el botón "Aplicar Filtros" fue presionado en este rerun, usar el resultado directo.
    # En cualquier otro rerun (metas, chat, etc.) usar el cache para preservar los filtros.
    _apply_just_pressed = st.session_state.pop("_apply_just_pressed", False)
    _tab_activo_now = st.session_state.get("tab_activo", "dashboard")
    if (
        not _apply_just_pressed
        and _tab_activo_now in ("chat", "metas")
        and "df_filtered_cache" in st.session_state
    ):
        df_filtered = st.session_state["df_filtered_cache"]
    else:
        df_filtered = df_filtered_sidebar

    if df_filtered.empty:
        st.warning("No hay datos con los filtros seleccionados.")
        return

    # Header corporativo Parawa
    _subtitle = (
        str(len(df_filtered)) + " registros · "
        + str(df_filtered['Distribuidor'].nunique()) + " distribuidor(es) · "
        + datetime.now().strftime('%d/%m/%Y')
    )
    render_page_header("Centro de Comando", _subtitle, username, region)

    tab_dashboard, tab_chat, tab_data, tab_metas, tab_powerbi = st.tabs(
        ["📊 Dashboard + Agente","💬 Chat Inteligente","📁 Explorar Datos","🎯 Metas","📈 Power BI"]
    )

    with tab_dashboard:
        render_dashboard(df_filtered, df_user=df_user, username=username)

    with tab_chat:
        render_chat(df_filtered, username)

    with tab_data:
        st.subheader("📁 Explorar Datos")
        st.caption("DataFrame filtrado completo")
        st.dataframe(df_filtered, use_container_width=True, hide_index=True)
        csv = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar CSV", csv,
            f"datos_{datetime.now().strftime('%Y%m%d_%H%M')}.csv","text/csv",
        )

    with tab_metas:
        render_metas_tab(df_user, df_all, username)

    with tab_powerbi:
        render_power_bi_tab()


def render_power_bi_tab():
    """Pestaña con librería de reportes Power BI desde JSON."""
    st.markdown(
        "<h2 style='color:#1A1A2E;font-weight:900;font-size:20px;margin:20px 0 4px 0;'>"
        "📊 Librería de Reportes Power BI</h2>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div style='background:#FFF8E1;border-left:4px solid #F57C00;border-radius:8px;"
        "padding:12px 16px;margin-bottom:16px;display:flex;align-items:center;gap:10px;'>"
        "<span style='font-size:20px;'>🔒</span>"
        "<span style='font-size:13px;font-weight:700;color:#374151;'>"
        "Los filtros del sidebar <strong>no aplican</strong> en esta vista. "
        "Usa los filtros dentro del reporte de Power BI directamente."
        "</span></div>",
        unsafe_allow_html=True
    )

    reportes = cargar_reportes_power_bi()

    if not reportes:
        st.warning("⚠️ No hay reportes configurados. Agrega uno abajo.")
    else:
        titulos = [r["titulo"] for r in reportes]
        idx = st.selectbox(
            "**Selecciona un Reporte:**",
            options=range(len(reportes)),
            format_func=lambda i: titulos[i],
            key="powerbi_selector",
        )
        reporte = reportes[idx]
        st.markdown(
            "<p style='color:#666;font-size:13px;font-weight:600;margin-bottom:12px;'>"
            + reporte["descripcion"] + "</p>",
            unsafe_allow_html=True
        )
        link = reporte.get("link", "")
        if link and link.startswith("http"):
            st.markdown(
                "<div style='position:relative;width:100%;height:900px;overflow:hidden;'>"
                "<iframe width='100%' height='924px' src='"
                + link
                + "' frameborder='0' allowFullScreen='true'"
                " style='position:absolute;top:0;left:0;border:none;'></iframe>"
                "<div style='position:absolute;bottom:0;left:0;width:100%;height:28px;"
                "background:#F4F6F8;z-index:10;'></div>"
                "</div>",
                unsafe_allow_html=True
            )
        else:
            st.info("💡 Este reporte aún no tiene link configurado. Edita el archivo `data/power_bi_reports.json`.")

    with st.expander("➕ Agregar Nuevo Reporte"):
        col1, col2 = st.columns(2)
        with col1:
            nuevo_titulo = st.text_input("Título del Reporte", placeholder="ej: Ventas Nacional", key="pbi_titulo")
        with col2:
            nuevo_desc = st.text_input("Descripción", placeholder="ej: Dashboard de ventas nacionales", key="pbi_desc")
        nuevo_link = st.text_input("Link Power BI", placeholder="https://app.powerbi.com/view?r=...", key="pbi_link")
        if st.button("💾 Agregar Reporte", use_container_width=True):
            if nuevo_titulo and nuevo_link and nuevo_link.startswith("http"):
                agregar_reporte(nuevo_titulo, nuevo_desc, nuevo_link)
                st.success("✅ Reporte agregado correctamente")
                st.rerun()
            else:
                st.error("❌ Completa título y un link HTTP válido")


if __name__ == "__main__":
    main()