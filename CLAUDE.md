# CLAUDE.md — Centro de Comando Parawa

Instrucciones y contexto para Claude Code al trabajar en este proyecto.

---

## Proyecto

**Centro de Comando Parawa** es un dashboard analítico inteligente para gestión comercial y seguimiento de ventas del Grupo Parawa. Desplegado en Streamlit Cloud:
- URL producción: `https://centro-comando-parawa.streamlit.app`
- Repositorio: `https://github.com/InformacionComercialParawa/centro-comando-parawa`

**Stack:**
- Frontend/Backend: Streamlit
- Datos: Pandas, PyArrow (Parquet), Google Drive API
- Visualizaciones: Plotly
- IA: Google Gemini API (agentes analíticos)
- Auth: bcrypt + RBAC por región
- Mobile: PWA (manifest.json + sw.js)

---

## Estructura del proyecto

```
centro-comando-parawa/
├── app.py                        # Aplicación principal (~3000 líneas)
├── requirements.txt              # Dependencias Python
├── .streamlit/
│   ├── config.toml               # Tema visual (NO modificar colores sin consultar)
│   └── secrets.toml              # Credenciales (NO commitear, está en .gitignore)
├── agents/
│   ├── base_agent.py
│   ├── sales_analyst.py          # Agente análisis de ventas
│   ├── regional_analyst.py       # Agente análisis regional
│   ├── national_analyst.py       # Agente análisis nacional
│   ├── goals_engine.py           # Motor de metas
│   └── kpi_engine.py             # Motor de KPIs
├── assets/
│   ├── Logo_Parawa*.png          # Logos (no modificar)
│   ├── manifest.json             # PWA manifest
│   └── sw.js                     # Service Worker
└── data/
    └── power_bi_reports.json     # Config reportes Power BI
```

---

## Secciones principales de app.py

| Líneas aprox. | Sección |
|---------------|---------|
| 1–54 | Imports, config Power BI |
| 55–63 | `st.set_page_config()` |
| 65–148 | PWA + botón hamburguesa |
| 239–315 | Funciones de gráficos Plotly |
| 452–559 | RBAC — autenticación y filtrado por región |
| 565–785 | Carga de datos (Parquet, Google Drive) |
| 792–830 | Gestión de metas |
| 978–1129 | Filtros sidebar |
| 1134–1384 | KPIs estratégicos + proyección anual |
| 1390–2790 | Dashboard principal (5 tabs) |
| 2791–fin | `def main()` |

---

## Colores de marca

| Nombre | Hex |
|--------|-----|
| Cian principal | `#00ACC1` |
| Cian oscuro (hover) | `#00838F` |
| Fondo app | `#F4F6F8` |
| Fondo cards | `#FFFFFF` |
| Texto | `#212121` |

---

## Convenciones de código

- Streamlit: todo el código de UI va dentro de `def main()` excepto la inicialización de `st.session_state` global y `st.set_page_config()` que van fuera.
- CSS/HTML: usar `st.markdown(..., unsafe_allow_html=True)`.
- Datos sensibles: NUNCA en el código. Van en `.streamlit/secrets.toml`.
- No agregar docstrings ni type hints a funciones existentes que no los tengan.
- No refactorizar código que no se está modificando.

---

## Git y deploy

```bash
# Flujo estándar
git add <archivos>
git commit -m "tipo: descripción corta"
git push
# Streamlit Cloud redeploya automáticamente en ~2 minutos
```

**Tipos de commit:** `feat`, `fix`, `refactor`, `docs`, `chore`

**Identidad git configurada:**
- Nombre: Roberto Castaño
- Email: coordinador.informacioncomercial@grupoparawa.com

**NUNCA hacer:**
- `git push --force`
- Commitear `.streamlit/secrets.toml`
- Commitear archivos `.parquet` o `.xlsx` (están en .gitignore)

---

## PWA (Progressive Web App)

- `assets/manifest.json` — configuración de instalación
- `assets/sw.js` — service worker para offline
- El bloque PWA en `app.py` (línea ~65) inyecta meta tags y detecta si la app corre instalada (`window.isPWA`)
- El botón hamburguesa solo aparece cuando `isPWA = True`

---

## Preferencias del usuario

- Roberto trabaja con Claude Code desde VS Code en Windows
- Prefiere que Claude ejecute `git add`, `git commit` y `git push` automáticamente al terminar cambios
- No explicar paso a paso lo que se va a hacer — ejecutar directo y mostrar resultado
- Respuestas cortas y directas
- Si algo ya fue commiteado/pusheado, no volver a hacerlo
