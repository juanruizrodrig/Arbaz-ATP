import streamlit as st
from pathlib import Path
import pandas as pd
from datetime import date
import json
from sqlalchemy import create_engine, text
import os
import textwrap


# =========================
# Configuración
# =========================
st.set_page_config(page_title="ATP", layout="wide")

def inject_fixed_top_nav():
    current = st.session_state.get("page", "Expedientes")

    def active_cls(name: str) -> str:
        # Clase base + clase de activo si coincide la página
        return "atp-nav-link atp-nav-active" if current == name else "atp-nav-link"

    html = (
        "<style>"
        ".atp-top-nav{position:fixed;top:12px;left:12px;z-index:999999;"
        "display:flex;gap:8px;}"
        ".atp-nav-link{display:inline-block;background:#000000;border:2px solid #ffffff;"
        "border-radius:8px;padding:8px 14px;font-size:14px;line-height:1;cursor:pointer;"
        "box-shadow:0 2px 6px rgba(0,0,0,0.4);color:#ffffff;font-weight:700;text-decoration:none;"
        "transition:background 0.15s ease-in-out;}"
        ".atp-nav-link:hover{background:#222222;}"
        ".atp-nav-active{background:#444444;}"
        "</style>"
        '<div class="atp-top-nav">'
        f'<a href="?page=Expedientes&src=main" class="{active_cls("Expedientes")}" '
        'target="_self">Expedientes</a>'
        f'<a href="?page=Nuevo%20expediente" class="{active_cls("Nuevo expediente")}" '
        'target="_self">Nuevo expediente</a>'
        f'<a href="?page=Dashboards" class="{active_cls("Dashboards")}" '
        'target="_self">Dashboards</a>'
        f'<a href="?page=Personalización" class="{active_cls("Personalización")}" '
        'target="_self">Personalización</a>'
        '</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


def inject_table_styles():
    st.markdown(
        """
        <style>
        /* -------------------------
           AgGrid (st_aggrid) styling
           ------------------------- */

        .ag-theme-streamlit .ag-header {
            background-color: #202020 !important;
        }
        .ag-theme-streamlit .ag-header-cell,
        .ag-theme-streamlit .ag-header-group-cell {
            background-color: #202020 !important;
            color: #f5f5f5 !important;
            font-weight: 700 !important;
            border-bottom: 2px solid #444444 !important;
            border-right: 1px solid #444444 !important;
        }

        .ag-header-cell-label {
            white-space: normal !important;
            line-height: 1.3 !important;
            padding-top: 8px !important;
            padding-bottom: 8px !important;
        }
        .ag-header-cell-text {
            white-space: normal !important;
            word-wrap: break-word !important;
        }

        .ag-header {
            height: auto !important;
            min-height: 60px !important;
        }
        .ag-header-row {
            height: auto !important;
            min-height: 60px !important;
        }

        /* Forzar bordes verticales en celdas */
        .ag-theme-streamlit .ag-cell {
            border-right: 1px solid #444444 !important;
            border-bottom: 1px solid #333333 !important;
        }
        
        .ag-theme-streamlit .ag-row {
            border-bottom: 1px solid #333333 !important;
        }

        /* -------------------------
           Streamlit dataframe styling
           ------------------------- */

        div[data-testid="stDataFrame"] thead tr th {
            background-color: #202020 !important;
            color: #f5f5f5 !important;
            font-weight: 700 !important;
            white-space: normal !important;
            line-height: 1.15 !important;
            padding-top: 10px !important;
            padding-bottom: 10px !important;
            border-right: 1px solid rgba(255,255,255,0.2) !important;
        }

        div[data-testid="stDataFrame"] td,
        div[data-testid="stDataFrame"] th {
            border-right: 1px solid rgba(255,255,255,0.2) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_table_styles()

FRIENDLY_HEADERS = {
    "id": "ID",
    "num_aviso": "Nº Aviso",
    "expediente": "Expediente",
    "aseguradora": "Aseguradora",
    "fecha_aviso": "Fecha aviso",
    "fecha_siniestro": "Fecha siniestro",
    "fecha_efecto": "Fecha efecto",
    "nombre_asegurado": "Nombre asegurado",
    "apellidos": "Apellidos",
    "direccion": "Dirección",
    "cp": "Código postal",
    "ciudad": "Ciudad",
    "telefonos": "Teléfonos",
    "poliza": "Póliza",
    "observaciones": "Observaciones",
    "estado": "Estado",
    "responsable": "Responsable",
    "creation_state": "Origen alta",
    "created_at": "Creado el",
    "coincidencias": "Coincidencias",
}


# Estilos internos de la app
st.markdown(
    """
    <style>
    .page-title {
        text-align: center;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Logo
LOGO_PATH = Path(__file__).parent / "assets" / "1710704149063.jfif"
st.image(str(LOGO_PATH), width=180)


# Añadimos el engine + helpers
@st.cache_resource
def get_engine():
    host = os.environ["DB_HOST"]
    port = os.environ.get("DB_PORT", "5432")
    name = os.environ["DB_NAME"]
    user = os.environ["DB_USER"]
    pwd  = os.environ["DB_PASSWORD"]

    url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{name}"
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=5,       # ok para ~10 concurrentes
        max_overflow=10,
        pool_timeout=30,
    )

engine = get_engine()

def fetch_all(sql: str, params: dict | None = None) -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params or {}).mappings().all()
        return [dict(r) for r in rows]

def execute(sql: str, params: dict | None = None) -> None:
    with engine.begin() as conn:
        conn.execute(text(sql), params or {})

def execute_returning_one(sql: str, params: dict | None = None) -> dict:
    with engine.begin() as conn:
        row = conn.execute(text(sql), params or {}).mappings().first()
        if row is None:
            raise RuntimeError("No rows returned")
        return dict(row)

# =========================
# Query params robusto (compatible)
# =========================
def qp_get() -> dict:
    # Streamlit nuevo
    if hasattr(st, "query_params"):
        return dict(st.query_params)
    # Streamlit viejo
    return st.experimental_get_query_params()

def qp_set(**kwargs):
    # Streamlit nuevo
    if hasattr(st, "query_params"):
        st.query_params.clear()
        for k, v in kwargs.items():
            st.query_params[k] = str(v)
    else:
        st.experimental_set_query_params(**{k: str(v) for k, v in kwargs.items()})

qp = qp_get()

# -------------------------
# Inicialización robusta de estado
# -------------------------
DEFAULT_PAGE = "Expedientes"

# 1) Page
if "page" in qp:
    st.session_state.page = qp["page"] if isinstance(qp["page"], str) else qp["page"][0]
elif "page" not in st.session_state:
    st.session_state.page = DEFAULT_PAGE

# 2) selected_id
if "id" in qp:
    st.session_state.selected_id = qp["id"] if isinstance(qp["id"], str) else qp["id"][0]
elif "selected_id" not in st.session_state:
    st.session_state.selected_id = None

# 3) src
if "src" in qp:
    st.session_state.src = qp["src"] if isinstance(qp["src"], str) else qp["src"][0]
elif "src" not in st.session_state:
    st.session_state.src = "main"

page = st.session_state.page

inject_fixed_top_nav()

# =========================
# Navegación superior
# =========================

def go(page_name: str, selected_id: str | None = None, src: str | None = None):
    st.session_state.page = page_name

    if src is not None:
        st.session_state.src = src

    if selected_id is None:
        qp_set(page=page_name, src=st.session_state.get("src", "main"))
    else:
        st.session_state.selected_id = str(selected_id)
        qp_set(page=page_name, id=str(selected_id), src=st.session_state.get("src", "main"))

    st.rerun()

def page_title(text: str):
    st.markdown(f"<h2 class='page-title'>{text}</h2>", unsafe_allow_html=True)


st.markdown("---")

# =========================
# Normalización de datos
def normalize_cell(x):
    """Convierte valores de AgGrid a tipos compatibles con psycopg2."""
    # None / NaN
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None

    # pandas NaT (clave del bug)
    if pd.isna(x):
        return None

    # "NaT" como string (a veces aparece tras AgGrid)
    if isinstance(x, str):
        s = x.strip()
        if s.lower() == "nat" or s == "":
            return None
        return s

    # Si viene dict (AgGrid a veces devuelve objetos)
    if isinstance(x, dict):
        for k in ("date", "value", "label"):
            if k in x:
                return normalize_cell(x[k])
        return json.dumps(x, ensure_ascii=False)

    if isinstance(x, list):
        return json.dumps(x, ensure_ascii=False)

    if isinstance(x, pd.Timestamp):
        return x.date()

    return x


def normalize_df_for_db(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(normalize_cell)

    for c in ("fecha_aviso", "fecha_siniestro", "fecha_efecto"):
        if c in out.columns:
            dt = pd.to_datetime(out[c], errors="coerce")
            out[c] = dt.dt.date
            out[c] = out[c].where(dt.notna(), None)  # usa dt.notna(), no out[c].notna()

    return out


# ========================
# IMPRESION del aviso
# ========================

from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

def _fmt(x):
    if x is None:
        return ""
    # pandas Timestamp / date
    try:
        import pandas as pd
        if isinstance(x, pd.Timestamp):
            x = x.date()
    except Exception:
        pass
    return str(x)

def df_to_table_data(df: pd.DataFrame, columns: list[str], headers: list[str]) -> list[list[str]]:
    data = [headers]
    for _, r in df.iterrows():
        row = []
        for c in columns:
            row.append(_fmt(r.get(c)))
        data.append(row)
    return data


def make_pdf_table(data: list[list[str]], col_widths=None) -> Table:
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    return t

def build_expediente_pdf_with_matches(row: dict, matches: dict[str, pd.DataFrame]) -> bytes:
    """
    matches: dict con claves: 'NOMBRE','DIR','TEL','POLIZA'
             y valores: DataFrame con coincidencias (pueden estar vacíos)
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="Expediente")

    styles = getSampleStyleSheet()
    story = []

    # --- Cabecera expediente (igual que ya tienes)
    story.append(Paragraph("Detalle del expediente", styles["Title"]))
    story.append(Spacer(1, 12))

    data = [
        ["ID", _fmt(row.get("id"))],
        ["Nº Aviso", _fmt(row.get("num_aviso"))],
        ["Expediente", _fmt(row.get("expediente"))],
        ["Aseguradora", _fmt(row.get("aseguradora"))],
        ["Estado", _fmt(row.get("estado"))],
        ["Responsable", _fmt(row.get("responsable"))],
        ["Fecha aviso", _fmt(row.get("fecha_aviso"))],
        ["Fecha siniestro", _fmt(row.get("fecha_siniestro"))],
        ["Fecha efecto", _fmt(row.get("fecha_efecto"))],
        ["Nombre asegurado", _fmt(row.get("nombre_asegurado"))],
        ["Apellidos", _fmt(row.get("apellidos"))],
        ["Dirección", _fmt(row.get("direccion"))],
        ["CP", _fmt(row.get("cp"))],
        ["Ciudad", _fmt(row.get("ciudad"))],
        ["Teléfonos", _fmt(row.get("telefonos"))],
        ["Póliza", _fmt(row.get("poliza"))],
        ["Observaciones", _fmt(row.get("observaciones"))],
    ]

    table = Table(data, colWidths=[140, 380])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (1, 0), colors.whitesmoke),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)

    # --- Sección coincidencias
    story.append(Spacer(1, 16))
    story.append(Paragraph("Coincidencias detectadas", styles["Heading2"]))
    story.append(Spacer(1, 6))

    any_match = any(df is not None and not df.empty for df in matches.values())
    if not any_match:
        story.append(Paragraph("No se han encontrado coincidencias para este expediente.", styles["Normal"]))
    else:
        # Config por tipo (qué coincide + columnas a mostrar)
        configs = [
            ("NOMBRE", "Coincidencia por Nombre y Apellidos",
             f"Coincide con: {_fmt(row.get('nombre_asegurado'))} {_fmt(row.get('apellidos'))}".strip(),
             ["num_aviso", "expediente", "fecha_aviso", "aseguradora", "estado", "responsable"],
             ["Nº Aviso", "Expediente", "Fecha aviso", "Aseguradora", "Estado", "Responsable"]),
            ("DIR", "Coincidencia por Dirección",
             f"Coincide con: {_fmt(row.get('direccion'))}",
             ["num_aviso", "expediente", "fecha_aviso", "aseguradora", "estado", "responsable"],
             ["Nº Aviso", "Expediente", "Fecha aviso", "Aseguradora", "Estado", "Responsable"]),
            ("TEL", "Coincidencia por Teléfono",
             f"Coincide con: {_fmt(row.get('telefonos'))}",
             ["num_aviso", "expediente", "fecha_aviso", "aseguradora", "estado", "responsable"],
             ["Nº Aviso", "Expediente", "Fecha aviso", "Aseguradora", "Estado", "Responsable"]),
            ("POLIZA", "Coincidencia por Póliza",
             f"Coincide con: {_fmt(row.get('poliza'))}",
             ["num_aviso", "expediente", "fecha_aviso", "aseguradora", "estado", "responsable"],
             ["Nº Aviso", "Expediente", "Fecha aviso", "Aseguradora", "Estado", "Responsable"]),
        ]

        for key, title, detail_line, cols, headers in configs:
            dfm = matches.get(key)
            if dfm is None or dfm.empty:
                continue

            story.append(Spacer(1, 10))
            story.append(Paragraph(title, styles["Heading3"]))
            story.append(Paragraph(detail_line, styles["Normal"]))
            story.append(Spacer(1, 6))

            tdata = df_to_table_data(dfm, cols, headers)
            story.append(make_pdf_table(tdata, col_widths=[65, 90, 70, 95, 60, 80]))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Generado desde la aplicación ATP.", styles["Normal"]))

    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes

# =========================
# Helpers DB
# =========================
@st.cache_data(ttl=30)
def load_main():
    rows = fetch_all(
        """
        SELECT
            m.id,
            m.num_aviso,
            m.expediente,
            m.aseguradora,
            m.fecha_aviso,
            m.fecha_siniestro,
            m.fecha_efecto,
            m.nombre_asegurado,
            m.apellidos,
            m.direccion,
            m.cp,
            m.ciudad,
            m.telefonos,
            m.poliza,
            m.observaciones,
            m.estado,
            m.creation_state,
            m.responsable,
            COALESCE(
                NULLIF(
                    CONCAT_WS(
                        E'\n',
                        CASE
                            WHEN m.nombre_asegurado IS NOT NULL
                             AND m.apellidos IS NOT NULL
                             AND EXISTS (
                                SELECT 1 FROM main x
                                WHERE x.id <> m.id
                                  AND x.nombre_asegurado = m.nombre_asegurado
                                  AND x.apellidos = m.apellidos
                             )
                            THEN
                                'NOMBRE: ' ||
                                (
                                    SELECT COUNT(*) FROM main x
                                    WHERE x.id <> m.id
                                      AND x.nombre_asegurado = m.nombre_asegurado
                                      AND x.apellidos = m.apellidos
                                )
                        END,
                        CASE
                            WHEN m.telefonos IS NOT NULL AND TRIM(m.telefonos) <> ''
                             AND EXISTS (
                                SELECT 1 FROM main x
                                WHERE x.id <> m.id
                                  AND x.telefonos = m.telefonos
                             )
                            THEN
                                'TEL: ' ||
                                (
                                    SELECT COUNT(*) FROM main x
                                    WHERE x.id <> m.id
                                      AND x.telefonos = m.telefonos
                                )
                        END,
                        CASE
                            WHEN m.direccion IS NOT NULL AND TRIM(m.direccion) <> ''
                             AND EXISTS (
                                SELECT 1 FROM main x
                                WHERE x.id <> m.id
                                  AND x.direccion = m.direccion
                             )
                            THEN
                                'DIR: ' ||
                                (
                                    SELECT COUNT(*) FROM main x
                                    WHERE x.id <> m.id
                                      AND x.direccion = m.direccion
                                )
                        END,
                        CASE
                            WHEN m.poliza IS NOT NULL AND TRIM(m.poliza) <> ''
                             AND EXISTS (
                                SELECT 1 FROM main x
                                WHERE x.id <> m.id
                                  AND x.poliza = m.poliza
                             )
                            THEN
                                'POLIZA: ' ||
                                (
                                    SELECT COUNT(*) FROM main x
                                    WHERE x.id <> m.id
                                      AND x.poliza = m.poliza
                                )
                        END
                    ),
                    ''
                ),
                'NO'
            ) AS coincidencias
        FROM main m
        ORDER BY m.id DESC;
        """
    )
    df = pd.DataFrame(rows)
    if not df.empty and "id" in df.columns:
        df = df.sort_values("id", ascending=False, kind="mergesort")
    return df


@st.cache_data(ttl=30)
def load_pending():
    rows = fetch_all("""
        SELECT
            id,
            expediente,
            aseguradora,
            fecha_aviso,
            fecha_siniestro,
            fecha_efecto,
            nombre_asegurado,
            apellidos,
            direccion,
            cp,
            ciudad,
            telefonos,
            poliza,
            observaciones,
            estado,
            creation_state,
            responsable,
            created_at
        FROM main_pending
        WHERE creation_state = 'auto_new'
        ORDER BY created_at DESC, id DESC;
    """)
    return pd.DataFrame(rows)

@st.cache_data(ttl=30)
def load_one_pending(exp_id: int) -> pd.DataFrame:
    rows = fetch_all("""
        SELECT
            id,
            expediente,
            aseguradora,
            fecha_aviso,
            fecha_siniestro,
            fecha_efecto,
            nombre_asegurado,
            apellidos,
            direccion,
            cp,
            ciudad,
            telefonos,
            poliza,
            observaciones,
            estado,
            creation_state,
            responsable,
            created_at
        FROM main_pending
        WHERE id = :id
        LIMIT 1;
    """, {"id": exp_id})

    return pd.DataFrame(rows)

def update_pending(exp_id: int, payload: dict):
    execute("""
        UPDATE main_pending
        SET
            expediente = :expediente,
            aseguradora = :aseguradora,
            fecha_aviso = :fecha_aviso,
            fecha_siniestro = :fecha_siniestro,
            fecha_efecto = :fecha_efecto,
            nombre_asegurado = :nombre_asegurado,
            apellidos = :apellidos,
            direccion = :direccion,
            cp = :cp,
            ciudad = :ciudad,
            telefonos = :telefonos,
            poliza = :poliza,
            observaciones = :observaciones,
            estado = :estado,
            responsable = :responsable
        WHERE id = :id
    """, {**payload, "id": exp_id})


def insert_expediente(data: dict):
    return execute_returning_one("""
        INSERT INTO main (
            expediente,
            aseguradora,
            fecha_aviso,
            fecha_siniestro,
            fecha_efecto,
            nombre_asegurado,
            apellidos,
            direccion,
            cp,
            ciudad,
            telefonos,
            poliza,
            observaciones,
            estado,
            responsable
        )
        VALUES (
            :expediente, :aseguradora, :fecha_aviso, :fecha_siniestro, :fecha_efecto,
            :nombre_asegurado, :apellidos, :direccion, :cp, :ciudad,
            :telefonos, :poliza, :observaciones, :estado, :responsable
        )
        RETURNING id, num_aviso, expediente;
    """, data)

@st.cache_data(ttl=30)
def load_picklists():
    # Valores únicos reales en BD (limpios y ordenados)
    aseg = fetch_all("""
        SELECT DISTINCT aseguradora
        FROM main
        WHERE aseguradora IS NOT NULL AND TRIM(aseguradora) <> ''
        ORDER BY aseguradora;
    """)
    resp = fetch_all("""
        SELECT DISTINCT responsable
        FROM main
        WHERE responsable IS NOT NULL AND TRIM(responsable) <> ''
        ORDER BY responsable;
    """)
    est = fetch_all("""
        SELECT DISTINCT estado
        FROM main
        WHERE estado IS NOT NULL AND TRIM(estado) <> ''
        ORDER BY estado;
    """)

    aseguradoras = [r["aseguradora"] for r in aseg]
    responsables = [r["responsable"] for r in resp]
    estados = [r["estado"] for r in est]

    return aseguradoras, responsables, estados

@st.cache_data(ttl=30)
def load_one(exp_id: int) -> pd.DataFrame:
    row = fetch_all("""
        SELECT
            id,
            num_aviso,
            expediente,
            aseguradora,
            fecha_aviso,
            fecha_siniestro,
            fecha_efecto,
            nombre_asegurado,
            apellidos,
            direccion,
            cp,
            ciudad,
            telefonos,
            poliza,
            observaciones,
            estado,
            creation_state,
            responsable
        FROM main
        WHERE id = :id
    """, {"id": exp_id})
    return pd.DataFrame(row)

@st.cache_data(ttl=30)
def load_coincidences_name(nombre: str, apellidos: str, current_id: int) -> pd.DataFrame:
    """Busca por coincidencia exacta de nombre Y apellidos."""
    if not nombre or not apellidos:
        return pd.DataFrame()
    rows = fetch_all("""
        SELECT id, num_aviso, expediente, fecha_aviso, aseguradora, estado, responsable
        FROM main
        WHERE nombre_asegurado = :nombre
        AND apellidos = :apellidos
        AND id <> :current_id
        ORDER BY fecha_aviso DESC;
    """, {"nombre": nombre, "apellidos": apellidos, "current_id": current_id})
    return pd.DataFrame(rows)


ALLOWED_MATCH_FIELDS = {"direccion", "telefonos", "poliza"}

@st.cache_data(ttl=30)
def load_matches_by_field(field_name: str, value: str, current_id: int) -> pd.DataFrame:
    """Busca por coincidencia en un campo específico (direccion, telefonos, poliza)."""
    if field_name not in ALLOWED_MATCH_FIELDS:
        raise ValueError("Invalid field_name")

    if not value or str(value).strip() in ["", "None", "nan"]:
        return pd.DataFrame()

    query = f"""
        SELECT id, num_aviso, expediente, fecha_aviso, aseguradora, estado, responsable
        FROM main
        WHERE {field_name} = :value AND id <> :current_id
        ORDER BY fecha_aviso DESC;
    """
    rows = fetch_all(query, {"value": str(value).strip(), "current_id": current_id})
    return pd.DataFrame(rows)

def coincidencias_msg(n: int, label_value: str) -> str:
    if n == 1:
        return f"Se ha encontrado 1 coincidencia con: {label_value}"
    return f"Se han encontrado {n} coincidencias con: {label_value}"

# -------------------------
# Definiciones globales para AgGrid (antes de las páginas)
# -------------------------
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

row_style_zebra = JsCode("""
function(params) {
    const idx = params.node.rowIndex;
    const baseStyle = {
        color: '#f2f2f2',
        borderRight: '1px solid #444444',    // Mantén esto
        borderBottom: '1px solid #333333'    // Mantén esto
    };

    if (idx % 2 === 0) {
        return {
            backgroundColor: '#161616',
            ...baseStyle
        };
    } else {
        return {
            backgroundColor: '#262626',
            ...baseStyle
        };
    }
}
""")

# Estilo tipo link para la celda id
link_style_id = JsCode("""
function(params){
    if (params.colDef.field === 'id') {
        return {
            textDecoration: 'underline',
            cursor: 'pointer',
            color: '#1f77b4',
            borderRight: '1px solid #444444'
        };
    }
    return {};
}
""")

# Navegación: sólo clic en la celda id
on_cell_clicked_id = JsCode("""
function(e) {
    try {
    // Sólo reaccionar si haces clic en la columna id
    if (!e || !e.colDef || e.colDef.field !== 'id') return;

    // El id fiable es e.data.id (e.value puede venir vacío)
    const id = (e.data && e.data.id != null) ? e.data.id : e.value;
    if (id === undefined || id === null || String(id).trim() === "") return;

    const src =
        (e.api &&
        e.api.gridOptionsWrapper &&
        e.api.gridOptionsWrapper.gridOptions &&
        e.api.gridOptionsWrapper.gridOptions.context)
        ? e.api.gridOptionsWrapper.gridOptions.context.src
        : "main";

    const baseUrl = window.location.origin + window.location.pathname;
    const newUrl = baseUrl
        + "?page=Detalle&id=" + encodeURIComponent(id)
        + "&src=" + encodeURIComponent(src);

    // Misma pestaña
    window.location.href = newUrl;
    } catch (err) {
    console.error("Navigation error:", err);
    }
}
""")

cellstyle_estado = JsCode("""
function(params) {
    let raw = params.value;
    if (raw && typeof raw === 'object' && raw.value !== undefined) {
    raw = raw.value;
    }
    if (raw === null || raw === undefined) return {};

    const v = String(raw).trim().toLowerCase();
    let style = { color: 'black', fontWeight: '600' };

    if (v === 'cerrado') style.backgroundColor = '#d7f7d4';
    else if (v === 'nuevo') style.backgroundColor = '#d6e9ff';
    else if (v === 'en curso') style.backgroundColor = '#ffe1b3';

    return style;
}
""")

# -------------------------
# FUNCIÓN BUILD_GRID (GLOBAL)
# -------------------------
DATE_COLS = ["fecha_aviso", "fecha_siniestro", "fecha_efecto"]

def build_grid(
    df_input: pd.DataFrame,
    grid_key: str,
    height: int = 550,
    src: str = "main",
):
    df_grid_local = df_input.copy()

    # Normalización de fechas a texto
    for c in DATE_COLS:
        if c in df_grid_local.columns:
            df_grid_local[c] = (
                pd.to_datetime(df_grid_local[c], errors="coerce")
                .dt.strftime("%Y-%m-%d")
            )
            df_grid_local[c] = df_grid_local[c].fillna("")

    gb = GridOptionsBuilder.from_dataframe(df_grid_local)
    
    # Configuraciones generales del grid
    gb.configure_grid_options(
        getRowStyle=row_style_zebra,
        enableCellTextSelection=True,
        ensureDomOrder=True,
        headerHeight=None,
        groupHeaderHeight=None,
        context={"src": src},
        suppressRowClickSelection=False,
        suppressCellFocus=False,
        domLayout="normal",
    )

    # Configuración por defecto de columnas
    gb.configure_default_column(
        editable=False,
        resizable=True,
        sortable=True,
        filter=True,
        wrapHeaderText=True,
        autoHeaderHeight=True,
        minWidth=130,
        cellStyle={'borderRight': '1px solid #444444'},
    )

    # Aplicar todos los headerName del diccionario
    for col_name, friendly_name in FRIENDLY_HEADERS.items():
        if col_name in df_grid_local.columns:
            gb.configure_column(
                col_name,
                headerName=friendly_name,
                editable=False,
            )

    # Configuraciones específicas que necesitan cellStyle
    if "id" in df_grid_local.columns:
        df_grid_local["id"] = df_grid_local["id"].astype(str)
        gb.configure_column(
            "id",
            headerName="ID",
            cellStyle=link_style_id,
        )

    if "estado" in df_grid_local.columns:
        gb.configure_column(
            "estado",
            headerName="Estado",
            cellStyle=cellstyle_estado,
        )

    if "coincidencias" in df_grid_local.columns:
        gb.configure_column(
            "coincidencias",
            headerName="Coincidencias",
            wrapText=True,
            autoHeight=True,
        )

    if "num_aviso" in df_grid_local.columns and df_grid_local["num_aviso"].astype(str).str.strip().eq("").all():
        gb.configure_column("num_aviso", hide=True)

    gb.configure_selection("single", use_checkbox=False)

    grid_response = AgGrid(
        df_grid_local,
        gridOptions=gb.build(),
        height=height,
        fit_columns_on_grid_load=False,
        allow_unsafe_jscode=True,
        update_on=["selectionChanged"],
        theme="streamlit",
        key=grid_key,
    )
    return grid_response

# =========================
# PÁGINA: Expedientes
# =========================
if page == "Expedientes":
    page_title("📄 Expedientes en tramitación")

    try:
        df_original = load_main()
    except Exception as e:
        st.error("❌ Error cargando datos")
        st.exception(e)
        st.stop()

    if df_original.empty:
        st.info("No hay expedientes registrados.")
        st.stop()

    df = df_original.copy()

    # -------------------------
    # Filtros dinámicos
    # -------------------------
    st.markdown("### 🔍 Filtros")

    # Inicializar filtros en session_state
    if "filters" not in st.session_state:
        st.session_state.filters = []

    # Campos filtrables
    filterable_fields = {
        "Expediente": "expediente",
        "Nº Aviso": "num_aviso",
        "Aseguradora": "aseguradora",
        "Ciudad": "ciudad",
        "Código Postal": "cp",
        "Nombre asegurado": "nombre_asegurado",
        "Apellidos": "apellidos",
        "Fecha aviso": "fecha_aviso",
        "Fecha siniestro": "fecha_siniestro",
        "Fecha efecto": "fecha_efecto",
    }

    operators = {
        "contiene": "contains",
        "igual a": "eq",
        "distinto de": "ne",
        "empieza por": "startswith",
        "mayor que": "gt",
        "menor que": "lt",
    }

    # Render filtros existentes
    for i, f in enumerate(st.session_state.filters):
        cols = st.columns([3, 3, 4, 1])

        with cols[0]:
            field_label = st.selectbox(
                "Campo",
                list(filterable_fields.keys()),
                index=list(filterable_fields.keys()).index(f["field"]),
                key=f"field_{i}",
            )

        with cols[1]:
            op_label = st.selectbox(
                "Operador",
                list(operators.keys()),
                index=list(operators.keys()).index(f["op"]),
                key=f"op_{i}",
            )

        with cols[2]:
            value = st.text_input(
                "Valor",
                value=f["value"],
                key=f"value_{i}",
            )

        with cols[3]:
            if st.button("❌", key=f"del_{i}"):
                st.session_state.filters.pop(i)
                st.rerun()

        f.update({"field": field_label, "op": op_label, "value": value})

    # Botón añadir filtro
    if st.button("➕ Añadir filtro"):
        st.session_state.filters.append(
            {"field": "Aseguradora", "op": "contiene", "value": ""}
        )
        st.rerun()

    # -------------------------
    # Aplicar filtros (AND lógico)
    # -------------------------
    for f in st.session_state.filters:
        col = filterable_fields[f["field"]]
        op = operators[f["op"]]
        val = f["value"]

        if not val:
            continue

        series = df[col].astype(str)

        if op == "contains":
            df = df[series.str.contains(val, case=False, na=False)]
        elif op == "startswith":
            df = df[series.str.startswith(val, na=False)]
        elif op == "eq":
            df = df[series == val]
        elif op == "ne":
            df = df[series != val]
        elif op == "gt":
            df = df[pd.to_numeric(series, errors="coerce") > float(val)]
        elif op == "lt":
            df = df[pd.to_numeric(series, errors="coerce") < float(val)]

    st.markdown("### 📑 Resultados")

    if df.empty:
        st.warning("No hay resultados con los filtros actuales.")
        st.stop()


    # --- TABLA GENERAL (main)
    grid_response = build_grid(
        df,
        grid_key="expedientes_grid",
        height=550,
        src="main",
    )

    # Navegación al detalle: si hay fila seleccionada, vamos a "Detalle"
    selected_main = grid_response.get("selected_rows", [])

    # Normalizamos selected_main a lista de diccionarios
    if isinstance(selected_main, pd.DataFrame):
        selected_main = selected_main.to_dict("records")
    elif selected_main is None:
        selected_main = []

    if isinstance(selected_main, list) and len(selected_main) > 0:
        first_row = selected_main[0]
        # Por seguridad, soportamos tanto dict como objetos raros
        if isinstance(first_row, dict):
            sel_id = first_row.get("id")
        else:
            # fallback muy defensivo
            sel_id = getattr(first_row, "id", None)

        if sel_id is not None and str(sel_id).strip() != "":
            go("Detalle", sel_id, src="main")

    edited_df = pd.DataFrame(grid_response["data"])


# --- NUEVA TABLA: Expedientes por aprobar (tabla main_pending)
    st.markdown("---")
    page_title("🟠 Expedientes por aprobar")

    try:
        df_pending = load_pending()
    except Exception as e:
        st.error("❌ Error cargando pendientes (main_pending)")
        st.exception(e)
        df_pending = pd.DataFrame()

    if df_pending.empty:
        st.info("No hay expedientes pendientes en main_pending.")
    else:
        # Columnas a mostrar (solo estas)
        PENDING_COLS = [
            "id",
            "expediente",
            "aseguradora",
            "fecha_aviso",
            "fecha_siniestro",
            "fecha_efecto",
            "nombre_asegurado",
            "apellidos",
            "direccion",
            "cp",
            "ciudad",
            "telefonos",
            "poliza",
        ]

        # Asegura que existan (por si alguna viene null o no está)
        for c in PENDING_COLS:
            if c not in df_pending.columns:
                df_pending[c] = ""

        # Recorta a las columnas deseadas y en el orden deseado
        df_pending = df_pending[PENDING_COLS]

        # build_grid espera num_aviso en tu implementación actual (para evitar reviente)
        # lo añadimos como placeholder, pero lo ocultaremos dentro del grid
        df_pending["num_aviso"] = ""  # placeholder para el grid

        grid_pending = build_grid(
            df_pending,
            grid_key="pending_grid",
            height=300,
            src="pending",   # origen = main_pending
        )

        # --- Navegación al detalle desde tabla de pendientes ---
        selected_pending = grid_pending.get("selected_rows", [])

        # Normalizamos selected_pending a lista de diccionarios
        if isinstance(selected_pending, pd.DataFrame):
            selected_pending = selected_pending.to_dict("records")
        elif selected_pending is None:
            selected_pending = []

        if isinstance(selected_pending, list) and len(selected_pending) > 0:
            first_row_p = selected_pending[0]

            if isinstance(first_row_p, dict):
                sel_id_pending = first_row_p.get("id")
            else:
                sel_id_pending = getattr(first_row_p, "id", None)

            if sel_id_pending is not None and str(sel_id_pending).strip() != "":
                # Importante: src="pending" para que Detalle use load_one_pending
                go("Detalle", sel_id_pending, src="pending")
            
# =========================
# PÁGINA: Nuevo expediente
# =========================
if page == "Nuevo expediente":
    page_title("➕ Alta de expediente")

    # Picklists
    ASEGURADORAS_DB, RESPONSABLES_DB, _ = load_picklists()

    ASEGURADORAS = [""] + ASEGURADORAS_DB
    RESPONSABLES = [""] + RESPONSABLES_DB

    ESTADOS = ["Nuevo", "En curso", "Cerrado"]
    ESTADOS_SELECT = [""] + ESTADOS

    with st.form("alta_expediente"):
        
        expediente = st.text_input(
            "Expediente  *",
        )

        aseguradora = st.selectbox("Aseguradora *", ASEGURADORAS)


        col1, col2, col3 = st.columns(3)
        with col1:
            fecha_aviso = st.date_input("Fecha de aviso *")
        with col2:
            fecha_siniestro = st.date_input("Fecha de siniestro *")
        with col3:
            fecha_efecto = st.date_input("Fecha de efecto *")

        nombre = st.text_input("Nombre asegurado *")
        apellidos = st.text_input("Apellidos *")
        direccion = st.text_input("Dirección *")

        col4, col5 = st.columns(2)
        with col4:
            cp = st.text_input("Código postal *")
        with col5:
            ciudad = st.text_input("Ciudad *")

        telefonos = st.text_input("Teléfonos *")
        poliza = st.text_input("Póliza *")
        observaciones = st.text_area("Observaciones")

        submitted = st.form_submit_button("✅ Crear expediente")

    if submitted:
        # -------------------------
        # Validación de obligatorios
        # -------------------------
        missing_fields = []

        if not expediente:
            missing_fields.append("Expediente")
        if not aseguradora:
            missing_fields.append("Aseguradora")
        if not fecha_aviso:
            missing_fields.append("Fecha de aviso")
        if not fecha_siniestro:
            missing_fields.append("Fecha de siniestro")
        if not fecha_efecto:
            missing_fields.append("Fecha de efecto")
        if not nombre:
            missing_fields.append("Nombre asegurado")
        if not apellidos:
            missing_fields.append("Apellidos")
        if not direccion:
            missing_fields.append("Dirección")
        if not cp:
            missing_fields.append("Código postal")
        if not ciudad:
            missing_fields.append("Ciudad")
        if not telefonos:
            missing_fields.append("Teléfonos")
        if not poliza:
            missing_fields.append("Póliza")

        if missing_fields:
            st.error(
                "❌ Faltan campos obligatorios:\n\n- "
                + "\n- ".join(missing_fields)
            )
            st.stop()

        # -------------------------
        # Inserción en base de datos
        # -------------------------
        try:
            result = execute_returning_one("""
                INSERT INTO main (
                    expediente,
                    aseguradora,
                    fecha_aviso,
                    fecha_siniestro,
                    fecha_efecto,
                    nombre_asegurado,
                    apellidos,
                    direccion,
                    cp,
                    ciudad,
                    telefonos,
                    poliza,
                    creation_state
                )
                VALUES (
                    :expediente, :aseguradora, :fecha_aviso, :fecha_siniestro, :fecha_efecto,
                    :nombre, :apellidos, :direccion, :cp, :ciudad,
                    :telefonos, :poliza, :creation_state
                )
                RETURNING id, num_aviso;
            """, {
                "expediente": expediente,
                "aseguradora": aseguradora,
                "fecha_aviso": fecha_aviso,
                "fecha_siniestro": fecha_siniestro,
                "fecha_efecto": fecha_efecto,
                "nombre": nombre,
                "apellidos": apellidos,
                "direccion": direccion,
                "cp": cp,
                "ciudad": ciudad,
                "telefonos": telefonos,
                "poliza": poliza,
                "creation_state": "manual",
            })

            st.success(
                f"✅ Expediente creado correctamente.\n\n"
                f"**Nº Aviso generado:** {result['num_aviso']}"
            )
            st.cache_data.clear()

        except Exception as e:
            st.error("❌ Error creando el expediente")
            st.exception(e)

# =========================
# PÁGINA: Dashboards
# =========================
if page == "Dashboards":
    page_title("📊 Dashboards")

    import plotly.express as px

    try:
        df_dash = load_main()
    except Exception as e:
        st.error("❌ Error cargando datos para dashboards")
        st.exception(e)
        st.stop()

    if df_dash.empty:
        st.info("No hay expedientes registrados.")
        st.stop()

    # -----
    # Filtro "a fecha del 2026" -> expedientes con fecha_aviso en 2026
    # -----
    df_dash["fecha_aviso"] = pd.to_datetime(df_dash["fecha_aviso"], errors="coerce")
    df_2026 = df_dash[df_dash["fecha_aviso"].dt.year == 2026].copy()

    if df_2026.empty:
        st.info("No hay expedientes con fecha de aviso en 2026.")
        st.stop()

    # Normaliza estado
    df_2026["estado"] = df_2026["estado"].fillna("Sin estado").astype(str).str.strip()
    df_2026.loc[df_2026["estado"].eq(""), "estado"] = "Sin estado"

    # Agrupa y calcula porcentajes
    estado_counts = (
        df_2026["estado"]
        .value_counts(dropna=False)
        .reset_index()
    )
    estado_counts.columns = ["estado", "count"]
    total = int(estado_counts["count"].sum())
    estado_counts["pct"] = (estado_counts["count"] / total * 100).round(1)

    st.markdown("### Distribución de expedientes por estado (año 2026)")

    fig = px.pie(
        estado_counts,
        names="estado",
        values="count",
        hole=0.55,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(showlegend=True)

    st.plotly_chart(fig, use_container_width=True)

    # (Opcional) tabla de apoyo debajo
    st.dataframe(
        estado_counts.sort_values("count", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

# =========================
# PÁGINA: Personalización
# =========================
if page == "Personalización":
    page_title("⚙️ Personalización")

    try:
        aseguradoras, responsables, estados_db = load_picklists()
    except Exception as e:
        st.error("❌ Error cargando valores de personalización")
        st.exception(e)
        st.stop()

    # Estados: catálogo oficial (tu lista fija) vs lo que existe en BD
    ESTADOS_OFICIALES = ["Nuevo", "En curso", "Cerrado"]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 🏢 Aseguradoras (en BD)")
        if aseguradoras:
            st.dataframe(pd.DataFrame({"aseguradora": aseguradoras}), use_container_width=True, hide_index=True)
        else:
            st.info("No hay aseguradoras registradas aún.")

    with c2:
        st.markdown("### 👤 Responsables (en BD)")
        if responsables:
            st.dataframe(pd.DataFrame({"responsable": responsables}), use_container_width=True, hide_index=True)
        else:
            st.info("No hay responsables registrados aún.")

    with c3:
        st.markdown("### 🚦 Estados (catálogo)")
        st.dataframe(pd.DataFrame({"estado": ESTADOS_OFICIALES}), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🔎 Estados detectados en BD (para control de calidad)")
    if estados_db:
        st.dataframe(pd.DataFrame({"estado_en_bd": estados_db}), use_container_width=True, hide_index=True)
    else:
        st.info("No hay estados registrados en BD.")

# =========================
# PÁGINA: Detalle expediente
# =========================
if page == "Detalle":
    page_title("🔎 Detalle del expediente")

    exp_id = st.session_state.get("selected_id")
    if not exp_id:
        st.error("No se ha recibido el id del expediente.")
        st.stop()

    try:
        exp_id_int = int(exp_id)
    except ValueError:
        st.error("Id de expediente inválido.")
        st.stop()

    src = st.session_state.get("src", "main")

    # Cargar registro según origen
    try:
        if src == "pending":
            df_one = load_one_pending(exp_id_int)
        else:
            df_one = load_one(exp_id_int)
    except Exception as e:
        st.error("❌ Error cargando el expediente")
        st.exception(e)
        st.stop()

    if df_one.empty:
        st.warning("No existe ese expediente.")
        st.stop()

    row = df_one.iloc[0].to_dict()

    def _norm(s):
        return (str(s) if s is not None else "").strip().lower()

    # =========================
    # APROBACIÓN DE EXPEDIENTE AUTO
    # =========================
    def _norm(s):
        return (str(s) if s is not None else "").strip().lower()

    # Calcula coincidencias (mismo criterio que tus tabs)
    df_nombre = load_coincidences_name(row.get("nombre_asegurado"), row.get("apellidos"), exp_id_int)
    df_dir    = load_matches_by_field("direccion",  row.get("direccion"),  exp_id_int)
    df_tel    = load_matches_by_field("telefonos",  row.get("telefonos"),  exp_id_int)
    df_pol    = load_matches_by_field("poliza",     row.get("poliza"),     exp_id_int)

    matches = {"NOMBRE": df_nombre, "DIR": df_dir, "TEL": df_tel, "POLIZA": df_pol}

    pdf_bytes = build_expediente_pdf_with_matches(row, matches)
    filename = f"expediente_{row.get('num_aviso','') or row.get('id','')}.pdf"

    st.download_button(
        label="Imprimir aviso en PDF",
        data=pdf_bytes,
        file_name=filename,
        mime="application/pdf",
        use_container_width=True,
    )

    # Picklists
    ASEGURADORAS_DB, RESPONSABLES_DB, _ = load_picklists()
    ASEGURADORAS = [""] + ASEGURADORAS_DB
    RESPONSABLES = [""] + RESPONSABLES_DB
    ESTADOS = ["Nuevo", "En curso", "Cerrado"]
    ESTADOS_SELECT = [""] + ESTADOS

    # Helpers fechas (acepta None)
    def to_date(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        try:
            return pd.to_datetime(x, errors="coerce").date()
        except Exception:
            return None

    # ---------- FORMULARIO DE DETALLE ----------
    with st.form("detalle_expediente_form"):
        ctop1, ctop2, ctop3 = st.columns([1, 2, 2])
        with ctop1:
            st.text_input("ID", value=str(row.get("id", "")), disabled=True)
        with ctop2:
            st.text_input("Nº Aviso", value=str(row.get("num_aviso", "")), disabled=True)
        with ctop3:
            expediente = st.text_input("Expediente *", value=str(row.get("expediente", "") or ""))

        c1, c2, c3 = st.columns(3)
        with c1:
            aseguradora = st.selectbox(
                "Aseguradora *",
                ASEGURADORAS,
                index=max(0, ASEGURADORAS.index(row.get("aseguradora")) if row.get("aseguradora") in ASEGURADORAS else 0),
            )
        with c2:
            estado = st.selectbox(
                "Estado",
                ESTADOS_SELECT,
                index=max(0, ESTADOS_SELECT.index(row.get("estado")) if row.get("estado") in ESTADOS_SELECT else 0),
            )
        with c3:
            responsable = st.selectbox(
                "Responsable",
                [""] + RESPONSABLES_DB,
                index=max(0, ([""] + RESPONSABLES_DB).index(row.get("responsable")) if row.get("responsable") in ([""] + RESPONSABLES_DB) else 0),
            )

        d1, d2, d3 = st.columns(3)
        with d1:
            fecha_aviso = st.date_input("Fecha de aviso", value=to_date(row.get("fecha_aviso")) or date.today())
        with d2:
            fecha_siniestro = st.date_input("Fecha de siniestro", value=to_date(row.get("fecha_siniestro")) or date.today())
        with d3:
            fecha_efecto = st.date_input("Fecha de efecto", value=to_date(row.get("fecha_efecto")) or date.today())

        nombre_asegurado = st.text_input("Nombre asegurado", value=str(row.get("nombre_asegurado", "") or ""))
        apellidos = st.text_input("Apellidos", value=str(row.get("apellidos", "") or ""))
        direccion = st.text_input("Dirección", value=str(row.get("direccion", "") or ""))

        c4, c5 = st.columns(2)
        with c4:
            cp = st.text_input("Código postal", value=str(row.get("cp", "") or ""))
        with c5:
            ciudad = st.text_input("Ciudad", value=str(row.get("ciudad", "") or ""))

        telefonos = st.text_input("Teléfonos", value=str(row.get("telefonos", "") or ""))
        poliza = st.text_input("Póliza", value=str(row.get("poliza", "") or ""))
        observaciones = st.text_area("Observaciones", value=str(row.get("observaciones", "") or ""))

        # --------- BOTONES INFERIORES ---------
        # Orden deseado: Volver, Guardar, Aprobar
        col_b1, col_b2, col_b3 = st.columns([1, 1, 1])

        with col_b1:
            back = st.form_submit_button("⬅️ Volver a Expedientes", use_container_width=True)

        with col_b2:
            save = st.form_submit_button("💾 Guardar cambios", use_container_width=True)

        # El de aprobar solo aparece para pendientes auto_new
        approve = False
        with col_b3:
            if src == "pending" and _norm(row.get("creation_state")) == "auto_new":
                approve = st.form_submit_button("✅ Aprobar", use_container_width=True)

    # ---------- LÓGICA POST-FORMULARIO ----------

    # Payload común, lo usamos tanto en guardar como en aprobar
    payload = {
        "expediente": expediente.strip() if expediente else None,
        "aseguradora": aseguradora.strip() if aseguradora else None,
        "fecha_aviso": fecha_aviso,
        "fecha_siniestro": fecha_siniestro,
        "fecha_efecto": fecha_efecto,
        "nombre_asegurado": nombre_asegurado.strip() if nombre_asegurado else None,
        "apellidos": apellidos.strip() if apellidos else None,
        "direccion": direccion.strip() if direccion else None,
        "cp": cp.strip() if cp else None,
        "ciudad": ciudad.strip() if ciudad else None,
        "telefonos": telefonos.strip() if telefonos else None,
        "poliza": poliza.strip() if poliza else None,
        "observaciones": observaciones.strip() if observaciones else None,
        "estado": estado.strip() if estado else None,
        "responsable": responsable.strip() if responsable else None,
    }

    # Volver a Expedientes
    if back:
        go("Expedientes")

    # Guardar cambios (sin aprobar)
    if save and not approve:
        try:
            if src == "pending":
                update_pending(exp_id_int, payload)
            else:
                execute("""
                    UPDATE main
                    SET
                        expediente = :expediente,
                        aseguradora = :aseguradora,
                        fecha_aviso = :fecha_aviso,
                        fecha_siniestro = :fecha_siniestro,
                        fecha_efecto = :fecha_efecto,
                        nombre_asegurado = :nombre_asegurado,
                        apellidos = :apellidos,
                        direccion = :direccion,
                        cp = :cp,
                        ciudad = :ciudad,
                        telefonos = :telefonos,
                        poliza = :poliza,
                        observaciones = :observaciones,
                        estado = :estado,
                        responsable = :responsable
                    WHERE id = :id
                """, {**payload, "id": exp_id_int})

            st.cache_data.clear()
            st.success("✅ Cambios guardados")
            st.rerun()

        except Exception as e:
            st.error("❌ Error guardando cambios")
            st.exception(e)

    # Aprobar: actualizamos pending con el payload y luego movemos a main
    if approve:
        try:
            if src != "pending":
                st.error("La aprobación solo aplica a expedientes pendientes.")
            else:
                # 1) Guardar lo editado en pending
                update_pending(exp_id_int, payload)

                # 2) Releer el registro pendiente ya actualizado
                df_latest = load_one_pending(exp_id_int)
                if df_latest.empty:
                    st.error("❌ No existe ya el expediente en pendientes.")
                    st.stop()
                latest = df_latest.iloc[0].to_dict()

                # 3) Insertar en main
                result = execute_returning_one("""
                    INSERT INTO main (
                        expediente,
                        aseguradora,
                        fecha_aviso,
                        fecha_siniestro,
                        fecha_efecto,
                        nombre_asegurado,
                        apellidos,
                        direccion,
                        cp,
                        ciudad,
                        telefonos,
                        poliza,
                        observaciones,
                        estado,
                        responsable,
                        creation_state
                    )
                    VALUES (
                        :expediente, :aseguradora, :fecha_aviso, :fecha_siniestro, :fecha_efecto,
                        :nombre_asegurado, :apellidos, :direccion, :cp, :ciudad,
                        :telefonos, :poliza, :observaciones, :estado, :responsable, :creation_state
                    )
                    RETURNING id, num_aviso;
                """, {
                    "expediente": latest.get("expediente"),
                    "aseguradora": latest.get("aseguradora"),
                    "fecha_aviso": latest.get("fecha_aviso"),
                    "fecha_siniestro": latest.get("fecha_siniestro"),
                    "fecha_efecto": latest.get("fecha_efecto"),
                    "nombre_asegurado": latest.get("nombre_asegurado"),
                    "apellidos": latest.get("apellidos"),
                    "direccion": latest.get("direccion"),
                    "cp": latest.get("cp"),
                    "ciudad": latest.get("ciudad"),
                    "telefonos": latest.get("telefonos"),
                    "poliza": latest.get("poliza"),
                    "observaciones": latest.get("observaciones"),
                    "estado": latest.get("estado"),
                    "responsable": latest.get("responsable"),
                    "creation_state": "auto_approved",
                })

                # 4) Eliminar de pending
                execute("DELETE FROM main_pending WHERE id = :id", {"id": exp_id_int})

                st.cache_data.clear()
                st.success(f"✅ Aprobado y movido a main. Nº Aviso: {result['num_aviso']}")

                # 5) Ir al detalle en MAIN
                st.session_state.src = "main"
                go("Detalle", result["id"], src="main")

        except Exception as e:
            st.error("❌ Error aprobando el expediente (pending → main)")
            st.exception(e)
            
    # ==========================================
    # SECCIÓN DE COINCIDENCIAS / DUPLICADOS
    # ==========================================
    st.markdown("---")
    st.subheader("🔍 Buscador de Coincidencias y Duplicados")

    search_configs = [
        {"label": "👤 Nombre y Apellidos", "field": "nombre_apellidos"},
        {"label": "🏠 Dirección", "field": "direccion"},
        {"label": "📞 Teléfono", "field": "telefonos"},
        {"label": "📄 Póliza", "field": "poliza"},
    ]

    tabs = st.tabs([conf["label"] for conf in search_configs])

    for i, conf in enumerate(search_configs):
        with tabs[i]:
            # 1) Cargar coincidencias según criterio
            if conf["field"] == "nombre_apellidos":
                df_match = load_coincidences_name(
                    row.get("nombre_asegurado"),
                    row.get("apellidos"),
                    exp_id_int
                )
            else:
                df_match = load_matches_by_field(
                    conf["field"],
                    row.get(conf["field"]),
                    exp_id_int
                )

            if df_match.empty:
                st.info(f"No hay otros expedientes con este {conf['label'].lower()}.")
                continue

            # 2) Mensaje de resumen
            if conf["field"] == "nombre_apellidos":
                ref = f"{(row.get('nombre_asegurado') or '').strip()} {(row.get('apellidos') or '').strip()}".strip()
            else:
                ref = str(row.get(conf["field"]) or "").strip()

            st.success(coincidencias_msg(len(df_match), ref))

            # 3) Preparar DataFrame para build_grid
            df_match_display = df_match.copy()
            
            # Asegurar que num_aviso existe
            if "num_aviso" not in df_match_display.columns:
                df_match_display["num_aviso"] = ""

            # 4) Usar build_grid (mismo formato que tabla principal)
            grid_m = build_grid(
                df_match_display,
                grid_key=f"grid_match_{conf['field']}_{exp_id_int}",
                height=200,
                src="main",
            )

            # 5) Navegación al detalle al seleccionar una fila
            selected_match = grid_m.get("selected_rows", [])

            if isinstance(selected_match, pd.DataFrame):
                selected_match = selected_match.to_dict("records")
            elif selected_match is None:
                selected_match = []

            if isinstance(selected_match, list) and len(selected_match) > 0:
                first_row = selected_match[0]
                if isinstance(first_row, dict):
                    sel_id = first_row.get("id")
                else:
                    sel_id = getattr(first_row, "id", None)

                if sel_id is not None and str(sel_id).strip() != "":
                    go("Detalle", sel_id, src="main")
