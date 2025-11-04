# ==========================================================
# 🤖 IANA DataCenter - VERSIÓN RÁPIDA (Basada en Autollantas)
# Autor: DataInsights Colombia
# ==========================================================

import streamlit as st
import pandas as pd
import re
import io
from typing import Optional
from sqlalchemy import text
from typing import Optional

# LangChain + Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
import numpy as np

# Micrófono en vivo (frontend) + fallback SR
# (Importaciones comentadas para la prueba)
# from streamlit_mic_recorder import speech_to_text, mic_recorder
# import speech_recognition as sr

# Agente de Correo
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import json

# ============================================
# 0) Configuración de la Página y Título
# ============================================
st.set_page_config(page_title="IANA DataCenter", page_icon="logo.png", layout="wide")

col1, col2 = st.columns([1, 4])
with col1:
    st.image("logo.png", width=120)
with col2:
    st.title("IANA: Tu Asistente IA para Análisis de Datos")
    st.markdown("Soy la red de agentes IA de **DataInsights**. Hazme una pregunta sobre los datos de tu negocio.")

# ============================================
# 1) Conexión a la Base de Datos y LLMs
# ============================================
@st.cache_resource
def get_database_connection():
    with st.spinner("🛰️ Conectando a la base de datos..."):
        try:
            creds = st.secrets["db_credentials"]
            uri = f"mysql+pymysql://{creds['user']}:{creds['password']}@{creds['host']}/{creds['database']}"
            
            engine_args = {
                "pool_recycle": 1800 
            }
            db = SQLDatabase.from_uri(uri, engine_args=engine_args)
            db.run("SELECT 1")
            st.success("✅ Conexión a la base de datos establecida.")
            return db
        except Exception as e:
            st.error(f"Error al conectar a la base de datos: {e}")
            st.error("REVISA TUS SECRETS: 'db_credentials' deben estar correctos Y tu firewall debe permitir IPs de Streamlit Cloud.")
            return None

@st.cache_resource
def get_llms():
    with st.spinner("🤝 Inicializando la red de agentes IANA..."):
        try:
            api_key = st.secrets["google_api_key"]
            common_config = dict(temperature=0.1, google_api_key=api_key)
            
            # --- USANDO GEMINI-PRO (EL ESTABLE) ---
            llm_sql = ChatGoogleGenerativeAI(model="gemini-pro", **common_config)
            llm_analista = ChatGoogleGenerativeAI(model="gemini-pro", **common_config)
            llm_orq = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.0, google_api_key=api_key)
            llm_validador = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.0, google_api_key=api_key)
            
            st.success("✅ Agentes de IANA listos.")
            return llm_sql, llm_analista, llm_orq, llm_validador
        except Exception as e:
            st.error(f"Error al inicializar los LLMs. Revisa tu API key de Google ('google_api_key'). Detalle: {e}")
            return None, None, None, None

db = get_database_connection()
llm_sql, llm_analista, llm_orq, llm_validador = get_llms()


@st.cache_resource
def get_sql_agent(_llm, _db):
    """ Este es el Agente LENTO (Plan B) """
    if not _llm or not _db: return None
    with st.spinner("🛠️ Configurando agente SQL de respaldo..."):
        toolkit = SQLDatabaseToolkit(db=_db, llm=_llm)
        agent = create_sql_agent(
            llm=_llm, 
            toolkit=toolkit, 
            verbose=False, 
            top_k=1000,
            handle_parsing_errors=True 
        )
        st.success("✅ Agente SQL de respaldo configurado.")
        return agent

# Este es el agente LENTO que solo se usa si el rápido falla
agente_sql_plan_b = get_sql_agent(llm_sql, db)

# ============================================
# 1.b) Reconocedor de Voz (fallback local)
# ============================================

# @st.cache_resource
# def get_recognizer():
#     r = sr.Recognizer()
#     r.energy_threshold = 300
#     r.dynamic_energy_threshold = True
#     return r

# def transcribir_audio_bytes(data_bytes: bytes, language: str) -> Optional[str]:
#     try:
#         r = get_recognizer()
#         with sr.AudioFile(io.BytesIO(data_bytes)) as source:
#             audio = r.record(source)
#         texto = r.recognize_google(audio, language=language)
#         return texto.strip() if texto else None
#     except Exception:
#         return None

# ============================================
# 2) Agente de Correo (Lógica Mejorada)
# ============================================
def extraer_detalles_correo(pregunta_usuario: str) -> dict:
    st.info("🧠 El agente de correo está interpretando tu solicitud...")
    contactos = dict(st.secrets.get("named_recipients", {}))
    default_recipient_name = st.secrets.get("email_credentials", {}).get("default_recipient", "")
    
    prompt = f"""
    Tu tarea es analizar la pregunta de un usuario y extraer los detalles para enviar un correo. Tu output DEBE SER un JSON válido.
    Agenda de Contactos Disponibles: {', '.join(contactos.keys())}
    Pregunta del usuario: "{pregunta_usuario}"
    Instrucciones para extraer:
    1.  `recipient_name`: Busca un nombre de la "Agenda de Contactos". Si no, usa "default".
    2.  `subject`: Crea un asunto corto.
    3.  `body`: Crea un cuerpo de texto breve.
    JSON Output para la pregunta actual:
    """
    try:
        response = llm_analista.invoke(prompt).content
        json_response = response.strip().replace("```json", "").replace("```", "").strip()
        details = json.loads(json_response)
        recipient_identifier = details.get("recipient_name", "default")
        
        if "@" in recipient_identifier:
            final_recipient = recipient_identifier
        elif recipient_identifier in contactos:
            final_recipient = contactos[recipient_identifier]
        else:
            final_recipient = default_recipient_name

        return {
            "recipient": final_recipient,
            "subject": details.get("subject", "Reporte de Datos - IANA"),
            "body": details.get("body", "Adjunto encontrarás los datos solicitados.")
        }
    except Exception as e:
        st.warning(f"No pude interpretar los detalles del correo (error: {e}), usaré los valores por defecto.")
        return {
            "recipient": default_recipient_name,
            "subject": "Reporte de Datos - IANA",
            "body": "Adjunto encontrarás los datos solicitados."
        }

def enviar_correo_agente(recipient: str, subject: str, body: str, df: Optional[pd.DataFrame] = None):
    with st.spinner(f"📧 Enviando correo a {recipient}..."):
        try:
            creds = st.secrets["email_credentials"]
            sender_email = creds["sender_email"]
            sender_password = creds["sender_password"]
            
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = recipient
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            if df is not None and not df.empty:
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                attachment = MIMEApplication(csv_buffer.getvalue(), _subtype='csv')
                attachment.add_header('Content-Disposition', 'attachment', filename="datos_iana.csv")
                msg.attach(attachment)
            
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(sender_email, sender_password)
                server.send_message(msg)
            
            st.success(f"✅ Correo enviado exitosamente a {recipient}!")
            return {"texto": f"¡Listo! El correo fue enviado a {recipient}."}
            
        except Exception as e:
            st.error(f"❌ No se pudo enviar el correo. Error: {e}")
            return {"tipo": "error", "texto": f"Lo siento, no pude enviar el correo. Detalle del error: {e}"}

# ============================================
# 3) Funciones Auxiliares y Agentes
# ============================================
def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.replace(r'[\u00A0\s]', '', regex=True).str.replace(',', '', regex=False).str.replace('$', '', regex=False).str.replace('%', '', regex=False)
    try: return pd.to_numeric(s2)
    except Exception: return s
def get_history_text(chat_history: list, n_turns=3) -> str:
    if not chat_history or len(chat_history) <= 1: return ""
    history_text = []
    relevant_history = chat_history[-(n_turns * 2 + 1) : -1]
    for msg in relevant_history:
        content = msg.get("content", {}); text_content = ""
        if isinstance(content, dict): text_content = content.get("texto", "")
        elif isinstance(content, str): text_content = content
        if text_content:
            role = "Usuario" if msg["role"] == "user" else "IANA"
            history_text.append(f"{role}: {text_content}")
    if not history_text: return ""
    return "\n--- Contexto de Conversación Anterior ---\n" + "\n".join(history_text) + "\n--- Fin del Contexto ---\n"

def get_last_sql_from_history(chat_history: list) -> Optional[str]:
    st.info("Buscando la última consulta SQL en el historial...")
    for msg in reversed(chat_history):
        if msg.get("role") == "assistant":
            content = msg.get("content", {})
            if isinstance(content, dict) and content.get("sql"):
                st.success("¡Consulta anterior encontrada!")
                return content["sql"]
    st.warning("No se encontró una consulta SQL previa en el historial.")
    return None
    
def markdown_table_to_df(texto: str) -> pd.DataFrame:
    lineas = [l.rstrip() for l in texto.splitlines() if l.strip().startswith('|')]
    if not lineas: return pd.DataFrame()
    lineas = [l for l in lineas if not re.match(r'^\|\s*-{2,}', l)]
    filas = [[c.strip() for c in l.strip('|').split('|')] for l in lineas]
    if len(filas) < 2: return pd.DataFrame()
    header, data = filas[0], filas[1:]
    max_cols = len(header); data = [r + ['']*(max_cols - len(r)) if len(r) < max_cols else r[:max_cols] for r in data]
    df = pd.DataFrame(data, columns=header)
    for c in df.columns: df[c] = _coerce_numeric_series(df[c])
    return df
def _df_preview(df: pd.DataFrame, n: int = 5) -> str:
    if df is None or df.empty: return ""
    try: return df.head(n).to_markdown(index=False)
    except Exception: return df.head(n).to_string(index=False)
def interpretar_resultado_sql(res: dict) -> dict:
    df = res.get("df")
    if df is not None and not df.empty and res.get("texto") is None:
        if df.shape == (1, 1):
            valor = df.iloc[0, 0]; nombre_columna = df.columns[0]
            res["texto"] = f"La respuesta para '{nombre_columna}' es: **{valor}**"
            st.info("💡 Resultado numérico interpretado para una respuesta directa.")
    return res
    
def _asegurar_select_only(sql: str) -> str:
    sql_clean = sql.strip().rstrip(';')
    if not re.match(r'(?is)^\s*select\b', sql_clean): raise ValueError("Solo se permite ejecutar consultas SELECT.")
    sql_clean = re.sub(r'(?is)\blimit\s+\d+\s*$', '', sql_clean).strip()
    return sql_clean

def style_dataframe(df: pd.DataFrame):
    value_cols = [
        c for c in df.select_dtypes("number").columns
        if not re.search(r"(?i)\b(mes|año|dia|fecha)\b", c)
    ]
    format_dict = {col: "{:,.0f}" for col in value_cols}
    def highlight_total(row):
        if str(row.iloc[0]).lower() == "total":
            return ["font-weight: bold; background-color: #f8f9fa; border-top: 2px solid #999;"] * len(row)
        else:
            return [""] * len(row)
    try:
        return df.style.apply(highlight_total, axis=1).format(format_dict)
    except Exception:
        return df

def limpiar_sql(sql_texto: str) -> str:
    if not sql_texto: return ""
    limpio = re.sub(r'```sql|```', '', sql_texto, flags=re.I)
    limpio = re.sub(r'(?im)^\s*sql[\s:]+', '', limpio)
    m = re.search(r'(?is)(select\b.+)$', limpio)
    if m: limpio = m.group(1)
    return limpio.strip().rstrip(';')

# ---
# ❗️ AGENTE SQL RÁPIDO (PLAN A) - Lógica de 'Autollantas'
# ---
def ejecutar_sql_real(pregunta_usuario: str, hist_text: str, last_sql: Optional[str] = None):
    st.info("🤖 El agente de datos (Plan A: Rápido) está traduciendo tu pregunta a SQL...")

    try:
        schema_info = db.get_table_info() 
    except Exception as e:
        st.error(f"Error crítico: No se pudo obtener el esquema de la base de datos. {e}")
        schema_info = "Error al obtener esquema. Asume columnas estándar."

    last_sql_context = ""
    if last_sql:
        last_sql_context = f"""
    --- CONSULTA ANTERIOR (Contexto) ---
    La última consulta SQL que ejecutaste fue:
    ```sql
    {last_sql}
    ```
    --- FIN CONSULTA ANTERIOR ---
    """

    prompt_con_instrucciones = f"""
    Tu tarea es generar una consulta SQL limpia (SOLO SELECT) para responder la pregunta del usuario, basándote ESTRICTAMENTE en el siguiente esquema de tabla.

    --- ESQUEMA DE LA BASE DE DATOS ---
    {schema_info}
    --- FIN DEL ESQUEMA ---

    {last_sql_context}

    --- REGLAS DE MODIFICACIÓN (¡MUY IMPORTANTE!) ---
    1. Si la "Pregunta del usuario" parece ser una continuación o modificación de la "CONSULTA ANTERIOR" (ej: "agregale el mes", "ahora por cliente"), DEBES modificar esa consulta anterior.
    2. Si la pregunta es nueva (ej: "¿cuál es el costo total?"), IGNORA la consulta anterior y crea una nueva desde cero.

    --- REGLAS DE NEGOCIO Y FORMATO ---
    1. Si piden "ventas 2025", asume YEAR(columna_de_fecha) = 2025.
    2. Si piden "margen", calcula el porcentaje: (SUM(Ventas) - SUM(Costos)) / SUM(Ventas).
    3. Siempre que sea posible, agrupa por la entidad solicitada (ej. por mes, por cliente).

    --- CONTEXTO Y PREGUNTA ---
    {hist_text}
    Pregunta del usuario: "{pregunta_usuario}"

    --- SALIDA ---
    Devuelve SOLO la consulta SQL (sin explicaciones, sin markdown ```sql```).
    """

    try:
        sql_query_bruta = llm_sql.invoke(prompt_con_instrucciones).content
        
        if not sql_query_bruta:
            st.error("El LLM no devolvió una consulta SQL válida.")
            return {"sql": None, "df": None, "error": "No se generó SQL."}

        st.text_area("🧩 SQL (Plan A) generado por el modelo:", sql_query_bruta, height=100)

        sql_query_limpia = limpiar_sql(sql_query_bruta)
        sql_query_limpia = _asegurar_select_only(sql_query_limpia)

        if not sql_query_limpia:
            st.error("El SQL generado quedó vacío después de la limpieza.")
            return {"sql": None, "df": None, "error": "SQL vacío tras limpieza."}

        st.code(sql_query_limpia, language="sql")

        with st.spinner("⏳ Ejecutando consulta..."):
            with db._engine.connect() as conn:
                df = pd.read_sql(text(sql_query_limpia), conn)

        st.success(f"✅ ¡Consulta ejecutada! Filas: {len(df)}")

        try:
            if not df.empty:
                value_cols = [
                    c for c in df.select_dtypes("number").columns
                    if not re.search(r"(?i)\b(mes|año|dia|fecha|id|codigo)\b", c)
                ]
                if value_cols and len(df) > 1:
                    total_row = {}
                    for col in df.columns:
                        if col in value_cols:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                total_row[col] = df[col].sum()
                            else: total_row[col] = np.nan
                        elif pd.api.types.is_numeric_dtype(df[col]):
                            total_row[col] = np.nan
                        else: total_row[col] = ""
                    first_col_name = df.columns[0]
                    total_row[first_col_name] = "Total"
                    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

                styled_df = style_dataframe(df)
                return {"sql": sql_query_limpia, "df": df, "styled": styled_df}
            else:
                return {"sql": sql_query_limpia, "df": df}

        except Exception as e:
            st.warning(f"⚠️ No se pudo aplicar formato ni totales: {e}")
            return {"sql": sql_query_limpia, "df": df}

    except Exception as e:
        st.warning(f"❌ Error en la consulta directa (Plan A). Detalle: {e}")
        return {"sql": None, "df": None, "error": str(e)}

# ---
# ❗️ AGENTE SQL LENTO (PLAN B)
# ---
def ejecutar_sql_en_lenguaje_natural(pregunta_usuario: str, hist_text: str):
    st.info("🤔 Activando el agente SQL experto (Plan B: Lento)...")

    simple_prompt = f"""
    {hist_text}
    Usando las herramientas de base de datos disponibles, responde la siguiente pregunta del usuario.
    Pregunta: "{pregunta_usuario}"
    """
    try:
        with st.spinner("💬 Pidiendo al agente SQL (Plan B) que responda..."):
            res = agente_sql_plan_b.invoke(simple_prompt)
            texto_salida = res["output"] if isinstance(res, dict) and "output" in res else str(res)
        
        st.info("📝 Intentando convertir la respuesta en una tabla de datos...");
        df_md = markdown_table_to_df(texto_salida)
        
        if df_md.empty: 
            st.warning("La conversión de Markdown a tabla no produjo filas. Se mostrará la salida cruda.")
            return {"texto": texto_salida, "df": df_md}

        resumen_texto = "Entendido. El agente de respaldo generó esta respuesta:"
        return {"texto": resumen_texto, "df": df_md}
        
    except Exception as e:
        st.error(f"❌ El agente SQL (Plan B) también encontró un problema: {e}")
        return {"texto": f"[SQL_ERROR] {e}", "df": pd.DataFrame()}

    
def analizar_con_datos(pregunta_usuario: str, hist_text: str, df: pd.DataFrame | None, feedback: str = None):
    st.info("\n🧠 El analista experto está examinando los datos...")
    correccion_prompt = ""
    if feedback:
        st.warning(f"⚠️ Reintentando con feedback: {feedback}")
        correccion_prompt = (f'INSTRUCCIÓN DE CORRECCIÓN: Tu respuesta anterior fue incorrecta. Feedback: "{feedback}". Genera una NUEVA respuesta corrigiendo este error.')
    preview = _df_preview(df, 500) or "(sin datos en vista previa; verifica la consulta)"
    prompt_analisis = f"""{correccion_prompt}\nEres IANA, un analista de datos senior EXTREMADAMENTE PRECISO y riguroso.\n---\n<<< REGLAS CRÍTICAS DE PRECISIÓN >>>\n1. **NO ALUCINAR**: NUNCA inventes números, totales, porcentajes o nombres de productos/categorías que no estén EXPRESAMENTE en la tabla de 'Datos'.\n2. **DATOS INCOMPLETOS**: Reporta los vacíos (p.ej., "sin datos para Marzo") sin inventar valores.\n3. **VERIFICAR CÁLCULOS**: Antes de escribir un número, revisa el cálculo (sumas/conteos/promedios) con los datos.\n4. **CITAR DATOS**: Basa CADA afirmación que hagas en los datos visibles en la tabla.\n---\nPregunta Original: {pregunta_usuario}\n{hist_text}\nDatos para tu análisis (usa SÓLO estos):\n{preview}\n---\nFORMATO OBLIGATORIO:\n📌 Análisis Ejecutivo de datos:\n1. Calcular totales y porcentajes clave.\n2. Detectar concentración.\n3. Identificar patrones temporales.\n4. Analizar dispersión.\nEntregar el resultado en 3 bloques:\n📌 Resumen Ejecutivo: hallazgos principales con números.\n🔍 Números de referencia: totales, promedios, ratios.\n⚠ Importante: Sé muy breve, directo y diciente."""
    with st.spinner("💡 Generando análisis avanzado..."):
        analisis = llm_analista.invoke(prompt_analisis).content
    st.success("💡 ¡Análisis completado!")
    return analisis
    
def responder_conversacion(pregunta_usuario: str, hist_text: str):
    st.info("💬 Activando modo de conversación...")
    prompt_personalidad = f"""Tu nombre es IANA, una IA amable. Ayuda a analizar datos.\nSi el usuario hace un comentario casual, responde amablemente de forma natural, muy humana y redirígelo a tus capacidades.\n{hist_text}\nPregunta: "{pregunta_usuario}" """
    respuesta = llm_analista.invoke(prompt_personalidad).content
    return {"texto": respuesta, "df": None, "analisis": None}

def generar_resumen_tabla(pregunta_usuario: str, res: dict) -> dict:
    st.info("✍️ Generando un resumen introductorio para la tabla...")
    df = res.get("df")
    if df is None or df.empty:
        return res
    prompt = f"""
    Actúa como IANA, un analista de datos amable y servicial.
    Tu tarea es escribir una breve y conversacional introducción para la tabla de datos que estás a punto de mostrar.
    Basa tu respuesta en la pregunta del usuario.
    
    Pregunta del usuario: "{pregunta_usuario}"
    
    Ejemplo 1:
    Pregunta: "cuáles son los proveedores"
    Respuesta: "¡Listo! Aquí tienes la lista de proveedores que encontré:"

    Ejemplo 2:
    Pregunta: "dame el total por mes"
    Respuesta: "Claro que sí. He preparado la tabla con los totales por mes:"
    
    Ahora, genera la introducción para la pregunta del usuario actual:
    """
    try:
        introduccion = llm_analista.invoke(prompt).content
        res["texto"] = introduccion
    except Exception as e:
        st.warning(f"No se pudo generar el resumen introductorio. Error: {e}")
        res["texto"] = "Aquí están los datos que solicitaste:"
    return res

# ============================================
# 4) Orquestador y Validación
# ============================================
def validar_y_corregir_respuesta_analista(pregunta_usuario: str, res_analisis: dict, hist_text: str) -> dict:
    MAX_INTENTOS = 2
    for intento in range(MAX_INTENTOS):
        st.info(f"🕵️‍♀️ Supervisor de Calidad: Verificando análisis (Intento {intento + 1})..."); contenido_respuesta = res_analisis.get("analisis", "") or ""
        if not contenido_respuesta.strip(): return {"tipo": "error", "texto": "El análisis generado estaba vacío."}
        df_preview = _df_preview(res_analisis.get("df"), 500) or "(sin vista previa de datos)"
        prompt_validacion = f"""Eres un supervisor de calidad estricto. Valida si el 'Análisis' se basa ESTRICTAMENTE en los 'Datos de Soporte'.\nFORMATO:\n- Si está 100% basado en los datos: APROBADO\n- Si alucina/inventa/no es relevante: RECHAZADO: [razón corta y accionable]\n---\nPregunta: "{pregunta_usuario}"\nDatos de Soporte:\n{df_preview}\n---\nAnálisis a evaluar:\n\"\"\"{contenido_respuesta}\"\"\"\n---\nEvaluación:"""
        try:
            resultado = llm_validador.invoke(prompt_validacion).content.strip(); up = resultado.upper()
            if up.startswith("APROBADO"):
                st.success("✅ Análisis aprobado por el Supervisor."); return res_analisis
            elif up.startswith("RECHAZADO"):
                feedback_previo = resultado.split(":", 1)[1].strip() if ":" in resultado else "Razón no especificada."
                st.warning(f"❌ Análisis rechazado. Feedback: {feedback_previo}")
                if intento < MAX_INTENTOS - 1:
                    st.info("🔄 Regenerando análisis con feedback...")
                    res_analisis["analisis"] = analizar_con_datos(pregunta_usuario, hist_text, res_analisis.get("df"), feedback=feedback_previo)
                else: return {"tipo": "error", "texto": "El análisis no fue satisfactorio incluso después de una corrección."}
            else: return {"tipo": "error", "texto": f"Respuesta ambigua del validador: {resultado}"}
        except Exception as e: return {"tipo": "error", "texto": f"Excepción durante la validación: {e}"}
    return {"tipo": "error", "texto": "Se alcanzó el límite de intentos de validación."}


def clasificar_intencion(pregunta: str) -> str:
    prompt_orq = f"""
Clasifica la intención del usuario en UNA SOLA PALABRA. Presta especial atención a los verbos de acción y palabras clave.

1. `analista`: Si la pregunta pide explícitamente una interpretación, resumen, comparación o explicación.
   PALABRAS CLAVE PRIORITARIAS: analiza, compara, resume, explica, por qué, tendencia, insights, dame un análisis, haz un resumen, interpreta.
   Si una de estas palabras clave está presente, la intención SIEMPRE es `analista`.

2. `consulta`: Si la pregunta pide datos crudos (listas, conteos, totales, valores, métricas) o resultados numéricos directos, y NO contiene palabras clave de `analista`.
   Ejemplos: 'cuántos proveedores hay', 'lista todos los productos', 'muéstrame el total', 'ventas por mes', 'margen por cliente', 'costo total', 'precio promedio'.
   PALABRAS CLAVE ADICIONALES: venta, ventas, costo, costos, margen, precio, unidades, rubro, cliente, artículo, producto, línea, familia, total, facturado, utilidad.

3. `correo`: Si la pregunta pide explícitamente enviar un correo, email o reporte.
   PALABRAS CLAVE: envía, mandar, correo, email, reporte a, envíale a.

4. `conversacional`: Si es un saludo o una pregunta general no relacionada con datos.
   Ejemplos: 'hola', 'gracias', 'qué puedes hacer', 'cómo estás'.

Pregunta: "{pregunta}"
Clasificación:
"""
    try:
        opciones = {"consulta", "analista", "conversacional", "correo"}
        r = llm_orq.invoke(prompt_orq).content.strip().lower().replace('"', '').replace("'", "")
        
        if any(pal in pregunta.lower() for pal in [
            "venta", "ventas", "margen", "costo", "costos", "precio", "unidades",
            "rubro", "cliente", "artículo", "producto", "línea", "familia", "total", "facturado"
        ]):
            return "consulta"
        
        if r not in opciones: return "consulta"
        return r
    except Exception:
        return "consulta"

def obtener_datos_sql(pregunta_usuario: str, hist_text: str, last_sql: Optional[str] = None) -> dict:
    
    res_real = ejecutar_sql_real(pregunta_usuario, hist_text, last_sql)
    
    if res_real.get("df") is not None and not res_real["df"].empty:
        return res_real
    elif res_real.get("df") is not None and res_real["df"].empty:
        res_real["texto"] = "La consulta se ejecutó, pero no se encontraron resultados."
        return res_real
        
    st.warning("La consulta directa (Plan A) falló. Intentando con el agente de lenguaje natural (Plan B)...")
    return ejecutar_sql_en_lenguaje_natural(pregunta_usuario, hist_text)

def guardian_agent(pregunta_usuario: str, sql_propuesta: str | None = None) -> bool:
    st.info("🧩 Guardian Agent: verificando seguridad de la solicitud...")
    palabras_peligrosas = [
        "drop", "delete", "truncate", "update", "insert", "alter",
        "create", "replace", "grant", "revoke", "commit", "rollback",
        "information_schema", "mysql", "sys", "pg_", "dual"
    ]
    if sql_propuesta:
        sql_lower = sql_propuesta.lower()
        if any(p in sql_lower for p in palabras_peligrosas):
            st.error("🚫 Guardian Agent: consulta SQL insegura detectada (palabras prohibidas).")
            return False

    prompt_guardian = f"""
Eres un agente de seguridad y cumplimiento. Tu tarea es revisar si la siguiente pregunta o consulta del usuario podría
implicar acceso a información sensible, manipulación de datos o riesgo de fuga de privacidad.

Pregunta del usuario: "{pregunta_usuario}"

Reglas:
1. Bloquea solo si pide datos personales (correos, teléfonos, NIT, direcciones, contraseñas, claves, API keys).
2. Bloquea si intenta modificar datos con verbos peligrosos (eliminar, borrar, actualizar, insertar, modificar, crear, drop, alter).
3. Bloquea si pide estructura interna del sistema o base de datos sensible.
4. PERMITE expresamente solicitudes de análisis financiero (márgenes, ventas, costos, totales, promedios, cantidades, precios).
5. PERMITE verbos analíticos comunes (ej. "agrega la columna mes", "agrupa por", "compara con") ya que se refieren a la *presentación* de los datos (un SELECT).

Responde solo con una palabra:
- "APROBADO" si es seguro.
- "BLOQUEADO" si no lo es.
    """
    try:
        decision = llm_validador.invoke(prompt_guardian).content.strip().upper()
        if "BLOQUEADO" in decision:
            st.error("🚫 Guardian Agent: solicitud bloqueada por seguridad.")
            return False
    except Exception as e:
        st.warning(f"Guardian Agent no pudo validar la solicitud ({e}), continuaré con precaución.")
        return True

    st.success("✅ Guardian Agent: solicitud aprobada.")
    return True

def orquestador(pregunta_usuario: str, chat_history: list):
    with st.expander("⚙️ Ver Proceso de IANA", expanded=True):
        hist_text = get_history_text(chat_history)
        last_sql = get_last_sql_from_history(chat_history)
        clasificacion = clasificar_intencion(pregunta_usuario)
        st.success(f"✅ ¡Intención detectada! Tarea: {clasificacion.upper()}.")

        if clasificacion == "conversacional":
            return responder_conversacion(pregunta_usuario, hist_text)
        
        if clasificacion == "correo":
            df_para_enviar = None
            for msg in reversed(st.session_state.get('messages', [])):
                if msg.get('role') == 'assistant':
                    content = msg.get('content', {}); df_prev = content.get('df')
                    if isinstance(df_prev, pd.DataFrame) and not df_prev.empty:
                        df_para_enviar = df_prev
                        st.info("📧 Datos de la tabla anterior encontrados para adjuntar al correo.")
                        break
            if df_para_enviar is None:
                st.warning("No encontré una tabla en la conversación reciente para enviar. El correo irá sin datos adjuntos.")
            detalles = extraer_detalles_correo(pregunta_usuario)
            return enviar_correo_agente(
                recipient=detalles["recipient"],
                subject=detalles["subject"],
                body=detalles["body"],
                df=df_para_enviar
            )
            
        if not guardian_agent(pregunta_usuario):
            return {"tipo": "error", "texto": "🚫 Solicitud bloqueada por el agente guardián por motivos de seguridad."}
        
        res_datos = {}
        df_previo = None
        use_previous_df = False

        for msg in reversed(st.session_state.get('messages', [])):
            if msg.get('role') == 'assistant':
                content = msg.get('content', {}); df_prev = content.get('df')
                if isinstance(df_prev, pd.DataFrame) and not df_prev.empty:
                    df_previo = df_prev
                    break
        
        if df_previo is not None:
            prompt_lower = pregunta_usuario.lower()
            prompt_clean = prompt_lower.strip().rstrip("?.!")
            contextual_keywords = ["anterior", "esos datos", "esa tabla", "la tabla"]
            simple_analysis_triggers = ["analiza", "analisis", "análisis", "haz un analisis", "dame un analisis"]

            if any(keyword in prompt_lower for keyword in contextual_keywords):
                use_previous_df = True
            elif clasificacion == "analista" and prompt_clean in simple_analysis_triggers:
                use_previous_df = True
        
        if use_previous_df:
            st.info("💡 Usando la tabla anterior para la nueva solicitud.")
            res_datos = {"df": df_previo, "sql": last_sql}
        else:
            res_datos = obtener_datos_sql(pregunta_usuario, hist_text, last_sql)
        

        if res_datos.get("df") is None: # Permitir DF vacíos
            return {"tipo": "error", "texto": "Lo siento, no pude obtener datos para tu pregunta. Intenta reformularla."}

        if clasificacion == "consulta":
            st.success("✅ Consulta directa completada.")
            res_interpretado = interpretar_resultado_sql(res_datos)
            
            if res_interpretado.get("texto") is None:
                res_interpretado = generar_resumen_tabla(pregunta_usuario, res_interpretado)
            
            return res_interpretado

        if clasificacion == "analista":
            st.info("🧠 Generando análisis inicial...")
            res_datos["analisis"] = analizar_con_datos(pregunta_usuario, hist_text, res_datos.get("df"))
            return validar_y_corregir_respuesta_analista(pregunta_usuario, res_datos, hist_text)

# ============================================
# 5) Interfaz: Micrófono en vivo + Chat
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": {"texto": "¡Hola! Soy IANA, tu asistente de IA. ¿Qué te gustaría saber?"}}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message.get("content", {});
        if isinstance(content, dict):
            if content.get("texto"): st.markdown(content["texto"])
            if content.get("styled") is not None: st.dataframe(content["styled"])
            elif isinstance(content.get("df"), pd.DataFrame) and not content["df"].empty:
                styled_df = style_dataframe(content["df"])
                st.dataframe(styled_df)
            if content.get("analisis"): st.markdown(content["analisis"])
        elif isinstance(content, str): 
            st.markdown(content)

# --- INICIO DE LA CORRECCIÓN (Simplificado) ---

st.markdown("### 🎤 Escribe tu pregunta")

def procesar_pregunta(prompt):
    if prompt:
        if not all([db, llm_sql, llm_analista, llm_orq, agente_sql_plan_b, llm_validador]):
            st.error("La aplicación no está completamente inicializada. Revisa los errores de conexión o de API key en los 'Secrets' de Streamlit.")
            return

        st.session_state.messages.append({"role": "user", "content": {"texto": prompt}})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            res = orquestador(prompt, st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": res})

        if res and res.get("tipo") != "error":
            if res.get("texto"): st.markdown(res["texto"])
            if res.get("styled") is not None:
                st.dataframe(res["styled"])
            elif isinstance(res.get("df"), pd.DataFrame) and not res["df"].empty:
                st.dataframe(res["df"])
            if res.get("analisis"):
                st.markdown("---"); st.markdown("### 🧠 Análisis de IANA"); st.markdown(res["analisis"])
                st.toast("Análisis generado ✅", icon="✅")
        elif res:
            st.error(res.get("texto", "Ocurrió un error inesperado."))
            st.toast("Hubo un error ❌", icon="❌")

# Reemplazamos el 'input_container' por un 'st.chat_input' simple
prompt_a_procesar = st.chat_input("... escribe tu pregunta aquí")

if prompt_a_procesar:
    procesar_pregunta(prompt_a_procesar)

# --- FIN DE LA CORRECCIÓN ---
