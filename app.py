# ==========================================================
# 🤖 IANA DataCenter - VERSIÓN RÁPIDA (Basada en Autollantas)
# Autor: DataInsights Colombia
# =PRODUCTIVO, USA EL DICCIONARIO DE DATOS EN EL PROMPT
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
# ❗️ NOTA: Usamos 'create_sql_agent' solo para el PLAN B (Lento)
from langchain_community.agent_toolkits.sql.base import create_sql_agent 
from langchain_community.utilities import SQLDatabase
import numpy as np

# Micrófono (Desactivado temporalmente para arreglar el bug de 'desaparición')
# from streamlit_mic_recorder import speech_to_text, mic_recorder
# import speech_recognition as sr

# Agente de Correo
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import json

# ============================================
# 0) NUEVO DICCIONARIO DE DATOS (¡LA CLAVE!)
# ============================================
# Todo el texto que me pasaste, metido en una variable
DATA_DICTIONARY = """
1. Visión General del Modelo
El modelo de datos es un Modelo Estrella con múltiples tablas de hechos.
- Tablas de Hechos (Facts): Vistas que contienen los números (ej. Valor_Facturado).
- Tablas de Dimensión (Dimensions): Vistas que contienen el contexto (ej. Nombre_Empresa).
La IA debe unir (JOIN) las tablas de Hechos con las tablas de Dimensión usando sus llaves (ej. ID_Empresa).

2. Diccionario de Vistas de Hechos (Las “Operativas”)

VIEW_Fact_Ingresos (V11)
- Propósito: Contiene cada línea individual de ítem facturado.
- Cuándo usarla (¡IMPORTANTE!): Usar SIEMPRE que el usuario pregunte por Ingresos, Ventas, Facturación o Valor de Venta.
- Grano: Una fila por ID_SOLICITUD + ID_Concepto.
- Columnas Clave:
    - ID_SOLICITUD: Llave para unir con VIEW_Fact_Solicitudes.
    - ID_Concepto: Llave para unir con VIEW_Dim_Concepto (Productos).
    - ID_Empresa: Llave para unir con VIEW_Dim_Empresa (Clientes).
    - Valor_Facturado: LA MÉTRICA PRINCIPAL DE INGRESOS.

VIEW_Fact_Costos (V5)
- Propósito: Contiene cada línea individual de costo de nómina.
- Cuándo usarla (¡IMPORTANTE!): Usar SIEMPRE que el usuario pregunte por Costos, Gastos, Nómina o Costos Operativos.
- Grano: Una fila por ID_SOLICITUD + ID_Usuario.
- Columnas Clave:
    - ID_SOLICITUD: Llave para unir con VIEW_Fact_Solicitudes.
    - ID_Usuario: Llave para unir con VIEW_Dim_Usuario (Empleados).
    - Rol_Empleado: Etiqueta (‘Líder’ o ‘Auxiliar’).
    - Costo_Total_Nomina: LA MÉTRICA PRINCIPAL DE COSTOS. (Incluye recargos, cesantías, etc.)

VIEW_Fact_Solicitudes (V9)
- Propósito: Tabla operativa principal con una fila por cada orden (solicitud).
- Cuándo usarla: Para medir rendimiento, tiempos y cumplimiento de metas.
- Grano: Una fila por ID_SOLICITUD.
- Columnas Clave:
    - ID_SOLICITUD: Llave primaria.
    - ID_Empresa: Llave para unir con VIEW_Dim_Empresa.
    - Tiempo_Ejecucion_Minutos / Horas: Tiempo real de trabajo.
    - Meta_Ejecucion_Min / Horas: Meta oficial para ejecución.
    - Cumple_Meta_Ejecucion: Flag (1 o 0). Usar AVG() para obtener % de cumplimiento.

3. Diccionario de Dimensiones “Inteligentes” (El “Contexto”)
(NO SUMAR MÉTRICAS EN ESTAS VISTAS. SOLO USAR PARA FILTRAR Y AGRUPAR).

VIEW_Dim_Empresa (V3)
- Propósito: Lista maestra de clientes.
- Columnas “Inteligentes”:
    - Nombre_Empresa
    - Segmento_Rentabilidad – Rentable, Medio, Regular.
    - Segmento_Valor – Top, Medio, Bajo según facturación.
    - Segmento_Frecuencia – Frecuente, Ocasional, Esporádico.

VIEW_Dim_Concepto (V6)
- Propósito: Lista maestra de productos o servicios.
- Columnas “Inteligientes”:
    - NOMBRE_CONCEPTO
    - Matriz_BCG – Estrella, Vaca Lechera, Dilema o Perro.
    - Perfil_Ciclo_Vida – Activo, En Riesgo, Obsoleto.

VIEW_Dim_Usuario (V7)
- Propósito: Lista maestra de empleados.
- Columnas “Inteligentes”:
    - Nombre_Usuario
    - Rol_Estandarizado – Rol limpio (Líder, Auxiliar, Otro).
    - Pct_Cumplimiento_Ejecucion
    - Segmento_Cumplimiento_Ejecucion – Cumple / No Cumple Meta.

VIEW_Dim_Ubicacion (V3)
- Propósito: Lista maestra de ubicaciones de servicio.
- Columnas “Inteligentes”:
    - Nombre_Ubicacion
    - Segmento_Actividad_Ubicacion – Muy Activa, Activa, Ocasional.
    - Segmento_Costo_Ubicacion – Alto, Medio o Bajo Costo.

4. Tablas de Reglas (La “Estrategia”)

dim_metas_tiempos
- Propósito: Tabla de reglas y metas estratégicas.
- Columnas Clave:
    - Nombre_Indicador: Tiempo de Espera, Ejecución o Total.
    - Meta_Minutos
    - Meta_Cumplimiento_Pct

5. ¡Instrucciones Especiales para la IA! (Lógicas Clave)

Cálculo del Margen Bruto (Rentabilidad)
- Fórmula: Margen = (SUM(Ingresos) - SUM(Costos)) / SUM(Ingresos)
- Pasos para la IA:
    1. Obtener SUM(Ingresos) desde VIEW_Fact_Ingresos[Valor_Facturado].
    2. Obtener SUM(Costos) desde VIEW_Fact_Costos[Costo_Total_Nomina].
    3. Aplicar la fórmula.
    4. Permitir agrupaciones por Cliente (VIEW_Dim_Empresa), Producto (VIEW_Dim_Concepto), Mes, etc.
"""

# ============================================
# 1) Configuración de la Página y Título
# ============================================
st.set_page_config(page_title="IANA DataCenter", page_icon="logo.png", layout="wide")

col1, col2 = st.columns([1, 4])
with col1:
    st.image("logo.png", width=120)
with col2:
    st.title("IANA: Tu Asistente IA para Análisis de Datos")
    st.markdown("Soy la red de agentes IA de **DataInsights**. Hazme una pregunta sobre los datos de tu negocio.")

# ============================================
# 2) Conexión a la Base de Datos y LLMs
# ============================================
@st.cache_resource
def get_database_connection():
    with st.spinner("🛰️ Conectando a la base de datos..."):
        try:
            creds = st.secrets["db_credentials"]
            uri = f"mysql+pymysql://{creds['user']}:{creds['password']}@{creds['host']}/{creds['database']}"
            
            engine_args = {"pool_recycle": 1800}
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
            
            # Usamos 'gemini-pro' que es el modelo estable
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
# 3) Agente de Correo (Sin cambios)
# ============================================
def extraer_detalles_correo(pregunta_usuario: str) -> dict:
    st.info("🧠 El agente de correo está interpretando tu solicitud...")
    contactos = dict(st.secrets.get("named_recipients", {}))
    default_recipient_name = st.secrets.get("email_credentials", {}).get("default_recipient", "")
    prompt = f"..." # (Omitido por brevedad, es el mismo de autollantas)
    try:
        response = llm_analista.invoke(prompt).content
        # ... (lógica de parseo de JSON omitida) ...
        details = json.loads(response.strip().replace("```json", "").replace("```", "").strip())
        recipient_identifier = details.get("recipient_name", "default")
        
        if "@" in recipient_identifier: final_recipient = recipient_identifier
        elif recipient_identifier in contactos: final_recipient = contactos[recipient_identifier]
        else: final_recipient = default_recipient_name

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
    # (El código de 'enviar_correo_agente' es idéntico, omitido por brevedad)
    pass # Tu código de smtplib va aquí

# ============================================
# 4) Funciones Auxiliares y Agentes
# ============================================
def get_history_text(chat_history: list, n_turns=3) -> str:
    # (El código de 'get_history_text' es idéntico, omitido por brevedad)
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
    # (El código de 'markdown_table_to_df' es idéntico, omitido por brevedad)
    lineas = [l.rstrip() for l in texto.splitlines() if l.strip().startswith('|')]
    if not lineas: return pd.DataFrame()
    lineas = [l for l in lineas if not re.match(r'^\|\s*-{2,}', l)]
    filas = [[c.strip() for c in l.strip('|').split('|')] for l in lineas]
    if len(filas) < 2: return pd.DataFrame()
    header, data = filas[0], filas[1:]
    max_cols = len(header); data = [r + ['']*(max_cols - len(r)) if len(r) < max_cols else r[:max_cols] for r in data]
    df = pd.DataFrame(data, columns=header)
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
    # (El código de 'style_dataframe' es idéntico, omitido por brevedad)
    try:
        value_cols = [c for c in df.select_dtypes("number").columns if not re.search(r"(?i)\b(mes|año|dia|fecha)\b", c)]
        format_dict = {col: "{:,.0f}" for col in value_cols}
        def highlight_total(row):
            if str(row.iloc[0]).lower() == "total":
                return ["font-weight: bold; background-color: #f8f9fa; border-top: 2px solid #999;"] * len(row)
            else:
                return [""] * len(row)
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
# ❗️ AGENTE SQL RÁPIDO (PLAN A) - ¡MODIFICADO CON TU DICCIONARIO!
# ---
def ejecutar_sql_real(pregunta_usuario: str, hist_text: str, last_sql: Optional[str] = None):
    st.info("🤖 El agente de datos (Plan A: Rápido) está traduciendo tu pregunta a SQL...")

    # NO LLAMAMOS A db.get_table_info()
    # Usamos el diccionario de datos que definimos al inicio.
    schema_info = DATA_DICTIONARY

    last_sql_context = ""
    if last_sql:
        last_sql_context = f"--- CONSULTA ANTERIOR (Contexto) ---\n```sql\n{last_sql}\n```\n"

    prompt_con_instrucciones = f"""
    Tu tarea es generar una consulta SQL limpia (SOLO SELECT) para responder la pregunta del usuario.
    DEBES SEGUIR ESTRICTAMENTE las reglas, esquemas y definiciones del siguiente "Manual de Datos":

    --- MANUAL DE DATOS (ESQUEMA Y REGLAS) ---
    {schema_info}
    --- FIN DEL MANUAL ---

    {last_sql_context}

    --- REGLAS DE MODIFICACIÓN ---
    1. Si la "Pregunta del usuario" parece ser una continuación o modificación de la "CONSULTA ANTERIOR" (ej: "agregale el mes", "ahora por cliente"), DEBES modificar esa consulta anterior.
    2. Si la pregunta es nueva (ej: "¿cuál es el costo total?"), IGNORA la consulta anterior y crea una nueva desde cero.

    --- CONTEXTO DE CHAT Y PREGUNTA ---
    {hist_text}
    Pregunta del usuario: "{pregunta_usuario}"

    --- SALIDA ---
    Devuelve SOLO la consulta SQL (sin explicaciones, sin markdown ```sql```).
    """

    try:
        # --- Llamada directa al LLM para generar SQL ---
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
                # (Lógica de añadir fila "Total" omitida por brevedad, es la misma)
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
    simple_prompt = f"{hist_text}\nPregunta: \"{pregunta_usuario}\""
    try:
        with st.spinner("💬 Pidiendo al agente SQL (Plan B) que responda..."):
            res = agente_sql_plan_b.invoke(simple_prompt) # Usa el agente LENTO
            texto_salida = res["output"] if isinstance(res, dict) and "output" in res else str(res)
        
        df_md = markdown_table_to_df(texto_salida)
        if df_md.empty: 
            return {"texto": texto_salida, "df": df_md}
        resumen_texto = "Entendido. El agente de respaldo generó esta respuesta:"
        return {"texto": resumen_texto, "df": df_md}
    except Exception as e:
        st.error(f"❌ El agente SQL (Plan B) también encontró un problema: {e}")
        return {"texto": f"[SQL_ERROR] {e}", "df": pd.DataFrame()}

    
def analizar_con_datos(pregunta_usuario: str, hist_text: str, df: pd.DataFrame | None, feedback: str = None):
    # (El código de 'analizar_con_datos' es idéntico, omitido por brevedad)
    st.info("\n🧠 El analista experto está examinando los datos...")
    preview = _df_preview(df, 500) or "(sin datos)"
    prompt_analisis = f"Eres IANA, un analista de datos senior...\nPregunta Original: {pregunta_usuario}\n{hist_text}\nDatos:\n{preview}\n---\nFORMATO OBLIGATORIO:..."
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
    # (El código de 'generar_resumen_tabla' es idéntico, omitido por brevedad)
    st.info("✍️ Generando un resumen introductorio para la tabla...")
    if res.get("df") is None or res["df"].empty: return res
    prompt = f"Actúa como IANA... Pregunta del usuario: \"{pregunta_usuario}\"\n... (ejemplos) ..."
    try:
        introduccion = llm_analista.invoke(prompt).content
        res["texto"] = introduccion
    except Exception as e:
        res["texto"] = "Aquí están los datos que solicitaste:"
    return res

# ============================================
# 5) Orquestador y Validación
# ============================================
def validar_y_corregir_respuesta_analista(pregunta_usuario: str, res_analisis: dict, hist_text: str):
    # (El código de 'validar_y_corregir' es idéntico, omitido por brevedad)
    pass # Tu lógica de validación va aquí

def clasificar_intencion(pregunta: str) -> str:
    # (El código de 'clasificar_intencion' es idéntico, omitido por brevedad)
    prompt_orq = f"Clasifica la intención del usuario en UNA SOLA PALABRA... (analista, consulta, correo, conversacional)...\nPregunta: \"{pregunta}\""
    try:
        opciones = {"consulta", "analista", "conversacional", "correo"}
        r = llm_orq.invoke(prompt_orq).content.strip().lower().replace('"', '').replace("'", "")
        if any(pal in pregunta.lower() for pal in ["venta", "ventas", "margen", "costo", "total", "facturado"]):
            return "consulta"
        if r not in opciones: return "consulta"
        return r
    except Exception:
        return "consulta"

def obtener_datos_sql(pregunta_usuario: str, hist_text: str, last_sql: Optional[str] = None) -> dict:
    
    # --- PLAN A: MÉTODO RÁPIDO (Con Diccionario de Datos) ---
    res_real = ejecutar_sql_real(pregunta_usuario, hist_text, last_sql)
    
    if res_real.get("df") is not None and not res_real["df"].empty:
        return res_real # ¡Éxito con el Plan A!
    elif res_real.get("df") is not None and res_real["df"].empty:
        res_real["texto"] = "La consulta se ejecutó, pero no se encontraron resultados."
        return res_real
        
    # --- PLAN B: MÉTODO LENTO (Agente) ---
    st.warning("La consulta directa (Plan A) falló. Intentando con el agente de lenguaje natural (Plan B)...")
    return ejecutar_sql_en_lenguaje_natural(pregunta_usuario, hist_text)

def guardian_agent(pregunta_usuario: str, sql_propuesta: str | None = None) -> bool:
    # (El código de 'guardian_agent' es idéntico, omitido por brevedad)
    st.info("🧩 Guardian Agent: verificando seguridad de la solicitud...")
    prompt_guardian = f"Eres un agente de seguridad... Pregunta: \"{pregunta_usuario}\"\n... (reglas) ...\nResponde APROBADO o BLOQUEADO."
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
            # (Lógica de correo omitida por brevedad)
            return {"texto": "Función de correo en desarrollo."}
            
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
            contextual_keywords = ["anterior", "esos datos", "esa tabla", "la tabla"]
            simple_analysis_triggers = ["analiza", "analisis", "análisis"]
            if any(keyword in prompt_lower for keyword in contextual_keywords) or \
               (clasificacion == "analista" and prompt_lower.strip() in simple_analysis_triggers):
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
            # return validar_y_corregir_respuesta_analista(pregunta_usuario, res_datos, hist_text)
            return res_datos # Omitimos la validación por ahora

# ============================================
# 6) Interfaz: Micrófono en vivo + Chat
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

# --- ENTRADA DE CHAT SIMPLIFICADA (SIN VOZ POR AHORA) ---

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

prompt_a_procesar = st.chat_input("... escribe tu pregunta aquí")

if prompt_a_procesar:
    procesar_pregunta(prompt_a_procesar)
