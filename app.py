import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import time
import requests
import altair as alt
from datetime import datetime, timezone
from zoneinfo import ZoneInfo # Requiere Python 3.9+

# Definir la zona horaria de la Ciudad de México
MEXICO_CITY_TZ = ZoneInfo("America/Mexico_City")

SENSOR_IDS = ["Sensor_001", "Sensor_002", "Sensor_003", "Sensor_004"] 

def send_discord_alert(sensor_id, sensor_value, anomaly_type, action_suggestion_text):
    DISCORD_WEBHOOK_URL = st.secrets.get("DISCORD_WEBHOOK_URL") 

    if not DISCORD_WEBHOOK_URL:
        st.warning("🚨 ADVERTENCIA: La URL del Webhook de Discord no está configurada en los Streamlit Secrets.")
        return

    # Obtener la hora actual en la zona horaria de la Ciudad de México para la alerta
    current_utc_time = datetime.now(timezone.utc)
    current_mexico_city_time = current_utc_time.astimezone(MEXICO_CITY_TZ)
    formatted_datetime_for_discord = current_mexico_city_time.strftime('%Y-%m-%d %H:%M:%S %Z%z')

    message_content = {
        "username": "Sistema de Monitoreo de Sensores",
        "avatar_url": "https://i.imgur.com/4S0t20e.png", 
        "embeds": [
            {
                "title": f"🚨 ALERTA: Anomalía Detectada en {sensor_id}",
                "description": f"Se ha detectado una **ANOMALÍA** en el sensor **{sensor_id}**.\n\n"
                               f"**Sugerencia de Acción:** {action_suggestion_text}",
                "color": 15548997, 
                "fields": [
                    {"name": "Sensor ID", "value": sensor_id, "inline": True},
                    {"name": "Tipo de Anomalía", "value": anomaly_type, "inline": True},
                    {"name": "Valor del Sensor", "value": f"{sensor_value:.2f}°C", "inline": True},
                    {"name": "Hora de Detección", "value": formatted_datetime_for_discord, "inline": False}
                ],
                "footer": {
                    "text": "Revisa el sistema de monitoreo en Streamlit Cloud"
                }
            }
        ]
    }

    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=message_content)
        response.raise_for_status() 
        st.success(f"Alerta de Discord enviada correctamente para {sensor_id} (Código: {response.status_code})")
    except Exception as e:
        st.error(f"Error al enviar alerta a Discord para {sensor_id}: {e}")

st.set_page_config(page_title="Precisa Temp Multi-Sensor", layout="wide") 

np.random.seed(42)

temperatura_normal_entrenamiento = 25 + 2 * np.random.randn(500)
temperatura_normal_entrenamiento = np.clip(temperatura_normal_entrenamiento, 20, 30)

temperatura_con_fallos_entrenamiento = np.copy(temperatura_normal_entrenamiento)
temperatura_con_fallos_entrenamiento[50:55] = np.random.uniform(45, 55, 5)
temperatura_con_fallos_entrenamiento[120:125] = np.random.uniform(5, 10, 5)
temperatura_con_fallos_entrenamiento[200:205] = 23.0
temperatura_con_fallos_entrenamiento[350:355] = np.random.uniform(40, 50, 5)

data_for_model_training = temperatura_con_fallos_entrenamiento.reshape(-1, 1)

if 'contamination_value' not in st.session_state:
    st.session_state['contamination_value'] = 0.03 

if 'sensor_models' not in st.session_state:
    st.session_state['sensor_models'] = {}
    for sensor_id in SENSOR_IDS:
        model = IsolationForest(contamination=st.session_state['contamination_value'], random_state=42)
        model.fit(data_for_model_training)
        st.session_state['sensor_models'][sensor_id] = model

if 'sensor_failure_state' not in st.session_state:
    st.session_state['sensor_failure_state'] = {sensor_id: {'is_failed': False, 'original_type': 'N/A', 'original_suggestion': ''} for sensor_id in SENSOR_IDS}


if 'total_anomalies_detected' not in st.session_state:
    st.session_state['total_anomalies_detected'] = 0
if 'total_alerts_sent' not in st.session_state:
    st.session_state['total_alerts_sent'] = 0

if 'simulation_speed' not in st.session_state:
    st.session_state['simulation_speed'] = 0.5 

if 'last_alert_time' not in st.session_state:
    st.session_state['last_alert_time'] = {sensor_id: 0 for sensor_id in SENSOR_IDS} 
COOLDOWN_SECONDS = 60 

historial_columnas = ['Hora', 'Sensor ID', 'Lectura (°C)', 'Estado', 'Tipo de Anomalía', 'valor_numerico']
if 'historial_lecturas_df' not in st.session_state:
    st.session_state['historial_lecturas_df'] = pd.DataFrame(columns=historial_columnas)
    st.session_state['historial_lecturas_df']['valor_numerico'] = st.session_state['historial_lecturas_df']['valor_numerico'].astype(float)

if 'displayed_alert_message' not in st.session_state:
    st.session_state['displayed_alert_message'] = ""
if 'displayed_suggestion_message' not in st.session_state:
    st.session_state['displayed_suggestion_message'] = ""
if 'active_alert_sensor_id' not in st.session_state:
    st.session_state['active_alert_sensor_id'] = None

# Función para reiniciar la simulación
def reset_simulation():
    st.session_state['total_anomalies_detected'] = 0
    st.session_state['total_alerts_sent'] = 0
    st.session_state['sensor_failure_state'] = {sensor_id: {'is_failed': False, 'original_type': 'N/A', 'original_suggestion': ''} for sensor_id in SENSOR_IDS}
    st.session_state['last_alert_time'] = {sensor_id: 0 for sensor_id in SENSOR_IDS}
    st.session_state['historial_lecturas_df'] = pd.DataFrame(columns=historial_columnas)
    st.session_state['historial_lecturas_df']['valor_numerico'] = st.session_state['historial_lecturas_df']['valor_numerico'].astype(float)
    st.session_state['displayed_alert_message'] = ""
    st.session_state['displayed_suggestion_message'] = ""
    st.session_state['active_alert_sensor_id'] = None
    # Re-entrenar modelos si es necesario (o solo si contamination_value cambió)
    for sensor_id in SENSOR_IDS:
        model = IsolationForest(contamination=st.session_state['contamination_value'], random_state=42)
        model.fit(data_for_model_training)
        st.session_state['sensor_models'][sensor_id] = model
    st.experimental_rerun() # Fuerza una re-ejecución completa para limpiar la UI

st.markdown("""
<style>
/* Inspiración: Estilo de terminal/dashboard moderno con tonos oscuros y acentos vibrantes */

/* Fondo general y texto principal */
.stApp {
    background-color: #0A192F; /* Azul oscuro profundo (similar a VS Code o terminal) */
    color: #CCD6F6; /* Gris azulado claro para texto principal */
}
/* Títulos y subtítulos */
h1, h2, h3, h4, h5, h6 {
    color: #64FFDA; /* Cian/Verde menta vibrante para títulos */
}

/* Estilo para los KPI boxes */
.stMetric > div {
    background-color: #112240; /* Azul oscuro ligeramente más claro para el fondo de las tarjetas */
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5); /* Sombra más pronunciada para un efecto flotante */
    color: #CCD6F6; /* Texto valor KPI */
    border: 1px solid #233554; /* Borde sutil */
}
.stMetric label {
    color: #8892B0; /* Gris azulado medio para etiquetas de KPI */
    font-size: 1.1em;
}
.stMetric div[data-testid="stMetricValue"] {
    font-size: 2.2em; /* Valor KPI más grande */
    font-weight: bold;
    color: #64FFDA; /* Cian vibrante para los valores */
}

/* Estilo para la tabla de historial */
.dataframe {
    background-color: #112240; /* Azul oscuro para el fondo de tabla */
    color: #CCD6F6; /* Texto de tabla gris azulado claro */
    border-radius: 8px;
    padding: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    font-size: 0.9em;
    border: 1px solid #233554;
}
.dataframe th {
    background-color: #233554; /* Azul más claro para encabezados de tabla */
    color: #64FFDA; /* Cian vibrante para encabezados */
    padding: 8px;
}
.dataframe td {
    padding: 8px;
}
/* Estilo para el resaltado de anomalías en la tabla */
.stDataFrame tbody tr td:nth-child(4) div[data-value*="ANOMALÍA"] { 
    background-color: #FF4D4D !important; /* Rojo vívido */
    color: white !important;
    font-weight: bold;
}
.stDataFrame tbody tr td div[data-value*="ANOMALÍA"] { 
    background-color: #FF4D4D !important; /* Rojo vívido */
    color: white !important;
}

/* Estilo para los mensajes de alerta Streamlit (st.error, st.warning, st.info, st.success) */
div[data-testid="stAlert"] {
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    border: 1px solid; /* Borde para que el color de alerta sea más prominente */
}
/* Ajuste para el texto dentro de las alertas */
div[data-testid="stAlert"] .st-bv div[data-testid="stMarkdownContainer"] { 
    font-size: 1.1em;
    color: #CCD6F6; /* Texto de alerta claro */
}
div[data-testid="stAlert"] div[data-testid="stAlertContent"] {
    color: #CCD6F6; /* Contenido de alerta claro */
}

/* Colores específicos para tipos de alerta */
div[data-testid="stAlert"].st-emotion-cache-1f06x6a.e1f1d6z70.css-1f06x6a.e1f1d6z70 { /* st.success */
    background-color: #28A745 !important; /* Verde más brillante */
    border-color: #218838 !important;
}
div[data-testid="stAlert"].st-emotion-cache-1f06x6a.e1f1d6z70.css-1f06x6a.e1f1d6z70 { /* st.error */
    background-color: #DC3545 !important; /* Rojo más brillante */
    border-color: #C82333 !important;
}
div[data-testid="stAlert"].st-emotion-cache-1f06x6a.e1f1d6z70.css-1f06x6a.e1f1d6z70 { /* st.warning */
    background-color: #FFC107 !important; /* Amarillo más brillante */
    border-color: #E0A800 !important;
}
div[data-testid="stAlert"].st-emotion-cache-1f06x6a.e1f1d6z70.css-1f06x6a.e1f1d6z70 { /* st.info */
    background-color: #17A2B8 !important; /* Cian más brillante */
    border-color: #138496 !important;
}

/* Ocultar la barra lateral por completo */
[data-testid="stSidebar"] {
    display: none !important;
}

/* Estilo para los botones */
.stButton>button {
    background-color: #64FFDA; /* Cian vibrante */
    color: #0A192F; /* Texto azul oscuro */
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    transition: all 0.2s ease-in-out;
}
.stButton>button:hover {
    background-color: #52E0C0; /* Cian más claro al pasar el ratón */
    box-shadow: 0 6px 12px rgba(0,0,0,0.4);
    transform: translateY(-2px);
}
.stButton>button:active {
    transform: translateY(0);
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

/* Estilo para los sliders */
.stSlider .st-bd .st-be { /* Track */
    background-color: #233554; /* Azul oscuro para el track */
}
.stSlider .st-bd .st-be .st-bf { /* Progress */
    background-color: #64FFDA; /* Cian vibrante para el progreso */
}
.stSlider .st-bd .st-be .st-bf .st-bg { /* Thumb */
    background-color: #64FFDA; /* Cian vibrante para el pulgar */
    border: 3px solid #CCD6F6; /* Borde blanco para el pulgar */
}
.stSlider label {
    color: #8892B0; /* Color de la etiqueta del slider */
}
</style>
""", unsafe_allow_html=True)

st.title("🌡️ Precisa Temp: Sistema de Predicción de Fallos en Sensores (Multi-Sensor)") 
st.markdown("---") 

st.subheader("Monitoreo de Temperatura en Tiempo Real")

kpi_cols_container = st.container() # Contenedor para los KPIs y botones de control
with kpi_cols_container:
    kpi_cols = st.columns(2) 
    with kpi_cols[0]:
        st.metric(label="Total Anomalías Detectadas", value=st.session_state['total_anomalies_detected'])
    with kpi_cols[1]:
        st.metric(label="Alertas Discord Enviadas", value=st.session_state['total_alerts_sent'])

control_cols = st.columns(2) 

with control_cols[0]:
    st.markdown("##### Control de Simulación") 
    st.session_state['simulation_speed'] = st.slider(
        "Velocidad de Lectura (segundos por lectura)",
        min_value=0.1, max_value=2.0, value=0.5, step=0.1,
        help="Define el tiempo de espera entre cada lectura simulada."
    )

with control_cols[1]:
    st.markdown("##### Control de Modelo IA")
    new_contamination_value = st.slider(
        "Sensibilidad Detección (Contamination)",
        min_value=0.01, max_value=0.10, value=st.session_state['contamination_value'], step=0.005,
        format="%.3f",
        help="Proporción esperada de anomalías. Mayor valor = más sensible."
    )
    if new_contamination_value != st.session_state['contamination_value']:
        st.session_state['contamination_value'] = new_contamination_value
        st.session_state['sensor_models'] = {}
        for sensor_id in SENSOR_IDS:
            model = IsolationForest(contamination=st.session_state['contamination_value'], random_state=42)
            model.fit(data_for_model_training)
            st.session_state['sensor_models'][sensor_id] = model
        st.info("Modelos de IA re-entrenados con nueva sensibilidad.")

st.markdown("---")

# Contenedores para alerta y sugerencia (definidos una sola vez)
alerta_container = st.empty()
action_suggestion_container = st.empty()

# Contenedor para los estados individuales de los sensores
sensor_status_container = st.empty()

grafico_container = st.empty()
historico_container = st.empty()
status_indicator_container = st.empty() 

st.write("Iniciando simulación de lecturas de múltiples sensores de temperatura...")

# Botones de acción globales (Reiniciar Simulación y Reconocer Alerta)
action_buttons_cols = st.columns(2)
with action_buttons_cols[0]:
    if st.button("🔄 Reiniciar Simulación"):
        reset_simulation()
with action_buttons_cols[1]:
    # El botón de reconocer alerta solo se muestra si hay una alerta activa
    if st.session_state['active_alert_sensor_id'] and st.button("✅ Reconocer Alerta Activa"):
        sensor_id_to_clear = st.session_state['active_alert_sensor_id']
        st.session_state['sensor_failure_state'][sensor_id_to_clear]['is_failed'] = False
        st.session_state['displayed_alert_message'] = ""
        st.session_state['displayed_suggestion_message'] = ""
        st.session_state['active_alert_sensor_id'] = None
        st.info(f"Alerta para {sensor_id_to_clear} reconocida. El sensor debería volver a la normalidad si no hay nuevas anomalías.")
        time.sleep(1) # Pequeña pausa para que el mensaje se vea
        st.experimental_rerun() # Fuerza una re-ejecución para actualizar la UI

for i in range(1, 101):
    
    anomalies_in_this_iteration = False
    
    current_iteration_alert_message = ""
    current_iteration_suggestion_message = ""
    current_iteration_alert_sensor_id = None # Para saber qué sensor activó la alerta en esta iteración

    current_utc_time = datetime.now(timezone.utc)
    current_mexico_city_time = current_utc_time.astimezone(MEXICO_CITY_TZ)
    formatted_time_for_display = current_mexico_city_time.strftime('%H:%M:%S')

    for idx, sensor_id in enumerate(SENSOR_IDS):
        nueva_lectura = 0.0
        tipo_anomalia_display = "N/A"
        sugerencia_accion_display = ""

        if st.session_state['sensor_failure_state'][sensor_id]['is_failed']:
            original_type = st.session_state['sensor_failure_state'][sensor_id]['original_type']
            original_suggestion = st.session_state['sensor_failure_state'][sensor_id]['original_suggestion']
            
            tipo_anomalia_display = original_type + " (Persistente)"
            sugerencia_accion_display = original_suggestion

            if original_type == "Pico Alto":
                nueva_lectura = np.random.uniform(45, 55, 1)[0]
            elif original_type == "Caída Baja":
                nueva_lectura = np.random.uniform(5, 10, 1)[0]
            elif original_type == "Valor Constante":
                nueva_lectura = 23.0 + np.random.uniform(-1, 1, 1)[0]
            else: 
                nueva_lectura = 25 + 2 * np.random.randn(1)[0]
                nueva_lectura = np.clip(nueva_lectura, 20, 30)
                
        else:
            if (i + idx) % 10 == 0:
                nueva_lectura = np.random.uniform(45, 55, 1)[0]
                tipo_anomalia_display = "Pico Alto"
                sugerencia_accion_display = "Revisar posibles sobrecargas, fallos en ventilación o componentes sobrecalentados."
                st.session_state['sensor_failure_state'][sensor_id] = {'is_failed': True, 'original_type': "Pico Alto", 'original_suggestion': sugerencia_accion_display}
            elif (i + idx) % 15 == 0:
                nueva_lectura = np.random.uniform(5, 10, 1)[0]
                tipo_anomalia_display = "Caída Baja"
                sugerencia_accion_display = "Verificar si el sensor está desconectado, dañado o hay un problema en la fuente de energía."
                st.session_state['sensor_failure_state'][sensor_id] = {'is_failed': True, 'original_type': "Caída Baja", 'original_suggestion': sugerencia_accion_display}
            elif (i + idx) % 25 == 0:
                nueva_lectura = 23.0 + np.random.uniform(-1, 1, 1)[0]
                tipo_anomalia_display = "Valor Constante"
                sugerencia_accion_display = "Inspeccionar el sensor por posibles fallas de congelación, cortocircuito o falta de comunicación."
                st.session_state['sensor_failure_state'][sensor_id] = {'is_failed': True, 'original_type': "Valor Constante", 'original_suggestion': sugerencia_accion_display}
            else:
                nueva_lectura = 25 + 2 * np.random.randn(1)[0]
                nueva_lectura = np.clip(nueva_lectura, 20, 30)
                tipo_anomalia_display = "N/A"
                sugerencia_accion_display = ""
                st.session_state['sensor_failure_state'][sensor_id] = {'is_failed': False, 'original_type': 'N/A', 'original_suggestion': ''}

        model = st.session_state['sensor_models'][sensor_id]
        prediccion = model.predict(np.array(nueva_lectura).reshape(-1, 1))

        estado_lectura = "Normal"
        
        if prediccion == -1 or st.session_state['sensor_failure_state'][sensor_id]['is_failed']:
            if prediccion == -1: 
                st.session_state['total_anomalies_detected'] += 1 
                current_time = time.time()
                if (current_time - st.session_state['last_alert_time'][sensor_id]) > COOLDOWN_SECONDS:
                    send_discord_alert(sensor_id, nueva_lectura, tipo_anomalia_display, sugerencia_accion_display) 
                    st.session_state['last_alert_time'][sensor_id] = current_time 
                    st.session_state['total_alerts_sent'] += 1 

            estado_lectura = "ANOMALÍA DETECTADA"
            anomalies_in_this_iteration = True 

            # Almacenar el mensaje de alerta y sugerencia para esta iteración
            current_iteration_alert_message = (f"🚨 **¡ALERTA!** Se ha detectado una **ANOMALÍA** "
                                                f"({tipo_anomalia_display}) en el sensor **{sensor_id}**: **{nueva_lectura:.2f}°C**. "
                                                f"¡Se recomienda revisar el sistema!")
            current_iteration_suggestion_message = (f"💡 **Sugerencia de Acción para {sensor_id}:** {sugerencia_accion_display}")
            current_iteration_alert_sensor_id = sensor_id # Guardar el ID del sensor que activó la alerta

        nueva_fila_historial = pd.DataFrame([{
            'Hora': formatted_time_for_display, 
            'Sensor ID': sensor_id,
            'Lectura (°C)': f"{nueva_lectura:.2f}",
            'Estado': estado_lectura,
            'Tipo de Anomalía': tipo_anomalia_display, 
            'valor_numerico': nueva_lectura
        }])
        st.session_state['historial_lecturas_df'] = pd.concat([st.session_state['historial_lecturas_df'], nueva_fila_historial], ignore_index=True)

    # Actualizar los KPIs
    with kpi_cols_container: # Usar el mismo contenedor para que se actualicen en su lugar
        kpi_cols = st.columns(2) 
        with kpi_cols[0]:
            st.metric(label="Total Anomalías Detectadas", value=st.session_state['total_anomalies_detected'])
        with kpi_cols[1]:
            st.metric(label="Alertas Discord Enviadas", value=st.session_state['total_alerts_sent'])

    # Actualizar el indicador de estado global
    with status_indicator_container:
        if anomalies_in_this_iteration:
            st.error("🔴 ESTADO ACTUAL: ANOMALÍA(S) DETECTADA(S)")
            st.session_state['displayed_alert_message'] = current_iteration_alert_message
            st.session_state['displayed_suggestion_message'] = current_iteration_suggestion_message
            st.session_state['active_alert_sensor_id'] = current_iteration_alert_sensor_id
        else:
            st.success("🟢 ESTADO ACTUAL: Normal")
            # Si no hay anomalías en esta iteración y no hay una alerta activa persistente, limpiar
            if not st.session_state['active_alert_sensor_id']: # Solo limpiar si no hay una alerta pendiente de reconocer
                st.session_state['displayed_alert_message'] = ""
                st.session_state['displayed_suggestion_message'] = ""
                st.session_state['active_alert_sensor_id'] = None # Asegurarse de que no haya ID de sensor activo
    
    # Mostrar los mensajes de alerta y sugerencia basados en el estado persistente
    if st.session_state['displayed_alert_message']:
        alerta_container.error(st.session_state['displayed_alert_message'])
    else:
        alerta_container.empty()

    if st.session_state['displayed_suggestion_message']:
        action_suggestion_container.info(st.session_state['displayed_suggestion_message'])
    else:
        action_suggestion_container.empty()

    # Mostrar el estado individual de los sensores
    with sensor_status_container.container():
        st.subheader("Estado Individual de Sensores")
        cols_status = st.columns(len(SENSOR_IDS))
        for idx, sensor_id in enumerate(SENSOR_IDS):
            with cols_status[idx]:
                is_failed = st.session_state['sensor_failure_state'][sensor_id]['is_failed']
                status_emoji = "🔴" if is_failed else "🟢"
                st.markdown(f"**{sensor_id}**: {status_emoji} {'Anómalo' if is_failed else 'Normal'}")


    with grafico_container.container():
        st.subheader("Gráfico de Tendencia de Temperatura")
        num_lecturas_grafico = 30 * len(SENSOR_IDS) 
        df_para_grafico = st.session_state['historial_lecturas_df'].tail(num_lecturas_grafico).copy()
        
        line_chart = alt.Chart(df_para_grafico).mark_line().encode( 
            x=alt.X('Hora', title='Tiempo', axis=alt.Axis(labelAngle=-45, titleColor='#CCD6F6', labelColor='#8892B0')),
            y=alt.Y('valor_numerico', title='Temperatura (°C)', axis=alt.Axis(titleColor='#CCD6F6', labelColor='#8892B0')), 
            color=alt.Color('Sensor ID', title='Sensor', scale=alt.Scale(range=['#64FFDA', '#FFD700', '#FF4D4D', '#00BFFF'])), 
            tooltip=[
                alt.Tooltip('Hora', title='Hora'), 
                alt.Tooltip('Sensor ID', title='Sensor'),
                alt.Tooltip('valor_numerico', title='Temp', format='.2f'),
                alt.Tooltip('Estado', title='Estado')
            ]
        ).properties(
            title=alt.Title(f'Últimas {num_lecturas_grafico // len(SENSOR_IDS)} Lecturas por Sensor', anchor='middle', color='#64FFDA') 
        ).interactive()

        anomaly_points = alt.Chart(df_para_grafico[df_para_grafico['Estado'] == 'ANOMALÍA DETECTADA']).mark_point(
            color='#FF4D4D', filled=True, size=120, shape='cross' 
        ).encode(
            x=alt.X('Hora'),
            y=alt.Y('valor_numerico'),
            tooltip=[
                alt.Tooltip('Hora', title='Hora'), 
                alt.Tooltip('Sensor ID', title='Sensor'),
                alt.Tooltip('valor_numerico', title='Temp', format='.2f'),
                alt.Tooltip('Estado', title='Estado'),
                alt.Tooltip('Tipo de Anomalía', title='Tipo Anomalía')
            ]
        )

        chart = alt.layer(line_chart, anomaly_points).resolve_scale(
            y='independent'
        )
        
        st.altair_chart(chart, use_container_width=True)

    with historico_container.container():
        st.subheader("Historial de Lecturas Recientes")
        def highlight_anomalies(s):
            return ['background-color: #FF4D4D; color: white; font-weight: bold;' if 'ANOMALÍA' in str(v) else '' for v in s]

        st.dataframe(st.session_state['historial_lecturas_df'].tail(15 * len(SENSOR_IDS)).style.apply(highlight_anomalies, axis=1))

    time.sleep(st.session_state['simulation_speed']) 

st.success("✅ Simulación de monitoreo de sensores de temperatura finalizada.")
