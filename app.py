import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import time
import requests
import altair as alt
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Definir la zona horaria de la Ciudad de México
MEXICO_CITY_TZ = ZoneInfo("America/Mexico_City")

SENSOR_IDS = ["Sensor_001", "Sensor_002", "Sensor_003", "Sensor_004"] 

def send_discord_alert(sensor_id, sensor_value, anomaly_type, action_suggestion_text):
    DISCORD_WEBHOOK_URL = st.secrets.get("DISCORD_WEBHOOK_URL") 

    if not DISCORD_WEBHOOK_URL:
        st.warning("🚨 ADVERTENCIA: La URL del Webhook de Discord no está configurada en los Streamlit Secrets.")
        return

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

# --- Definición de Temas (Claro y Oscuro) ---
THEMES = {
    "dark": {
        "app_bg": "#0A192F",
        "text_color": "#CCD6F6",
        "title_color": "#64FFDA",
        "metric_bg": "#112240",
        "metric_label": "#8892B0",
        "metric_value": "#64FFDA",
        "dataframe_bg": "#112240",
        "dataframe_th_bg": "#233554",
        "dataframe_th_color": "#64FFDA",
        "anomaly_highlight": "#FF4D4D",
        "alert_success_bg": "#28A745",
        "alert_error_bg": "#DC3545",
        "alert_warning_bg": "#FFC107",
        "alert_info_bg": "#17A2B8",
        "border_color": "#233554",
        "button_bg": "#64FFDA",
        "button_color": "#0A192F",
        "slider_track": "#233554",
        "slider_progress_thumb": "#64FFDA",
        "slider_thumb_border": "#CCD6F6",
        "chart_title": "#64FFDA",
        "chart_axis_title": "#CCD6F6",
        "chart_axis_label": "#8892B0",
        "chart_line_colors": ['#64FFDA', '#FFD700', '#FF4D4D', '#00BFFF']
    },
    "light": {
        "app_bg": "#F8F9FA", # Very light gray, almost white
        "text_color": "#212529", # Dark gray for main text
        "title_color": "#007BFF", # Bright blue for titles
        "metric_bg": "#FFFFFF", # White cards
        "metric_label": "#495057", # Medium dark gray
        "metric_value": "#007BFF", # Bright blue
        "dataframe_bg": "#FFFFFF",
        "dataframe_th_bg": "#E9ECEF", # Light gray for table headers
        "dataframe_th_color": "#212529", # Dark text for table headers
        "anomaly_highlight": "#DC3545", # Still vivid red
        "alert_success_bg": "#28A745",
        "alert_error_bg": "#DC3545",
        "alert_warning_bg": "#FFC107",
        "alert_info_bg": "#17A2B8",
        "border_color": "#DEE2E6", # Light gray border
        "button_bg": "#007BFF", # Bright blue button
        "button_color": "#FFFFFF", # White text on button
        "slider_track": "#CED4DA", # Light gray
        "slider_progress_thumb": "#007BFF", # Bright blue
        "slider_thumb_border": "#FFFFFF", # White thumb border
        "chart_title": "#212529", # Dark text
        "chart_axis_title": "#495057", # Medium dark text
        "chart_axis_label": "#495057", # Medium dark text
        "chart_line_colors": ['#007BFF', '#FF8C00', '#20C997', '#6F42C1'] # Blue, Dark Orange, Green, Purple
    }
}

# Función para obtener el estilo CSS según el tema seleccionado
def get_css_style(theme_name):
    theme = THEMES[theme_name]
    return f"""
<style>
.stApp {{
    background-color: {theme["app_bg"]};
    color: {theme["text_color"]};
}}
h1, h2, h3, h4, h5, h6 {{
    color: {theme["title_color"]};
}}

.stMetric > div {{
    background-color: {theme["metric_bg"]};
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    color: {theme["text_color"]};
    border: 1px solid {theme["border_color"]};
}}
.stMetric label {{
    color: {theme["metric_label"]};
    font-size: 1.1em;
}}
.stMetric div[data-testid="stMetricValue"] {{
    font-size: 2.2em;
    font-weight: bold;
    color: {theme["metric_value"]};
}}

.dataframe {{
    background-color: {theme["dataframe_bg"]};
    color: {theme["text_color"]};
    border-radius: 8px;
    padding: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    font-size: 0.9em;
    border: 1px solid {theme["border_color"]};
}}
.dataframe th {{
    background-color: {theme["dataframe_th_bg"]};
    color: {theme["dataframe_th_color"]};
    padding: 8px;
}}
.dataframe td {{
    padding: 8px;
}}
.stDataFrame tbody tr td:nth-child(4) div[data-value*="ANOMALÍA"] {{ 
    background-color: {theme["anomaly_highlight"]} !important;
    color: white !important;
    font-weight: bold;
}}
.stDataFrame tbody tr td div[data-value*="ANOMALÍA"] {{ 
    background-color: {theme["anomaly_highlight"]} !important;
    color: white !important;
}}

div[data-testid="stAlert"] {{
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    border: 1px solid;
}}
div[data-testid="stAlert"] .st-bv div[data-testid="stMarkdownContainer"] {{ 
    font-size: 1.1em;
    color: {theme["text_color"]};
}}
div[data-testid="stAlert"] div[data-testid="stAlertContent"] {{
    color: {theme["text_color"]};
}}

div[data-testid="stAlert"].st-emotion-cache-1f06x6a.e1f1d6z70.css-1f06x6a.e1f1d6z70 {{
    background-color: {theme["alert_success_bg"]} !important;
    border-color: {theme["alert_success_bg"]} !important;
}}
div[data-testid="stAlert"].st-emotion-cache-1f06x6a.e1f1d6z70.css-1f06x6a.e1f1d6z70 {{
    background-color: {theme["alert_error_bg"]} !important;
    border-color: {theme["alert_error_bg"]} !important;
}}
div[data-testid="stAlert"].st-emotion-cache-1f06x6a.e1f1d6z70.css-1f06x6a.e1f1d6z70 {{
    background-color: {theme["alert_warning_bg"]} !important;
    border-color: {theme["alert_warning_bg"]} !important;
}}
div[data-testid="stAlert"].st-emotion-cache-1f06x6a.e1f1d6z70.css-1f06x6a.e1f1d6z70 {{
    background-color: {theme["alert_info_bg"]} !important;
    border-color: {theme["alert_info_bg"]} !important;
}}

[data-testid="stSidebar"] {{
    background-color: {theme["dataframe_bg"]}; /* Sidebar background matches card/table background */
    color: {theme["text_color"]};
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    padding-top: 20px;
    padding-left: 10px;
    padding-right: 10px;
}}
.stRadio > label > div > div {{
    color: {theme["text_color"]}; /* Color de texto para las opciones de radio */
}}
.stRadio > label > div > div > div {{
    color: {theme["text_color"]}; /* Color de texto para las opciones de radio */
}}


.stButton>button {{
    background-color: {theme["button_bg"]};
    color: {theme["button_color"]};
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    transition: all 0.2s ease-in-out;
}}
.stButton>button:hover {{
    background-color: {theme["button_bg"]}CC; /* Slightly transparent on hover */
    box-shadow: 0 6px 12px rgba(0,0,0,0.4);
    transform: translateY(-2px);
}}
.stButton>button:active {{
    transform: translateY(0);
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}}

.stSlider .st-bd .st-be {{
    background-color: {theme["slider_track"]};
}}
.stSlider .st-bd .st-be .st-bf {{
    background-color: {theme["slider_progress_thumb"]};
}}
.stSlider .st-bd .st-be .st-bf .st-bg {{
    background-color: {theme["slider_progress_thumb"]};
    border: 3px solid {theme["slider_thumb_border"]};
}}
.stSlider label {{
    color: {theme["metric_label"]};
}}
</style>
"""

# Inicializar el tema en session_state si no existe
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'dark' # Tema oscuro por defecto

# Aplicar el CSS basado en el tema actual
st.markdown(get_css_style(st.session_state['theme']), unsafe_allow_html=True)

# --- Contenido de la Aplicación ---
st.title("🌡️ Precisa Temp: Sistema de Predicción de Fallos en Sensores (Multi-Sensor)") 
st.markdown("---") 

st.subheader("Monitoreo de Temperatura en Tiempo Real")

kpi_container_anomalies = st.empty()
kpi_container_alerts = st.empty()

with kpi_container_anomalies.container():
    st.metric(label="Total Anomalías Detectadas", value=st.session_state['total_anomalies_detected'])
with kpi_container_alerts.container():
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

alerta_container = st.empty()
action_suggestion_container = st.empty()

grafico_container = st.empty()
historico_container = st.empty()
status_indicator_container = st.empty() 

st.write("Iniciando simulación de lecturas de múltiples sensores de temperatura...")

# --- Selector de Tema en la Barra Lateral ---
st.sidebar.title("Configuración de Tema")
selected_theme_option = st.sidebar.radio(
    "Selecciona el tema:",
    ('Oscuro', 'Claro'),
    index=0 if st.session_state['theme'] == 'dark' else 1 # Marca la opción actual
)

# Detectar cambio y actualizar el tema
if selected_theme_option == 'Oscuro' and st.session_state['theme'] != 'dark':
    st.session_state['theme'] = 'dark'
    st.experimental_rerun() # Fuerza la re-ejecución para aplicar el nuevo CSS
elif selected_theme_option == 'Claro' and st.session_state['theme'] != 'light':
    st.session_state['theme'] = 'light'
    st.experimental_rerun() # Fuerza la re-ejecución para aplicar el nuevo CSS

# Obtener el tema actual para usar sus colores en Altair
current_theme_colors = THEMES[st.session_state['theme']]


for i in range(1, 101):
    alerta_container.empty() 
    action_suggestion_container.empty()
    
    anomalies_in_this_iteration = False
    
    current_iteration_alert_message = ""
    current_iteration_suggestion_message = ""

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

            current_iteration_alert_message = (f"🚨 **¡ALERTA!** Se ha detectado una **ANOMALÍA** "
                                                f"({tipo_anomalia_display}) en el sensor **{sensor_id}**: **{nueva_lectura:.2f}°C**. "
                                                f"¡Se recomienda revisar el sistema!")
            current_iteration_suggestion_message = (f"💡 **Sugerencia de Acción para {sensor_id}:** {sugerencia_accion_display}")
        
        nueva_fila_historial = pd.DataFrame([{
            'Hora': formatted_time_for_display, 
            'Sensor ID': sensor_id,
            'Lectura (°C)': f"{nueva_lectura:.2f}",
            'Estado': estado_lectura,
            'Tipo de Anomalía': tipo_anomalia_display, 
            'valor_numerico': nueva_lectura
        }])
        st.session_state['historial_lecturas_df'] = pd.concat([st.session_state['historial_lecturas_df'], nueva_fila_historial], ignore_index=True)

    with kpi_container_anomalies.container():
        st.metric(label="Total Anomalías Detectadas", value=st.session_state['total_anomalies_detected'])
    with kpi_container_alerts.container():
        st.metric(label="Alertas Discord Enviadas", value=st.session_state['total_alerts_sent'])

    with status_indicator_container:
        if anomalies_in_this_iteration:
            st.error("🔴 ESTADO ACTUAL: ANOMALÍA(S) DETECTADA(S)")
            st.session_state['displayed_alert_message'] = current_iteration_alert_message
            st.session_state['displayed_suggestion_message'] = current_iteration_suggestion_message
        else:
            st.success("🟢 ESTADO ACTUAL: Normal")
            st.session_state['displayed_alert_message'] = "" 
            st.session_state['displayed_suggestion_message'] = ""
    
    if st.session_state['displayed_alert_message']:
        alerta_container.error(st.session_state['displayed_alert_message'])
    else:
        alerta_container.empty()

    if st.session_state['displayed_suggestion_message']:
        action_suggestion_container.info(st.session_state['displayed_suggestion_message'])
    else:
        action_suggestion_container.empty()


    with grafico_container.container():
        st.subheader("Gráfico de Tendencia de Temperatura")
        num_lecturas_grafico = 30 * len(SENSOR_IDS) 
        df_para_grafico = st.session_state['historial_lecturas_df'].tail(num_lecturas_grafico).copy()
        
        line_chart = alt.Chart(df_para_grafico).mark_line().encode( 
            x=alt.X('Hora', title='Tiempo', axis=alt.Axis(labelAngle=-45, titleColor=current_theme_colors['chart_axis_title'], labelColor=current_theme_colors['chart_axis_label'])),
            y=alt.Y('valor_numerico', title='Temperatura (°C)', axis=alt.Axis(titleColor=current_theme_colors['chart_axis_title'], labelColor=current_theme_colors['chart_axis_label'])), 
            color=alt.Color('Sensor ID', title='Sensor', scale=alt.Scale(range=current_theme_colors['chart_line_colors'])), 
            tooltip=[
                alt.Tooltip('Hora', title='Hora'), 
                alt.Tooltip('Sensor ID', title='Sensor'),
                alt.Tooltip('valor_numerico', title='Temp', format='.2f'),
                alt.Tooltip('Estado', title='Estado')
            ]
        ).properties(
            title=alt.Title(f'Últimas {num_lecturas_grafico // len(SENSOR_IDS)} Lecturas por Sensor', anchor='middle', color=current_theme_colors['chart_title']) 
        ).interactive()

        anomaly_points = alt.Chart(df_para_grafico[df_para_grafico['Estado'] == 'ANOMALÍA DETECTADA']).mark_point(
            color=current_theme_colors['anomaly_highlight'], filled=True, size=120, shape='cross' 
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
            return [f'background-color: {current_theme_colors["anomaly_highlight"]}; color: white; font-weight: bold;' if 'ANOMALÍA' in str(v) else '' for v in s]

        st.dataframe(st.session_state['historial_lecturas_df'].tail(15 * len(SENSOR_IDS)).style.apply(highlight_anomalies, axis=1))

    time.sleep(st.session_state['simulation_speed']) 

st.success("✅ Simulación de monitoreo de sensores de temperatura finalizada.")
