import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import time
import matplotlib.pyplot as plt
import requests
import altair as alt

SENSOR_IDS = ["Sensor_001", "Sensor_002", "Sensor_003", "Sensor_004"] 

def send_discord_alert(sensor_id, sensor_value, anomaly_type, action_suggestion_text):
    DISCORD_WEBHOOK_URL = st.secrets.get("DISCORD_WEBHOOK_URL") 

    if not DISCORD_WEBHOOK_URL:
        st.warning("🚨 ADVERTENCIA: La URL del Webhook de Discord no está configurada en los Streamlit Secrets.")
        return

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
                    {"name": "Hora de Detección", "value": time.strftime('%Y-%m-%d %H:%M:%S'), "inline": False}
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

# Configuración de Matplotlib (aunque Altair es el principal, esto asegura consistencia si se usa plt)
plt.rcParams['text.color'] = '#E0E0E0' # Texto general
plt.rcParams['axes.labelcolor'] = '#E0E0E0' # Etiquetas de ejes
plt.rcParams['xtick.color'] = '#E0E0E0' # Ticks del eje X
plt.rcParams['ytick.color'] = '#E0E0E0' # Ticks del eje Y
plt.rcParams['axes.facecolor'] = '#1E2A38' # Fondo de los ejes del gráfico
plt.rcParams['figure.facecolor'] = '#1E2A38' # Fondo de la figura del gráfico
plt.rcParams['grid.color'] = '#404040' # Color de la cuadrícula
plt.rcParams['legend.facecolor'] = '#2C3E50' # Fondo de la leyenda

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

st.markdown("""
<style>
/* Fondo general y texto principal */
.stApp {
    background-color: #1E2A38; /* Azul oscuro profundo */
    color: #E0E0E0; /* Texto principal gris claro */
}
/* Títulos y subtítulos */
h1, h2, h3, h4, h5, h6 {
    color: #FFD700; /* Dorado brillante para títulos */
}

/* Estilo para los KPI boxes */
.stMetric > div {
    background-color: #2C3E50; /* Azul medianoche para el fondo de las tarjetas */
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.4); /* Sombra más pronunciada */
    color: #E0E0E0; /* Texto valor KPI */
}
.stMetric label {
    color: #BDC3C7; /* Plata para etiquetas de KPI */
    font-size: 1.1em;
}
.stMetric div[data-testid="stMetricValue"] {
    font-size: 2em;
    font-weight: bold;
    color: #00FFFF; /* Cian vibrante para los valores */
}

/* Estilo para la tabla de historial */
.dataframe {
    background-color: #2C3E50; /* Azul medianoche para el fondo de tabla */
    color: #E0E0E0; /* Texto de tabla gris claro */
    border-radius: 8px;
    padding: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.4);
    font-size: 0.9em;
}
.dataframe th {
    background-color: #34495E; /* Azul más claro para encabezados de tabla */
    color: #F0F0F0; /* Texto de encabezado de tabla */
    padding: 8px;
}
.dataframe td {
    padding: 8px;
}
/* Estilo para el resaltado de anomalías en la tabla */
.stDataFrame tbody tr td:nth-child(4) div[data-value*="ANOMALÍA"] { 
    background-color: #E74C3C !important; /* Rojo Alizarin más fuerte */
    color: white !important;
    font-weight: bold;
}
.stDataFrame tbody tr td div[data-value*="ANOMALÍA"] { 
    background-color: #E74C3C !important; 
    color: white !important;
}

/* Estilo para los mensajes de alerta Streamlit (st.error, st.warning, st.info, st.success) */
div[data-testid="stAlert"] {
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.4);
}
div[data-testid="stAlert"] .st-bv { 
    background-color: #2C3E50; /* Fondo de alerta azul medianoche */
    color: #E0E0E0; /* Color de texto de alerta */
}
div[data-testid="stAlert"] .st-bv div[data-testid="stMarkdownContainer"] { 
    font-size: 1.1em;
}
div[data-testid="stAlert"] div[data-testid="stAlertContent"] {
    color: #E0E0E0; 
}
/* Colores específicos para tipos de alerta */
div[data-testid="stAlert"].st-emotion-cache-1f06x6a.e1f1d6z70.css-1f06x6a.e1f1d6z70 { /* st.success */
    background-color: #27AE60 !important; /* Verde Esmeralda */
}
div[data-testid="stAlert"].st-emotion-cache-1f06x6a.e1f1d6z70.css-1f06x6a.e1f1d6z70 { /* st.error */
    background-color: #C0392B !important; /* Rojo Ladrillo */
}
div[data-testid="stAlert"].st-emotion-cache-1f06x6a.e1f1d6z70.css-1f06x6a.e1f1d6z70 { /* st.warning */
    background-color: #F39C12 !important; /* Naranja Calabaza */
}
div[data-testid="stAlert"].st-emotion-cache-1f06x6a.e1f1d6z70.css-1f06x6a.e1f1d6z70 { /* st.info */
    background-color: #3498DB !important; /* Azul Peter River */
}


/* Ocultar la barra lateral por completo */
[data-testid="stSidebar"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

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

for i in range(1, 101):
    alerta_container.empty()
    action_suggestion_container.empty()

    anomalies_in_this_iteration = False

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

            mensaje_alerta = (f"🚨 **¡ALERTA!** Se ha detectado una **ANOMALÍA** "
                              f"({tipo_anomalia_display}) en el sensor **{sensor_id}**: **{nueva_lectura:.2f}°C**. "
                              f"¡Se recomienda revisar el sistema!")
            alerta_container.error(mensaje_alerta)
            action_suggestion_container.info(f"💡 **Sugerencia de Acción para {sensor_id}:** {sugerencia_accion_display}")
        
        nueva_fila_historial = pd.DataFrame([{
            'Hora': time.strftime('%H:%M:%S'),
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
        else:
            st.success("🟢 ESTADO ACTUAL: Normal")

    with grafico_container.container():
        st.subheader("Gráfico de Tendencia de Temperatura")
        num_lecturas_grafico = 30 * len(SENSOR_IDS) 
        df_para_grafico = st.session_state['historial_lecturas_df'].tail(num_lecturas_grafico).copy()
        
        df_para_grafico['Hora_dt'] = pd.to_datetime(df_para_grafico['Hora'], format='%H:%M:%S')
        df_para_grafico = df_para_grafico.sort_values(by='Hora_dt').reset_index(drop=True)

        line_chart = alt.Chart(df_para_grafico).mark_line().encode( 
            x=alt.X('Hora_dt', title='Tiempo', axis=alt.Axis(format='%H:%M:%S')), 
            y=alt.Y('valor_numerico', title='Temperatura (°C)'),
            color=alt.Color('Sensor ID', title='Sensor'),
            tooltip=[
                alt.Tooltip('Hora', title='Hora'), 
                alt.Tooltip('Sensor ID', title='Sensor'),
                alt.Tooltip('valor_numerico', title='Temp', format='.2f'),
                alt.Tooltip('Estado', title='Estado')
            ]
        ).properties(
            title=f'Últimas {num_lecturas_grafico // len(SENSOR_IDS)} Lecturas por Sensor'
        ).interactive()

        anomaly_points = alt.Chart(df_para_grafico[df_para_grafico['Estado'] == 'ANOMALÍA DETECTADA']).mark_point(
            color='#FF0000', filled=True, size=100, shape='cross'
        ).encode(
            x=alt.X('Hora_dt'), 
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
            return ['background-color: #E74C3C; color: white; font-weight: bold;' if 'ANOMALÍA' in str(v) else '' for v in s]

        st.dataframe(st.session_state['historial_lecturas_df'].tail(15 * len(SENSOR_IDS)).style.apply(highlight_anomalies, axis=1))

    time.sleep(st.session_state['simulation_speed']) 

st.success("✅ Simulación de monitoreo de sensores de temperatura finalizada.")
