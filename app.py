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

plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white' 
plt.rcParams['axes.facecolor'] = '#1a1a1a'
plt.rcParams['figure.facecolor'] = '#1a1a1a'
plt.rcParams['grid.color'] = '#404040'
plt.rcParams['legend.facecolor'] = '#2a2a2a'

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
.stApp {
    background-color: #1a1a1a;
    color: #e0e0e0;
}
h1, h2, h3, h4, h5, h6 {
    color: #f0f0f0;
}
.stMetric > div {
    background-color: #333333;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    color: #f0f0f0;
}
.stMetric label {
    color: #a0a0a0;
    font-size: 1.1em;
}
.stMetric div[data-testid="stMetricValue"] {
    font-size: 2em;
    font-weight: bold;
    color: #00FFFF;
}
.dataframe {
    background-color: #333333;
    color: #e0e0e0;
    border-radius: 8px;
    padding: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    font-size: 0.9em;
}
.dataframe th {
    background-color: #444444;
    color: #f0f0f0;
    padding: 8px;
}
.dataframe td {
    padding: 8px;
}
.stDataFrame tbody tr td:nth-child(4) div[data-value*="ANOMALÍA"] {
    background-color: #FF6347 !important;
    color: white !important;
    font-weight: bold;
}
.stDataFrame tbody tr td div[data-value*="ANOMALÍA"] { 
    background-color: #FF6347 !important; 
    color: white !important;
}
div[data-testid="stAlert"] {
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
}
div[data-testid="stAlert"] .st-bv { 
    background-color: #333333; 
    color: #f0f0f0;
}
div[data-testid="stAlert"] .st-bv div[data-testid="stMarkdownContainer"] { 
    font-size: 1.1em;
}
div[data-testid="stAlert"] div[data-testid="stAlertContent"] {
    color: #f0f0f0; 
}
[data-testid="stSidebar"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🌡️ Precisa Temp: Sistema de Predicción de Fallos en Sensores (Multi-Sensor)") 
st.markdown("---") 

st.subheader("Monitoreo de Temperatura en Tiempo Real")

control_cols = st.columns(3) 

with control_cols[0]:
    kpi_cols = st.columns(2) 
    with kpi_cols[0]:
        st.metric(label="Total Anomalías Detectadas", value=st.session_state['total_anomalies_detected'])
    with kpi_cols[1]:
        st.metric(label="Alertas Discord Enviadas", value=st.session_state['total_alerts_sent'])

with control_cols[1]:
    st.markdown("##### Control de Simulación") 
    st.session_state['simulation_speed'] = st.slider(
        "Velocidad de Lectura (segundos por lectura)",
        min_value=0.1, max_value=2.0, value=0.5, step=0.1,
        help="Define el tiempo de espera entre cada lectura simulada."
    )

with control_cols[2]:
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

# Contenedores para mensajes de alerta y sugerencias de acción
alerta_container = st.empty()
action_suggestion_container = st.empty()
# Contenedores para el gráfico y el historial (reintroducidos)
grafico_container = st.empty()
historico_container = st.empty()


st.write("Iniciando simulación de lecturas de múltiples sensores de temperatura...")

for i in range(1, 101):
    alerta_container.empty()
    action_suggestion_container.empty()

    anomalies_in_this_iteration = False

    for idx, sensor_id in enumerate(SENSOR_IDS):
        nueva_lectura = 0.0
        tipo_anomalia = "N/A"
        sugerencia_accion = ""

        if (i + idx) % 10 != 0 and (i + idx) % 15 != 0 and (i + idx) % 25 != 0:
            nueva_lectura = 25 + 2 * np.random.randn(1)[0]
            nueva_lectura = np.clip(nueva_lectura, 20, 30)
        elif (i + idx) % 10 == 0:
            nueva_lectura = np.random.uniform(45, 55, 1)[0]
            tipo_anomalia = "Pico Alto"
            sugerencia_accion = "Revisar posibles sobrecargas, fallos en ventilación o componentes sobrecalentados." 
        elif (i + idx) % 15 == 0:
            nueva_lectura = np.random.uniform(5, 10, 1)[0]
            tipo_anomalia = "Caída Baja"
            sugerencia_accion = "Verificar si el sensor está desconectado, dañado o hay un problema en la fuente de energía." 
        elif (i + idx) % 25 == 0:
            nueva_lectura = 23.0 + np.random.uniform(-1, 1, 1)[0]
            tipo_anomalia = "Valor Constante"
            sugerencia_accion = "Inspeccionar el sensor por posibles fallas de congelación, cortocircuito o falta de comunicación." 

        model = st.session_state['sensor_models'][sensor_id]
        prediccion = model.predict(np.array(nueva_lectura).reshape(-1, 1))

        estado_lectura = "Normal"
        
        if prediccion == -1:
            st.session_state['total_anomalies_detected'] += 1 
            estado_lectura = "ANOMALÍA DETECTADA"
            anomalies_in_this_iteration = True

            mensaje_alerta = (f"🚨 **¡ALERTA!** Se ha detectado una **ANOMALÍA** "
                              f"({tipo_anomalia}) en el sensor **{sensor_id}**: **{nueva_lectura:.2f}°C**. "
                              f"¡Se recomienda revisar el sistema!")
            alerta_container.error(mensaje_alerta)
            action_suggestion_container.info(f"💡 **Sugerencia de Acción para {sensor_id}:** {sugerencia_accion}")

            current_time = time.time()
            if (current_time - st.session_state['last_alert_time'][sensor_id]) > COOLDOWN_SECONDS:
                send_discord_alert(sensor_id, nueva_lectura, tipo_anomalia, sugerencia_accion) 
                st.session_state['last_alert_time'][sensor_id] = current_time 
                st.session_state['total_alerts_sent'] += 1 
                st.info(f"✅ Alerta de Discord enviada para {sensor_id} (próxima alerta en {COOLDOWN_SECONDS}s).")
            else:
                tiempo_restante = int(COOLDOWN_SECONDS - (current_time - st.session_state['last_alert_time'][sensor_id]))
                st.warning(f"⚠️ Anomalía detectada en {sensor_id}, pero alerta omitida (cooldown activo). Próxima alerta en {tiempo_restante} segundos.")
        
        nueva_fila_historial = pd.DataFrame([{
            'Hora': time.strftime('%H:%M:%S'),
            'Sensor ID': sensor_id,
            'Lectura (°C)': f"{nueva_lectura:.2f}",
            'Estado': estado_lectura,
            'Tipo de Anomalía': tipo_anomalia,
            'valor_numerico': nueva_lectura
        }])
        st.session_state['historial_lecturas_df'] = pd.concat([st.session_state['historial_lecturas_df'], nueva_fila_historial], ignore_index=True)

    with control_cols[0]:
        kpi_cols = st.columns(2) 
        with kpi_cols[0]:
            st.metric(label="Total Anomalías Detectadas", value=st.session_state['total_anomalies_detected'])
        with kpi_cols[1]:
            st.metric(label="Alertas Discord Enviadas", value=st.session_state['total_alerts_sent'])

    # --- Gráfico de Tendencia de Temperatura (reintroducido) ---
    with grafico_container.container():
        st.subheader("Gráfico de Tendencia de Temperatura")
        num_lecturas_grafico = 50 * len(SENSOR_IDS) 
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

    # --- Historial de Lecturas Recientes (reintroducido) ---
    with historico_container.container():
        st.subheader("Historial de Lecturas Recientes")
        def highlight_anomalies(s):
            return ['background-color: #FF6347; color: white; font-weight: bold;' if 'ANOMALÍA' in str(v) else '' for v in s]

        st.dataframe(st.session_state['historial_lecturas_df'].tail(15 * len(SENSOR_IDS)).style.apply(highlight_anomalies, axis=1))

    time.sleep(st.session_state['simulation_speed']) 

st.success("✅ Simulación de monitoreo de sensores de temperatura finalizada.")
