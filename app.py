import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import time
import matplotlib.pyplot as plt
import requests

# --- Función de Envío a Discord ---
def send_discord_alert(sensor_value, anomaly_type):
    DISCORD_WEBHOOK_URL = "https://discordapp.com/api/webhooks/1389093241013407885/zVYNx5TuL8bO0Wyln4irru67BIhuTFa3H4zTIC5ln0iqph1sKaG5WlUhOIWZ-RxMM1Ot" 

    if DISCORD_WEBHOOK_URL == "TU_WEBHOOK_DE_DISCORD_AQUI": 
        st.warning("🚨 ADVERTENCIA: La URL del Webhook de Discord no está configurada. La alerta no se enviará a Discord.")
        return

    message_content = {
        "username": "Sistema de Monitoreo de Sensores",
        "avatar_url": "https://i.imgur.com/4S0t20e.png", 
        "embeds": [
            {
                "title": "🚨 ALERTA: Anomalía Detectada en Sensor de Temperatura",
                "description": f"Se ha detectado una **ANOMALÍA** en el sensor de temperatura.",
                "color": 15548997, 
                "fields": [
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
        st.success(f"Alerta de Discord enviada correctamente (Código: {response.status_code})")
    except requests.exceptions.RequestException as e:
        st.error(f"Error al enviar alerta a Discord: {e}")

# --- NEW: Inicialización de Session State para contadores y velocidad ---
if 'total_anomalies_detected' not in st.session_state:
    st.session_state['total_anomalies_detected'] = 0
if 'total_alerts_sent' not in st.session_state:
    st.session_state['total_alerts_sent'] = 0
if 'simulation_speed' not in st.session_state:
    st.session_state['simulation_speed'] = 0.5 # Velocidad por defecto

st.set_page_config(page_title="Monitor de Sensor de Temperatura", layout="wide") 

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

model = IsolationForest(contamination=0.03, random_state=42)
model.fit(data_for_model_training)

# --- Configuración del Cooldown para Alertas ---
if 'last_alert_time' not in st.session_state:
    st.session_state['last_alert_time'] = 0 
COOLDOWN_SECONDS = 60 

# --- NEW: Control de Velocidad de Simulación en la sidebar ---
st.sidebar.header("Control de Simulación")
st.session_state['simulation_speed'] = st.sidebar.slider(
    "Velocidad de Lectura (segundos por lectura)",
    min_value=0.1, max_value=2.0, value=0.5, step=0.1,
    help="Define el tiempo de espera entre cada lectura simulada."
)

# --- NEW: Indicador Visual de Estado del Monitoreo (Contenedor) ---
status_indicator_container = st.empty() 

st.title("🌡️ Sistema de Predicción de Fallos en Sensor de Temperatura")
st.markdown("---")

st.markdown("""
<style>
.stApp {
    background-color: #1a1a1a;
}
</style>
""", unsafe_allow_html=True)


st.subheader("Monitoreo de Temperatura en Tiempo Real")

# --- NEW: KPIs Display ---
kpi_container = st.container() 
with kpi_container:
    col1, col2 = st.columns(2) 
    with col1:
        st.metric(label="Total Anomalías Detectadas", value=st.session_state['total_anomalies_detected'])
    with col2:
        st.metric(label="Alertas Discord Enviadas", value=st.session_state['total_alerts_sent'])

lectura_actual_container = st.empty()
estado_lectura_container = st.empty()
alerta_container = st.empty()
grafico_container = st.empty()
historico_container = st.empty()

historial_columnas = ['Hora', 'Lectura (°C)', 'Estado', 'Tipo de Anomalía', 'valor_numerico']
historial_lecturas_df = pd.DataFrame(columns=historial_columnas)
historial_lecturas_df['valor_numerico'] = historial_lecturas_df['valor_numerico'].astype(float)

st.write("Iniciando simulación de lecturas del sensor de temperatura...")

for i in range(1, 51): 
    nueva_lectura = 0.0
    tipo_anomalia = "N/A"

    if i % 10 != 0 and i % 15 != 0 and i % 25 != 0:
        nueva_lectura = 25 + 2 * np.random.randn(1)[0]
        nueva_lectura = np.clip(nueva_lectura, 20, 30)
    elif i % 10 == 0:
        nueva_lectura = np.random.uniform(45, 55, 1)[0]
        tipo_anomalia = "Pico Alto"
    elif i % 15 == 0:
        nueva_lectura = np.random.uniform(5, 10, 1)[0]
        tipo_anomalia = "Caída Baja"
    elif i % 25 == 0:
        nueva_lectura = 23.0
        tipo_anomalia = "Valor Constante"

    prediccion = model.predict(np.array(nueva_lectura).reshape(1, -1))

    estado_lectura = "Normal"
    color_lectura = "green"
    mensaje_alerta = ""

    # --- Update total anomalies counter ---
    if prediccion == -1:
        st.session_state['total_anomalies_detected'] += 1 

        estado_lectura = "ANOMALÍA DETECTADA"
        color_lectura = "red"
        mensaje_alerta = (f"🚨 **¡ALERTA!** Se ha detectado una **ANOMALÍA** "
                          f"({tipo_anomalia}) en la lectura del sensor: **{nueva_lectura:.2f}°C**. "
                          f"¡Se recomienda revisar el sistema!")
        alerta_container.error(mensaje_alerta)
        
        current_time = time.time()
        if (current_time - st.session_state['last_alert_time']) > COOLDOWN_SECONDS:
            send_discord_alert(nueva_lectura, tipo_anomalia) 
            st.session_state['last_alert_time'] = current_time 
            st.session_state['total_alerts_sent'] += 1 # Increment alerts sent
            st.info(f"✅ Alerta de Discord enviada (próxima alerta en {COOLDOWN_SECONDS}s).")
        else:
            tiempo_restante = int(COOLDOWN_SECONDS - (current_time - st.session_state['last_alert_time']))
            st.warning(f"⚠️ Anomalía detectada, pero alerta omitida (cooldown activo). Próxima alerta en {tiempo_restante} segundos.")
    else:
        alerta_container.empty()

    # --- Update Visual Status Indicator ---
    with status_indicator_container:
        if estado_lectura == "ANOMALÍA DETECTADA":
            st.error("🔴 ESTADO ACTUAL: ANOMALÍA DETECTADA")
        else:
            st.success("🟢 ESTADO ACTUAL: Normal")

    nueva_fila_historial = pd.DataFrame([{
        'Hora': time.strftime('%H:%M:%S'),
        'Lectura (°C)': f"{nueva_lectura:.2f}",
        'Estado': estado_lectura,
        'Tipo de Anomalía': tipo_anomalia,
        'valor_numerico': nueva_lectura
    }])
    historial_lecturas_df = pd.concat([historial_lecturas_df, nueva_fila_historial], ignore_index=True)

    with lectura_actual_container.container():
        st.metric(label="Última Lectura de Temperatura", value=f"{nueva_lectura:.2f}°C", delta=None)

    with estado_lectura_container.container():
        st.markdown(f"<p style='color:{color_lectura}; font-size: 20px; font-weight: bold;'>Estado: {estado_lectura}</p>", unsafe_allow_html=True)

    with grafico_container.container():
        st.subheader("Gráfico de Tendencia de Temperatura")
        num_lecturas_grafico = 50 
        df_para_grafico = historial_lecturas_df.tail(num_lecturas_grafico)

        fig, ax = plt.subplots(figsize=(6, 2.5))
        
        ax.plot(df_para_grafico['Hora'], df_para_grafico['valor_numerico'], label='Temperatura', color='skyblue', linewidth=2)
        
        anomalias_grafico = df_para_grafico[df_para_grafico['Estado'] == 'ANOMALÍA DETECTADA']
        if not anomalias_grafico.empty:
            ax.scatter(anomalias_grafico['Hora'], anomalias_grafico['valor_numerico'], color='red', s=100, marker='X', linewidths=1, edgecolors='white', label='Anomalía')

        ax.set_xlabel('Hora', fontsize=10)
        ax.set_ylabel('Temperatura (°C)', fontsize=10)
        ax.set_title(f'Últimas {num_lecturas_grafico} Lecturas de Temperatura', fontsize=12)
        
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    with historico_container.container():
        st.subheader("Historial de Lecturas Recientes")
        st.dataframe(historial_lecturas_df.tail(15).style.applymap(lambda x: 'background-color: #ffe6e6' if 'ANOMALÍA' in str(x) else '', subset=['Estado']))

    # --- Use Simulation Speed from Slider ---
    time.sleep(st.session_state['simulation_speed']) 

st.success("✅ Simulación de monitoreo de sensor de temperatura finalizada.")