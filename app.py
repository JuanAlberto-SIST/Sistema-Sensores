import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import time
import matplotlib.pyplot as plt
import requests

def send_discord_alert(sensor_value, anomaly_type, action_suggestion_text):
    DISCORD_WEBHOOK_URL = st.secrets["DISCORD_WEBHOOK_URL"] 

    if not DISCORD_WEBHOOK_URL:
        st.warning("🚨 ADVERTENCIA: La URL del Webhook de Discord no está configurada en los Streamlit Secrets.")
        return

    message_content = {
        "username": "Sistema de Monitoreo de Sensores",
        "avatar_url": "https://i.imgur.com/4S0t20e.png", 
        "embeds": [
            {
                "title": "🚨 ALERTA: Anomalía Detectada en Sensor de Temperatura",
                "description": f"Se ha detectado una **ANOMALÍA** en el sensor de temperatura.\n\n" # Añade salto de línea
                               f"**Sugerencia de Acción:** {action_suggestion_text}", # Agrega la sugerencia aquí
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
    except Exception as e:
        st.error(f"Error al enviar alerta a Discord: {e}")

st.set_page_config(page_title="Precisa Temp", layout="wide") 

# --- Configuración de Matplotlib para Tema Oscuro Profesional ---
plt.rcParams['text.color'] = '#e0e0e0' # Texto casi blanco
plt.rcParams['axes.labelcolor'] = '#e0e0e0'
plt.rcParams['xtick.color'] = '#e0e0e0'
plt.rcParams['ytick.color'] = '#e0e0e0' 
plt.rcParams['axes.facecolor'] = '#2a2a2a' # Fondo más oscuro para el área del gráfico
plt.rcParams['figure.facecolor'] = '#2a2a2a' # Fondo de la figura
plt.rcParams['grid.color'] = '#4a4a4a' # Cuadrícula más oscura y sutil
plt.rcParams['legend.facecolor'] = '#2a2a2a' # Fondo de la leyenda

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

if 'total_anomalies_detected' not in st.session_state:
    st.session_state['total_anomalies_detected'] = 0
if 'total_alerts_sent' not in st.session_state:
    st.session_state['total_alerts_sent'] = 0
if 'simulation_speed' not in st.session_state:
    st.session_state['simulation_speed'] = 0.5 

if 'last_alert_time' not in st.session_state:
    st.session_state['last_alert_time'] = 0 
COOLDOWN_SECONDS = 60 

# --- AQUI SE ELIMINARON TODAS LAS SECCIONES DE INTRODUCCIÓN Y BARRA LATERAL PARA UN DISEÑO LIMPIO ---
# El contenido se organiza ahora en "st.tabs" si lo quieres reintroducir después

status_indicator_container = st.empty() 

# --- NUEVO CSS para el fondo general CLARO (más limpio y profesional) ---
st.markdown("""
<style>
.stApp {
    background-color: #222222; /* Fondo gris oscuro suave */
    color: #e0e0e0; /* Texto principal casi blanco */
}
h1, h2, h3, h4, h5, h6 {
    color: #f0f0f0; /* Títulos más claros */
}
.stMetric > div { /* Estilo para los KPIs */
    background-color: #333333;
    border-radius: 8px;
    padding: 10px;
    color: #f0f0f0;
}
.stMetric label {
    color: #a0a0a0; /* Etiquetas de KPI más suaves */
}
.stExpander { /* Estilo para los expanders si se usan */
    background-color: #333333;
    border-radius: 8px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)


st.title("🌡️ Precisa Temp: Sistema de Predicción de Fallos en Sensores")
st.markdown("---")

st.subheader("Monitoreo de Temperatura en Tiempo Real")

# --- NUEVA ESTRUCTURA DE LAYOUT Y KPIs ---
main_cols = st.columns([0.7, 0.3]) # Columna principal para monitoreo, columna pequeña para controles

with main_cols[0]: # Columna principal izquierda
    kpi_cols = st.columns(2) 
    with kpi_cols[0]:
        st.metric(label="Total Anomalías Detectadas", value=st.session_state['total_anomalies_detected'])
    with kpi_cols[1]:
        st.metric(label="Alertas Discord Enviadas", value=st.session_state['total_alerts_sent'])

    st.markdown("---") # Separador visual

    lectura_actual_container = st.empty()
    estado_lectura_container = st.empty()
    alerta_container = st.empty()
    action_suggestion_container = st.empty() # Mover contenedor de sugerencia aquí para que esté cerca de la alerta

    st.subheader("Gráfico de Tendencia de Temperatura")
    grafico_container = st.empty() # Definir gráfico contenedor aquí
    
    st.subheader("Historial de Lecturas Recientes")
    historico_container = st.empty() # Definir historial contenedor aquí

with main_cols[1]: # Columna derecha (para el slider de velocidad)
    st.sidebar.header("Control de Simulación")
    st.session_state['simulation_speed'] = st.sidebar.slider(
        "Velocidad de Lectura (segundos por lectura)",
        min_value=0.1, max_value=2.0, value=0.5, step=0.1,
        help="Define el tiempo de espera entre cada lectura simulada."
    )
    st.markdown("---") # Separador


historial_columnas = ['Hora', 'Lectura (°C)', 'Estado', 'Tipo de Anomalía', 'valor_numerico']
historial_lecturas_df = pd.DataFrame(columns=historial_columnas)
historial_lecturas_df['valor_numerico'] = historial_lecturas_df['valor_numerico'].astype(float)

st.write("Iniciando simulación de lecturas del sensor de temperatura...")

for i in range(1, 51): 
    nueva_lectura = 0.0
    tipo_anomalia = "N/A"
    sugerencia_accion = "" 

    if i % 10 != 0 and i % 15 != 0 and i % 25 != 0:
        nueva_lectura = 25 + 2 * np.random.randn(1)[0]
        nueva_lectura = np.clip(nueva_lectura, 20, 30)
    elif i % 10 == 0:
        nueva_lectura = np.random.uniform(45, 55, 1)[0]
        tipo_anomalia = "Pico Alto"
        sugerencia_accion = "Revisar posibles sobrecargas, fallos en ventilación o componentes sobrecalentados." 
    elif i % 15 == 0:
        nueva_lectura = np.random.uniform(5, 10, 1)[0]
        tipo_anomalia = "Caída Baja"
        sugerencia_accion = "Verificar si el sensor está desconectado, dañado o hay un problema en la fuente de energía." 
    elif i % 25 == 0:
        nueva_lectura = 23.0
        tipo_anomalia = "Valor Constante"
        sugerencia_accion = "Inspeccionar el sensor por posibles fallas de congelación, cortocircuito o falta de comunicación." 

    prediccion = model.predict(np.array(nueva_lectura).reshape(1, -1))

    estado_lectura = "Normal"
    color_lectura = "green"
    mensaje_alerta = ""

    if prediccion == -1:
        st.session_state['total_anomalies_detected'] += 1 

        estado_lectura = "ANOMALÍA DETECTADA"
        color_lectura = "red"
        mensaje_alerta = (f"🚨 **¡ALERTA!** Se ha detectado una **ANOMALÍA** "
                          f"({tipo_anomalia}) en la lectura del sensor: **{nueva_lectura:.2f}°C**. "
                          f"¡Se recomienda revisar el sistema!")
        alerta_container.error(mensaje_alerta)
        
        action_suggestion_container.info(f"💡 **Sugerencia de Acción:** {sugerencia_accion}")

        current_time = time.time()
        if (current_time - st.session_state['last_alert_time']) > COOLDOWN_SECONDS:
            send_discord_alert(nueva_lectura, tipo_anomalia, sugerencia_accion) 
            st.session_state['last_alert_time'] = current_time 
            st.session_state['total_alerts_sent'] += 1 
            st.info(f"✅ Alerta de Discord enviada (próxima alerta en {COOLDOWN_SECONDS}s).")
        else:
            tiempo_restante = int(COOLDOWN_SECONDS - (current_time - st.session_state['last_alert_time']))
            st.warning(f"⚠️ Anomalía detectada, pero alerta omitida (cooldown activo). Próxima alerta en {tiempo_restante} segundos.")
    else:
        alerta_container.empty()
        action_suggestion_container.empty() 

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
        
        ax.plot(df_para_grafico['Hora'], df_para_grafico['valor_numerico'], label='Temperatura', color='#00FFFF', linewidth=2) # Color cian para la línea
        
        anomalias_grafico = df_para_grafico[df_para_grafico['Estado'] == 'ANOMALÍA DETECTADA']
        if not anomalias_grafico.empty:
            ax.scatter(anomalias_grafico['Hora'], anomalias_grafico['valor_numerico'], color='#FF0000', s=100, marker='X', linewidths=1, edgecolors='white', label='Anomalía') # Rojo para anomalías

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

    time.sleep(st.session_state['simulation_speed']) 

st.success("✅ Simulación de monitoreo de sensor de temperatura finalizada.")