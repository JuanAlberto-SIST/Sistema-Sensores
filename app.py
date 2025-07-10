import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import time
import matplotlib.pyplot as plt
import altair as alt
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
                "description": f"Se ha detectado una **ANOMALÍA** en el sensor de temperatura.\n\n"
                               f"**Sugerencia de Acción:** {action_suggestion_text}",
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

if 'total_anomalies_detected' not in st.session_state:
    st.session_state['total_anomalies_detected'] = 0
if 'total_alerts_sent' not in st.session_state:
    st.session_state['total_alerts_sent'] = 0
if 'simulation_speed' not in st.session_state:
    st.session_state['simulation_speed'] = 0.5 

if 'last_alert_time' not in st.session_state:
    st.session_state['last_alert_time'] = 0 
COOLDOWN_SECONDS = 60 

st.title("🌡️ Precisa Temp: Sistema de Predicción de Fallos en Sensores")
st.markdown("---")

st.header("Análisis de Viabilidad del Emprendimiento")
st.markdown("") 

st.subheader("Contexto y Declaración del Problema")
with st.expander("Ver el problema que resolvemos..."): 
    st.markdown("""
    Las **fallas frecuentes en sensores de temperatura industrial** generan mediciones imprecisas que afectan la calidad del producto y la seguridad operativa. Sensores inexactos causan **combustión ineficiente, más emisiones y gasto extra**. Esto provoca **paros no planificados, pérdida de calidad en productos, riesgos para la seguridad industrial y mayores costos**.
    """)
st.markdown("---") 

st.subheader("Nuestra Solución: Sensores Inteligentes y Software")
with st.expander("Descubrir cómo lo solucionamos..."): 
    st.markdown("""
    **Precisa Temp** ofrece **sensores inteligentes y software que previenen fallas en temperatura para procesos industriales**. Nuestro sistema monitorea sensores térmicos en **tiempo real** y **detecta fallas para evitar paros y mejorar la eficiencia industrial**. Combina **autodiagnóstico en tiempo real con mantenimiento predictivo basado en machine learning**, integrándose fácilmente a sistemas existentes.
    """)
st.markdown("---") 

st.subheader("Beneficios Clave de Precisa Temp")
with st.expander("Explorar los beneficios..."): 
    st.markdown("""
    * **Beneficios Funcionales:** Medición precisa y continua de la temperatura. Detección temprana de variaciones para evitar daños en equipos. Reducción de tiempos de inactividad mediante alertas preventivas.
    * **Beneficios Emocionales:** Proporciona tranquilidad y confianza al saber que los equipos están protegidos y los procesos funcionan sin riesgos ni pérdidas.
    * **Beneficios para la Sociedad:** Mejora la eficiencia energética y reduce el consumo, disminuyendo emisiones contaminantes.
    """)
st.markdown("---") 

st.header("Demostración del Monitoreo en Tiempo Real")
st.markdown("") 


st.sidebar.header("Control de Simulación")
st.session_state['simulation_speed'] = st.sidebar.slider(
    "Velocidad de Lectura (segundos por lectura)",
    min_value=0.1, max_value=2.0, value=0.5, step=0.1,
    help="Define el tiempo de espera entre cada lectura simulada."
)

status_indicator_container = st.empty() 

st.markdown("""
<style>
.stApp {
    background-color: #1a1a1a;
}
</style>
""", unsafe_allow_html=True)


st.subheader("Monitoreo de Temperatura en Tiempo Real")

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
action_suggestion_container = st.empty()

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

    prediccion = model.predict(np.array(nueva_lectura).reshape(-1, 1))

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

    time.sleep(st.session_state['simulation_speed']) 

st.success("✅ Simulación de monitoreo de sensor de temperatura finalizada.")