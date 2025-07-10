import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import time
import matplotlib.pyplot as plt
import requests
import altair as alt

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

status_indicator_container = st.empty() 

st.markdown("""
<style>
/* Fondo general y texto principal */
.stApp {
    background-color: #222222; /* Un gris oscuro más suave */
    color: #e0e0e0; /* Texto principal casi blanco */
}
/* Títulos y subtítulos */
h1, h2, h3, h4, h5, h6 {
    color: #f0f0f0; /* Títulos más claros */
}

/* Estilo para los KPI boxes */
.stMetric > div {
    background-color: #333333; /* Fondo de tarjeta más claro */
    border-radius: 8px;
    padding: 15px; /* Más padding */
    box-shadow: 0 4px 8px rgba(0,0,0,0.3); /* Sombra más pronunciada */
    color: #f0f0f0; /* Texto valor KPI */
}
.stMetric label {
    color: #a0a0a0; /* Etiquetas de KPI más suaves */
    font-size: 1.1em; /* Un poco más grande */
}
.stMetric div[data-testid="stMetricValue"] {
    font-size: 2em; /* Valor KPI más grande */
    font-weight: bold;
    color: #00FFFF; /* Color cian vibrante para los valores */
}

/* Estilo para la tabla de historial */
.dataframe {
    background-color: #333333; /* Fondo de tabla más claro */
    color: #e0e0e0; /* Texto de tabla casi blanco */
    border-radius: 8px;
    padding: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    font-size: 0.9em; /* Fuente un poco más pequeña para la tabla */
}
.dataframe th {
    background-color: #444444; /* Fondo de encabezado de tabla */
    color: #f0f0f0;
    padding: 8px;
}
.dataframe td {
    padding: 8px;
}
/* Estilo para el resaltado de anomalías en la tabla */
.stDataFrame tbody tr td:nth-child(3) div[data-value*="ANOMALÍA"] { /* Esto apunta a la columna Estado */
    background-color: #FF6347 !important; /* Rojo tomate, más vibrante */
    color: white !important;
    font-weight: bold;
}
.stDataFrame tbody tr td div[data-value*="ANOMALÍA"] { /* Estilo para cualquier celda con ANOMALÍA */
    background-color: #FF6347 !important; 
    color: white !important;
}

/* Estilo para los mensajes de alerta Streamlit (st.error, st.warning, st.info, st.success) */
div[data-testid="stAlert"] {
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
}
div[data-testid="stAlert"] .st-bv { /* Mensajes de info/warning/success */
    background-color: #333333; /* Un fondo neutro para mensajes */
    color: #f0f0f0;
}
div[data-testid="stAlert"] .st-bv div[data-testid="stMarkdownContainer"] { /* Ajuste para el texto dentro de la alerta */
    font-size: 1.1em;
}
div[data-testid="stAlert"] div[data-testid="stAlertContent"] {
    color: #f0f0f0; /* Color del texto dentro de la alerta */
}


</style>
""", unsafe_allow_html=True)


st.title("🌡️ Precisa Temp: Sistema de Predicción de Fallos en Sensores")
st.markdown("---")


st.subheader("Monitoreo de Temperatura en Tiempo Real")

# Contenedor para KPIs y controles de velocidad (para que estén juntos)
top_section_cols = st.columns([0.7, 0.3])

with top_section_cols[0]: # Columna izquierda para KPIs
    kpi_cols = st.columns(2) 
    with kpi_cols[0]:
        st.metric(label="Total Anomalías Detectadas", value=st.session_state['total_anomalies_detected'])
    with kpi_cols[1]:
        st.metric(label="Alertas Discord Enviadas", value=st.session_state['total_alerts_sent'])

with top_section_cols[1]: # Columna derecha para el slider (sin barra lateral)
    st.markdown("##### Control de Simulación") # Título para el slider
    st.session_state['simulation_speed'] = st.slider( # Ahora es st.slider, no st.sidebar.slider
        "Velocidad de Lectura (segundos por lectura)",
        min_value=0.1, max_value=2.0, value=0.5, step=0.1,
        help="Define el tiempo de espera entre cada lectura simulada."
    )


st.markdown("---") # Separador visual

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
        if estado_lectura == "ANOMALÍA DETECTADA": # Corregido para coincidir
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
        df_para_grafico = historial_lecturas_df.tail(num_lecturas_grafico).reset_index()

        line_chart = alt.Chart(df_para_grafico).mark_line(color='#00FFFF').encode( 
            x=alt.X('index', axis=None), 
            y=alt.Y('valor_numerico', title='Temperatura (°C)'),
            tooltip=[
                alt.Tooltip('Hora', title='Hora'), 
                alt.Tooltip('valor_numerico', title='Temp', format='.2f'),
                alt.Tooltip('Estado', title='Estado')
            ]
        ).properties(
            title=f'Últimas {num_lecturas_grafico} Lecturas de Temperatura'
        )

        anomaly_points = alt.Chart(df_para_grafico[df_para_grafico['Estado'] == 'ANOMALÍA DETECTADA']).mark_point(
            color='#FF0000', filled=True, size=100, shape='cross'
        ).encode(
            x=alt.X('index'),
            y=alt.Y('valor_numerico'),
            tooltip=[
                alt.Tooltip('Hora', title='Hora'), 
                alt.Tooltip('valor_numerico', title='Temp', format='.2f'),
                alt.Tooltip('Estado', title='Estado'),
                alt.Tooltip('Tipo de Anomalía', title='Tipo Anomalía')
            ]
        )

        chart = alt.layer(line_chart, anomaly_points).interactive() 
        
        st.altair_chart(chart, use_container_width=True)

    with historico_container.container():
        st.subheader("Historial de Lecturas Recientes")
        def highlight_anomalies(s):
            return ['background-color: #FF6347; color: white; font-weight: bold;' if 'ANOMALÍA' in str(v) else '' for v in s]

        st.dataframe(historial_lecturas_df.tail(15).style.apply(highlight_anomalies, axis=1))


    time.sleep(st.session_state['simulation_speed']) 

st.success("✅ Simulación de monitoreo de sensor de temperatura finalizada.")