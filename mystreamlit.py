import streamlit as st
import pandas as pd
import joblib

# 1. Cargar la Pipeline
# Cargamos el objeto joblib que contiene el modelo y el preprocesamiento
pipeline = joblib.load('modelo_final.joblib')

st.set_page_config(page_title="Validador G07", page_icon="🏦", layout="wide") 
st.title("Panel de Predicción de Depósitos")
st.write("Introduzca los datos del cliente directamente en el panel central para validar el modelo.")

# 2. Interfaz organizada en 3 columnas
st.subheader("Formulario de Entrada de Datos")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Edad (Age)", value=43)
    job = st.selectbox("Trabajo (Job)", ['management', 'technician', 'admin.', 'blue-collar', 'retired', 'services', 'entrepreneur', 'housemaid', 'self-employed', 'student', 'unemployed', 'unknown'])
    marital = st.selectbox("Estado Civil (Marital)", ['married', 'single', 'divorced'])
    education = st.selectbox("Educación (Education)", ['tertiary', 'secondary', 'primary', 'unknown'])
    default = st.selectbox("Impagos (Default)", ['no', 'yes'])

with col2:
    balance = st.number_input("Saldo (Balance)", value=78)
    housing = st.selectbox("Hipoteca (Housing)", ['yes', 'no'])
    loan = st.selectbox("Préstamo (Loan)", ['no', 'yes'])
    contact = st.selectbox("Contacto (Contact)", ['cellular', 'unknown', 'telephone'])
    day = st.number_input("Día (Day)", value=21)
    month = st.selectbox("Mes (Month)", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])

with col3:
    duration = st.number_input("Duración (Duration)", value=36)
    campaign = st.number_input("Campaña (Campaign)", value=1)
    pdays = st.number_input("Pdays", value=109)
    previous = st.number_input("Previous", value=1)
    poutcome = st.selectbox("Poutcome", ['other', 'unknown', 'failure', 'success'])

st.divider()

# 3. Botón de predicción
if st.button("Realizar Predicción Final", use_container_width=True):
    data = pd.DataFrame({
        'age': [age], 'job': [job], 'marital': [marital], 'education': [education],
        'default': [default], 'balance': [balance], 'housing': [housing], 'loan': [loan],
        'contact': [contact], 'day': [day], 'month': [month], 'duration': [duration],
        'campaign': [campaign], 'pdays': [pdays], 'previous': [previous], 'poutcome': [poutcome]
    })
    
    # Se calculan la predicción final y las probabilidades asociadas
    pred = pipeline.predict(data)[0]
    proba = pipeline.predict_proba(data)[0]
    classes = pipeline.classes_ # Esto devuelve ['no', 'yes']
    
    st.subheader("🔍 Resultado del Análisis")
    
    # Mensaje principal de predicción
    if pred == 'no':
        st.error(f"EL MODELO PREDICE: {pred.upper()} (El cliente NO contratará)")
    else:
        st.success(f"EL MODELO PREDICE: {pred.upper()} (¡El cliente SÍ contratará!)")
    
    # 4. Visualización de probabilidades 
    st.write("### Nivel de Certeza del Modelo:")
    p_col1, p_col2 = st.columns(2)
    
    for i, (cls, p) in enumerate(zip(classes, proba)):
        label = "Probabilidad de Éxito (YES)" if cls == 'yes' else "Probabilidad de Rechazo (NO)"
        if cls == 'yes':
            p_col2.metric(label=label, value=f"{p*100:.2f}%")
        else:
            p_col1.metric(label=label, value=f"{p*100:.2f}%")

    with st.expander("Click para ver detalles técnicos de la instancia enviada"):
        st.table(data)