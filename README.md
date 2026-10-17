Práctica 1: Predicción de Suscripción a un Producto Bancario
Asignatura: Aprendizaje Automático
Grupo: G07
Integrantes: Rodrigo Mohedano García Semilla (SEED): 72 (basada en el NIA)

📌 Descripción del Proyecto
El objetivo principal de este trabajo es desarrollar y desplegar un flujo completo de aprendizaje automático para predecir si un cliente bancario suscribirá un depósito a plazo fijo (deposit = 'yes' / 'no').

A través de datos demográficos, financieros y de interacción comercial, el proceso abarca desde el análisis exploratorio inicial hasta el empaquetado del modelo final y su despliegue operativo en una aplicación web interactiva.

⚙️ Estructura del Repositorio y Entregables
Notebook_Principal.ipynb: Contiene el análisis exploratorio (EDA), la definición del esquema de validación, la comparación de escaladores, el entrenamiento de modelos base, la optimización de hiperparámetros (HPO), el análisis de sensibilidad y la tarea de elección abierta. Se entrega con todas las salidas ejecutadas.
Notebook_Competicion.ipynb: Cuaderno reproducible encargado de cargar el pipeline final ya ajustado (modelo_final.joblib), procesar el conjunto ciego de evaluación (bank_competition_72.pkl) y generar el archivo de salida correspondiente.
modelo_final.joblib: Pipeline serializada completa que integra el preprocesamiento (ColumnTransformer con escalado y codificación One-Hot) junto con el estimador final entrenado.
predicciones.csv: Archivo con las etiquetas predichas para el conjunto de test de competición.
mystreamlit.py: Script de Python con la aplicación interactiva desarrollada en Streamlit para la inferencia de nuevos perfiles en tiempo real.
Evidencias_Streamlit.pdf: Documento con capturas de pantalla comparativas que certifican la concordancia exacta entre las probabilidades de la pipeline en local y la interfaz web.
🔬 Metodología y Decisiones Clave
Estrategia de Evaluación:

Outer (Rendimiento futuro): Holdout estratificado con división 2/3 (entrenamiento) y 1/3 (test).
Inner (Optimización y Selección): Validación cruzada estratificada de 3 particiones (Stratified 3-Fold Cross-Validation).
Métrica principal: Exactitud (Accuracy), complementada con matrices de confusión y análisis de probabilidades.
Preprocesamiento:

La variable pdays fue analizada y tratada para reflejar adecuadamente la ausencia de contacto previo (valores -1).
Para las variables numéricas se evaluaron StandardScaler, MinMaxScaler y RobustScaler, siendo seleccionado el escalador de mejor desempeño mediante KNN.
Las variables categóricas fueron transformadas mediante OneHotEncoder(handle_unknown='ignore').
Modelos Evaluados y Optimización (HPO):

Modelos básicos: K-Nearest Neighbors (KNN) y Árboles de Decisión (con análisis de interpretabilidad mediante árboles de profundidad controlada).
Modelos avanzados: Regresión Logística (con y sin regularización L1/L2) y Support Vector Machines (SVM).
Tarea de elección abierta: Exploración y optimización de ensambles (Random Forest / balanceo) para maximizar la capacidad de generalización.
Todos los resultados y tiempos de cómputo fueron tabulados y comparados sistemáticamente.
🚀 Instrucciones de Ejecución
1. Requisitos e Instalación
Se recomienda el uso de un entorno virtual con Python 3.10 o superior:

pip install numpy pandas scikit-learn matplotlib seaborn joblib streamlit
2. Generación de Predicciones
Para reproducir las predicciones sobre el conjunto de test de competición:

Abrir y ejecutar todas las celdas de Notebook_Competicion.ipynb.

El archivo predicciones.csv será generado automáticamente en la raíz del proyecto mediante rutas relativas.

3. Ejecución de la Aplicación Web (Streamlit)
Para iniciar la interfaz de predicción local:

streamlit run mystreamlit.py
🤖 Declaración de Uso de IA Generativa
Durante el desarrollo de la práctica, fueron utilizadas herramientas de Inteligencia Artificial Generativa como apoyo técnico para:

Optimización y modularización del código de preprocesamiento en Pipelines.

Estructuración de tablas comparativas de resultados y visualización en Pandas.

Asistencia en la redacción formal de justificaciones teóricas y conclusiones analíticas.

Todas las salidas, hiperparámetros y lógicas generadas fueron contrastadas y verificadas empíricamente mediante validación cruzada.
