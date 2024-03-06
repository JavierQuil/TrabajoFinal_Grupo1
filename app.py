import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import requests
from io import BytesIO

# Cargar el modelo entrenado
with open('modelo_optimizado_Grupo_1.pkl', 'rb') as file:
    modelo = pickle.load(file)

# Obtener la imagen desde la URL
url = "https://www.amsac.pe/images/Laptop2024.jpg"
response = requests.get(url)
image = Image.open(BytesIO(response.content))

# Definir la interfaz de usuario en Streamlit
st.title('Predicción de Precios de Laptops')
#agregar imagen
st.image(image, caption="Laptop 2024")

# Controles de entrada para las características
ram = st.number_input('RAM (GB)', min_value=1, max_value=64, value=8)
ghz = st.number_input('GHz del CPU', min_value=0.1, max_value=5.0, value=2.5)
screen_width = st.number_input('Ancho de Pantalla', min_value=800, max_value=4000, value=1920)
screen_height = st.number_input('Alto de Pantalla', min_value=600, max_value=3000, value=1080)
type_gaming = st.selectbox('¿Es Gaming?', ['No', 'Sí'])
type_notebook = st.selectbox('¿Es Notebook?', ['No', 'Sí'])

# Convertir entradas a formato numérico
type_gaming = 1 if type_gaming == 'Sí' else 0
type_notebook = 1 if type_notebook == 'Sí' else 0

# Botón para realizar predicción
if st.button('Predecir Precio'):
    # Crear DataFrame con las entradas
    input_data = pd.DataFrame([[ram,ghz, screen_width, screen_height, type_gaming, type_notebook]],
                    columns=['Ram', 'GHz', 'screen_width', 'screen_height', 'SSD', 'TypeName_Gaming', 'TypeName_Notebook'])

    # Estandarización de las características
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_data)

    # Realizar predicción
    prediction = modelo.predict(input_scaled)

    # Mostrar predicción
    st.write(f'Precio predecido: {prediction[0]:.2f} euros')


