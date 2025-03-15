import streamlit as st
import cv2
import numpy as np
import base64
import streamlit.components.v1 as components
from ultralytics import YOLO
import torch
import tempfile

# Verificar si Torch y CUDA están disponibles
st.write(f"Torch versión: {torch.__version__}")
st.write(f"CUDA disponible: {torch.cuda.is_available()}")

# Cargar el modelo YOLOv8 con manejo de errores
try:
    model = YOLO("yolov8n.pt")
except Exception as e:
    st.error(f"Error al cargar el modelo YOLO: {e}")
    st.stop()

# Configurar Streamlit
st.title("Detección de Personas en Tiempo Real")
start = st.checkbox("Iniciar detección")  # Botón de encendido/apagado

stframe = st.empty()  # Espacio para mostrar el video

# Función para generar una alerta sonora
def alerta_sonora():
    try:
        with open("audio.wav", "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
        audio_html = f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
        </audio>
        """
        components.html(audio_html, height=0)
    except Exception as e:
        st.error(f"Error al cargar el archivo de audio: {e}")

# Capturar imagen desde la cámara
camera_input = st.camera_input("Captura desde tu cámara")

if camera_input and start:
    try:
        # Guardar la imagen en un archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(camera_input.getvalue())
            temp_path = temp_file.name

        # Leer la imagen con OpenCV
        frame = cv2.imread(temp_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detección con YOLO
        results = model.predict(frame)
        detected = False  # Variable para saber si hay personas

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0].item())  # Clase detectada

                if model.names[cls] == "person":  # Si detecta una persona
                    detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas de la caja
                    conf = box.conf[0].item()  # Confianza

                    # Dibujar caja y etiqueta en la imagen
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Si detectó una persona, activar la alerta sonora
        if detected:
            alerta_sonora()

        # Mostrar la imagen con detección en Streamlit
        stframe.image(frame, channels="RGB", use_column_width=True)

    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")
