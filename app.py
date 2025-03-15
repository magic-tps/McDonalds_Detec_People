import streamlit as st
import cv2
import numpy as np
import platform
import os
import base64
import streamlit.components.v1 as components
from ultralytics import YOLO

# Cargar el modelo YOLOv8
model = YOLO("yolov8n.pt")  # Puedes cambiar a 'yolov8s.pt' para mayor precisión

# Configurar Streamlit
st.title("Detección de Personas en Tiempo Real")
start = st.checkbox("Iniciar detección")  # Botón de encendido/apagado
stframe = st.empty()  # Espacio para mostrar el video

# Inicializar la cámara
cap = cv2.VideoCapture(0)

def alerta_sonora():
    """Genera una alerta sonora cuando se detecta una persona."""
    try:
        audio_file = open("audio.wav", "rb").read()
        audio_base64 = base64.b64encode(audio_file).decode('utf-8')
        audio_html = f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
        </audio>
        """
        components.html(audio_html, height=0)
    except Exception as e:
        st.error(f"Error al cargar el archivo de audio: {e}")

# Verificar si la cámara está disponible
if not cap.isOpened():
    st.error("No se pudo acceder a la cámara.")
else:
    while start:
        ret, frame = cap.read()
        if not ret:
            st.error("Error al capturar el video.")
            break

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

        # Mostrar video en Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", use_column_width=True)

cap.release()
