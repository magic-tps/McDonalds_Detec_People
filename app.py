import streamlit as st
import cv2
import numpy as np
import platform
import os
import pygame
from ultralytics import YOLO

# Cargar el modelo YOLOv8
model = YOLO("yolov8n.pt")  # Puedes cambiar a 'yolov8s.pt' para mayor precisión

# Configurar Streamlit
st.title("Detección de Personas en Tiempo Real")
start = st.checkbox("Iniciar detección")  # Botón de encendido/apagado

stframe = st.empty()  # Espacio para mostrar el video
cap = cv2.VideoCapture(1)  # Iniciar cámara

# Función para generar una alerta sonora
def alerta_sonora():
    sistema = platform.system()
    if sistema == "Windows":
        import winsound
        winsound.Beep(1000, 500)  # Sonido a 1000 Hz por 500 ms
    else:
        os.system("printf '\a'")  # Beep en Linux/macOS

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
