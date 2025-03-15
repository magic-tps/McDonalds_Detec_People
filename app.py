import streamlit as st
import cv2
import numpy as np
import torch
import os
import platform
import base64
import streamlit.components.v1 as components
from ultralytics import YOLO

# Verificar si la GPU está disponible para YOLO
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Cargar el modelo YOLOv8 en el dispositivo adecuado
model = YOLO("yolov8n.pt").to(device)

# Configurar Streamlit
st.title("🔍 Detección de Personas en Tiempo Real")
start = st.checkbox("Iniciar detección")

# Espacio para mostrar el video
stframe = st.empty()

# Función para generar alerta sonora
def alerta_sonora():
    sistema = platform.system()
    
    if sistema == "Windows":
        import winsound
        winsound.Beep(1000, 500)  # Sonido en Windows (1000 Hz por 500 ms)
    elif sistema == "Darwin":  # macOS
        os.system("afplay /System/Library/Sounds/Glass.aiff")
    else:  # Linux
        os.system("paplay /usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga")

# Captura de la cámara desde Streamlit
camera_input = st.camera_input("📷 Captura desde tu cámara")

if camera_input and start:
    # Leer la imagen capturada
    file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convertir a RGB para YOLO
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Realizar detección con YOLO
    results = model.predict(frame_rgb, device=device)
    detected = False  # Variable para verificar si hay personas

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0].item())  # Obtener la clase detectada
            if model.names[cls] == "person":
                detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas de la caja
                conf = box.conf[0].item()  # Confianza del modelo

                # Dibujar cuadro y etiqueta en la imagen
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Si detectó una persona, activar la alerta sonora
    if detected:
        alerta_sonora()

    # Mostrar la imagen con detección en Streamlit
    stframe.image(frame, channels="BGR", use_column_width=True)
