import streamlit as st
import cv2
import numpy as np
import platform
import os
from ultralytics import YOLO

# Cargar el modelo YOLOv8
model = YOLO("yolov8n.pt")  # Puedes cambiar a 'yolov8s.pt' para mayor precisión

# Configurar Streamlit
st.title("Detección de Personas en Tiempo Real")
start = st.checkbox("Iniciar detección")  # Botón de encendido/apagado

stframe = st.empty()  # Espacio para mostrar el video

# Función para generar una alerta sonora
def alerta_sonora():
    sistema = platform.system()
    if sistema == "Windows":
        import winsound
        winsound.Beep(1000, 500)  # Sonido a 1000 Hz por 500 ms
    else:
        os.system("printf '\a'")  # Beep en Linux/macOS

# Configurar la cámara (usando DroidCam o la cámara web local)
camera_option = st.radio(
    "Selecciona la fuente de la cámara:",
    ("Cámara Web Local", "DroidCam (Celular)")
)

if camera_option == "Cámara Web Local":
    cap = cv2.VideoCapture(0)  # Usar la cámara web local
else:
    droidcam_ip = st.text_input("Ingresa la dirección IP de DroidCam (ejemplo: 192.168.1.100):")
    droidcam_port = st.text_input("Ingresa el puerto de DroidCam (por defecto: 4747):", "4747")
    if droidcam_ip and droidcam_port:
        cap = cv2.VideoCapture(f"http://{droidcam_ip}:{droidcam_port}/video")  # Usar DroidCam
    else:
        st.warning("Por favor, ingresa la dirección IP y el puerto de DroidCam.")
        cap = None

# Verificar si la cámara está disponible
if cap is not None and not cap.isOpened():
    st.error("No se pudo acceder a la cámara.")
else:
    while start and cap is not None:
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

if cap is not None:
    cap.release()