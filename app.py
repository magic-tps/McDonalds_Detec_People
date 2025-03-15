import streamlit as st
import numpy as np
import cv2
import platform
import playsound  # Usaremos playsound para reproducir audio
from ultralytics import YOLO

# Cargar el modelo YOLOv8
model = YOLO("yolov8n.pt")  # Puedes cambiar a 'yolov8s.pt' para mayor precisión

# Configurar Streamlit
st.title("Detección de Personas en Tiempo Real")
start = st.checkbox("Iniciar detección")  # Botón de encendido/apagado

stframe = st.empty()  # Espacio para mostrar el video

# Usar el widget de cámara de Streamlit
camera_input = st.camera_input("Captura desde tu cámara")

# Función para generar una alerta sonora usando playsound
def alerta_sonora():
    try:
        # Reproducir un archivo de sonido cuando se detecta una persona
        playsound.playsound("alerta.wav", block=False)
    except Exception as e:
        st.error(f"Error al reproducir sonido: {e}")

# Verificar si se ha capturado una imagen
if camera_input:
    # Convertir la imagen capturada en un formato adecuado para YOLO
    frame = np.array(camera_input)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convertir de RGB a BGR

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
    stframe.image(frame, channels="BGR", use_column_width=True)
