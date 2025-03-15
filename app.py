import streamlit as st
import cv2
import numpy as np
import base64
import streamlit.components.v1 as components
from ultralytics import YOLO

# Cargar el modelo YOLOv8
model = YOLO("yolov8n.pt")  # Puedes cambiar a 'yolov8s.pt' para mayor precisión

# Configurar Streamlit
st.title("Detección de Personas en Tiempo Real")
start = st.checkbox("Iniciar detección")  # Botón de encendido/apagado

# Widget para subir un archivo de audio
uploaded_audio = st.file_uploader("Sube un archivo de sonido (formato WAV)", type=["wav"])

# Sonido predeterminado
DEFAULT_AUDIO_PATH = "default_audio.wav"  # Asegúrate de tener este archivo en el mismo directorio

stframe = st.empty()  # Espacio para mostrar el video

# Función para generar una alerta sonora
def alerta_sonora(audio_file):
    audio_html = """
    <audio id="audio" controls>
        <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
    </audio>
    <script>
        document.getElementById("audio").play();
    </script>
    """
    try:
        if audio_file is not None:
            audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
        else:
            with open(DEFAULT_AUDIO_PATH, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode('utf-8')
        components.html(audio_html.format(audio_base64=audio_base64), height=0)
    except Exception as e:
        st.error(f"Error al reproducir el archivo de audio: {e}")

# Usar el widget de cámara de Streamlit
camera_input = st.camera_input("Captura desde tu cámara")

if camera_input and start:
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
        alerta_sonora(uploaded_audio)

    # Mostrar la imagen con detección en Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    stframe.image(frame, channels="RGB", use_column_width=True)