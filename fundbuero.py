import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Titel der App
st.title("Fundbüro - Gegenstandserkennung")
st.write("Lade ein Foto des Fundstücks hoch (Flasche, Schere oder Federtasche).")

# Modell und Labels laden (gecached, damit es schneller geht)
@st.cache_resource
def load_my_model():
    model = load_model("keras_model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    return model, class_names

model, class_names = load_my_model()

# Streamlit File Uploader
uploaded_file = st.file_uploader("Bild auswählen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild öffnen
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Hochgeladenes Bild', use_container_width=True)
    
    # Vorbereitung für das Modell
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    
    # Normalisierung
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Vorhersage
    with st.spinner('Analysiere Bild...'):
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

    # Ergebnis anzeigen
    st.subheader(f"Ergebnis: {class_name[2:]}")
    st.write(f"Wahrscheinlichkeit: {confidence_score:.2%}")
