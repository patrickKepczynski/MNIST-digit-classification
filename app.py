import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps
import streamlit_drawable_canvas
from scipy.ndimage import center_of_mass

st.set_page_config(layout="wide", page_title="MNIST 97%")
st.title("Draw a number between 0 and 9. The machine learning model will try to guess your number!")

# Laddar modell
@st.cache_resource
def load_model():
    model = joblib.load('best_model_deploy.joblib')
    scaler = joblib.load('scaler_deploy.joblib')
    return model, scaler

model, scaler = load_model()
st.success("ML model loaded successfully! This ExtraTrees model is trained on the MNIST dataset and has 97.24% accuracy on test set.")

# 1. Ritplattan
canvas_result = streamlit_drawable_canvas.st_canvas(
    stroke_width=20,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# 2. Logiken

    # Förbehandling av ritad data
if canvas_result.image_data is not None:

    # Hämta bilddata (RGBA) och konvertera till heltal
    img_array = canvas_result.image_data.astype('uint8')
    
    # Extrahera Röd-kanalen för att få en gråskalebild (eftersom vi ritar svart/vitt)
    grayscale = img_array[:, :, 0]
    
    # Invertera färger: Ritplattan har vit bakgrund (255) och svart bläck (0)
    # MNIST-modellen är tränad på svart bakgrund (0) och vitt bläck (255)
    inverted = 255 - grayscale
    
    # Brusreducering: Ta bort svagt gråa "spökpixlar" genom att sätta allt under tröskelvärdet 50 till 0
    inverted[inverted < 50] = 0
    
    # Hitta koordinaterna för alla pixlar som faktiskt innehåller bläck (> 0)
    coords = np.argwhere(inverted > 0)
    
# 2. Centrering och skalning (Bounding Box)
    if len(coords) > 0:
        # Hitta ytkanterna för den ritade siffran (min/max x och y)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1
        
        # Beskär bilden (Crop) så att endast siffran behålls utan onödig tom yta
        cropped_array = inverted[y_min:y_max, x_min:x_max]
        
        # Konvertera tillbaka till en PIL-bild för att kunna använda avancerad skalning
        cropped_img = Image.fromarray(cropped_array)
        
        # Beräkna skalfaktor: MNIST-standard kräver att siffran ryms inom en 20x20-ruta
        w, h = cropped_img.size
        ratio = 20.0 / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
            
        # Ändra storlek med LANCZOS-filter för att behålla bildskärpa och undvika aliasing
        resized_img = cropped_img.resize(new_size)
        
        # Gör om till numpy-array igen för den avslutande matematiska centreringen
        resized_array = np.array(resized_img)
        
        # Masscentrum
        # Räknar ut tyngdpunkten på bläcket i siffran
        cy, cx = center_of_mass(resized_array)
        
        # Räknar ut var bilden ska klistras in för att tyngdpunkten ska hamna exakt i mitten (pixel 14, 14)
        paste_x = int(round(14.0 - cx))
        paste_y = int(round(14.0 - cy))
        
        # Skapar 28x28 bakgrund och klistrar in
        final_img = Image.new('L', (28, 28), color=0)
        final_img.paste(resized_img, (paste_x, paste_y))
        
        # Platta ut bilden till 784 features
        final_array = np.array(final_img).reshape(1, -1)

        # Binariserar bilden
        # Gör om alla pixlar som är större än 0 till 1, resten förblir 0
        final_array_bin = (final_array > 0).astype(int)

        # Skala den BINÄRA datan och gissa!
        scaled_data = scaler.transform(final_array_bin)
        prediction = model.predict(scaled_data)
        
        st.header(f"I think your number is: {prediction[0]}")
        st.write("Reset canvas on bin icon to the right of the arrow.")

        # Skalar upp pixelvärdena från [0, 1] till [0, 255] för att den binära matrisen ska kunna visualiseras korrekt som en bild.
        display_array = (final_array_bin.reshape(28, 28) * 255).astype('uint8')
        display_img = Image.fromarray(display_array)

        # Verifierar hur ML-modellen ser på vår input efter all förbehandling (viktcentrerad och binariserad)
        # st.image(display_img, caption="How the ML model views your input (weight-centered & binarized)", width=150)
    else:
        st.write("Draw something to see the prediction!")