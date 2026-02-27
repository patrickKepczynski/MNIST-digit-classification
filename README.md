# 🎨 MNIST Digit Recognizer - Streamlit App

This is an interactive Machine Learning web application built with Python and Streamlit. The app allows users to draw a number (0-9) on a digital canvas, and the ML model will predict the drawn digit in real-time.

## 🧠 The Machine Learning Model
The prediction engine is powered by an **ExtraTreesClassifier** trained on the classic MNIST dataset (70,000 images). 

**Key optimizations in the pipeline:**
* **Binarization:** The input data (both training data and user drawings) is binarized to remove grayscale noise, making the model highly robust against different drawing styles.
* **Centering:** The app calculates the center of mass of the drawn digit and aligns it perfectly before prediction, mimicking the original MNIST dataset format.
* **Accuracy:** The finalized model achieved a **>97% accuracy** on unseen test data.

## 🚀 Live Demo
You can try the live version of the app here: [LÄNK TILL DIN STREAMLIT APP]

## 🛠️ How to run it locally

1. **Clone this repository** (or download the files):
   ```bash
   git clone <your-repository-url>

2. **Install the required dependencies:**
Make sure you have Python installed, then run:
    ```bash
    pip install -r requirements.txt

3. **Run the Streamlit app:**
    ```bash
    streamlit run app.py 

## 📂 Project Structure

**app.py:** The main Streamlit application code.

**MNIST_ML_model.ipynb:** The Jupyter Notebook containing the data exploration, ML-model comparisons, pipeline creation and of course model training.

**Teoretiska_frågor_Python_ML.ipynb:** Questions and answers related to the course Machine Learning at YrkesAkademin Mölndal.

**best_model_deploy.joblib:** The exported ExtraTrees model.

**scaler_deploy.joblib:** The exported StandardScaler.