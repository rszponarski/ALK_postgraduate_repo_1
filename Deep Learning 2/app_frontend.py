import streamlit as st
import requests

st.title("Plant Seedlings Classifier")

uploaded_file = st.file_uploader(
    "Wybierz zdjęcie sadzonki", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image (uploaded_file, caption="Wybrane zdjęcie", use_column_width=True)

    if st.button("Klasyfikuj"):
        files = {"file": {uploaded_file.name, uploaded_file, uploaded_file.type}}
        with st.spinner ("Przetwarzanie..."):
            response = requests.post("http://localhost:8000/predict", files=files)
        if response.status_code == 200:
            data = response.json()
            st.success(f"Klasa: {data['class']}")
            st.write(f"Pewność: {data['confidence']:.2f}")
        else:
            st.error("Coś poszło nie tak, spróbuj ponownie")