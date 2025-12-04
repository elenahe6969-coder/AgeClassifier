import streamlit as st
from transformers import pipeline
from PIL import Image

st.title("👤 Age Classifier")
st.write("Upload a photo to predict age range")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file and st.button("Analyze"):
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Analyzing..."):
        classifier = pipeline("image-classification", model="nateraw/vit-age-classifier")
        predictions = classifier(image)
        predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
    
    st.success(f"Predicted Age Range: {predictions[0]['label']}")
    st.write(f"Confidence: {predictions[0]['score']:.1%}")
    
    st.write("All predictions:")
    for pred in predictions:
        st.write(f"- {pred['label']}: {pred['score']:.1%}")
