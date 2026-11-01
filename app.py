
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="Remote Sensing Image Classification Using EuroSAT",
    page_icon="🛰️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "eurosat_cnn_final.keras"

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]

DESCRIPTIONS = {
    "AnnualCrop": "Agricultural land used for seasonal crop cultivation.",
    "Forest": "Land covered with trees and forest vegetation.",
    "HerbaceousVegetation": "Areas dominated by grasses and low vegetation.",
    "Highway": "Road networks and major transportation routes.",
    "Industrial": "Factories, industrial buildings and infrastructure.",
    "Pasture": "Grassland used for grazing livestock.",
    "PermanentCrop": "Land used for crops grown over multiple seasons.",
    "Residential": "Urban areas containing houses and settlements.",
    "River": "Natural flowing water bodies.",
    "SeaLake": "Large water bodies including seas and lakes."
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# ---------------- SATELLITE THEME ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #071321;
    color: #edf5ff;
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1300px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

#MainMenu, footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

h1, h2, h3 {
    color: #f2f7ff;
    font-family: 'Inter', sans-serif;
    letter-spacing: -0.7px;
}

p, label {
    color: #b7c9dc;
}

.hero {
    background:
        radial-gradient(ellipse at 85% 20%, #145c87 0%, transparent 35%),
        linear-gradient(120deg, #102d49, #091a2c 65%, #123d59);
    border: 1px solid #285778;
    border-radius: 22px;
    padding: 60px 48px;
    position: relative;
    overflow: hidden;
}

.hero h1 {
    font-size: clamp(34px, 5vw, 58px);
    font-weight: 800;
    line-height: 1.15;
    max-width: 760px;
}

.hero p {
    font-size: 16px;
    line-height: 1.8;
    max-width: 650px;
}

.eyebrow {
    color: #55c9f4;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2.5px;
    margin-bottom: 15px;
}

.earth {
    position: absolute;
    width: 240px;
    height: 240px;
    border-radius: 50%;
    right: 7%;
    top: 18%;
    background:
        radial-gradient(circle at 35% 30%, #a9eaff, #3285b9 35%, #125078 65%, #071321 78%);
    box-shadow: 0 0 65px #3eb6ef55;
    border: 2px solid #7ed9ff66;
}

.stat-card {
    background: #10253a;
    border: 1px solid #25435e;
    border-radius: 15px;
    padding: 22px;
    min-height: 115px;
}

.stat-number {
    color: #59ccf5;
    font-size: 30px;
    font-weight: 800;
}

.stat-label {
    color: #a9bed2;
    font-size: 12px;
    letter-spacing: 1px;
    margin-top: 7px;
}

.feature-card {
    background: #10253a;
    border: 1px solid #25435e;
    border-radius: 15px;
    padding: 23px;
    min-height: 135px;
    margin-bottom: 13px;
}

.feature-card h3 {
    font-size: 19px;
}

.feature-card p {
    font-size: 14px;
    line-height: 1.7;
}

div[data-testid="stRadio"] > div {
    background: #10253a;
    border: 1px solid #25435e;
    border-radius: 12px;
    padding: 9px;
}

div[data-testid="stRadio"] label {
    color: #edf5ff !important;
}

.stButton button {
    background: #35b9eb;
    color: #061525;
    border: none;
    border-radius: 9px;
    min-height: 45px;
    font-weight: 800;
}

.stButton button:hover {
    background: #8cdefb;
    color: #061525;
}

div[data-testid="stFileUploader"] {
    background: #10253a;
    border: 1px dashed #3986ad;
    border-radius: 15px;
    padding: 20px;
}

div[data-testid="stMetric"] {
    background: #10253a;
    border: 1px solid #25435e;
    border-radius: 12px;
    padding: 16px;
}

hr {
    border-color: #25435e;
}

.footer {
    border-top: 1px solid #25435e;
    padding-top: 20px;
    margin-top: 50px;
    color: #8ba6bd;
    font-size: 11px;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- NAVIGATION ----------------

st.markdown(
    '<div class="eyebrow">🛰️ SATELLITE EARTH OBSERVATION</div>',
    unsafe_allow_html=True
)

page = st.radio(
    "Navigation",
    ["Home", "Classify", "Explore Data", "Model"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")

# ---------------- HOME ----------------

if page == "Home":

    st.markdown("""
    <div class="hero">
        <div class="earth"></div>
        <div class="eyebrow">ARTIFICIAL INTELLIGENCE · REMOTE SENSING</div>
        <h1>Remote Sensing Image Classification Using EuroSAT</h1>
        <p>
        Explore our planet through satellite imagery.
        Use deep learning to identify land use and land cover
        patterns from Earth observation images.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.markdown("### Project Overview")

    cols = st.columns(4)
    stats = [
        ("27,000", "SATELLITE IMAGES"),
        ("10", "LAND COVER CLASSES"),
        ("88.47%", "TEST ACCURACY"),
        ("64 × 64", "INPUT IMAGE SIZE")
    ]

    for col, (number, label) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{number}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    st.markdown("### Explore the Application")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="feature-card">
            <h3>🛰️ Satellite Image Classification</h3>
            <p>Upload an RGB satellite image and predict its land-cover category using a trained CNN model.</p>
        </div>
        <div class="feature-card">
            <h3>🌍 EuroSAT Dataset Explorer</h3>
            <p>Explore satellite image samples from ten different land-cover categories.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="feature-card">
            <h3>🧠 Deep Learning Model</h3>
            <p>Understand the CNN model, its training process and classification performance.</p>
        </div>
        <div class="feature-card">
            <h3>📡 Earth Observation</h3>
            <p>Discover how satellite imagery helps monitor vegetation, water, agriculture and urban areas.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Land Cover Categories")

    gallery = st.columns(5)

    for i, category in enumerate(CLASSES):
        with gallery[i % 5]:
            folder = BASE_DIR / category
            if folder.is_dir():
                samples = sorted([
                    f for f in folder.iterdir()
                    if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
                ])
                if samples:
                    st.image(str(samples[0]), caption=category, width="stretch")

# ---------------- CLASSIFY ----------------

elif page == "Classify":

    st.markdown('<div class="eyebrow">SATELLITE IMAGE ANALYSIS</div>', unsafe_allow_html=True)
    st.title("Classify Satellite Images")
    st.write("Upload an RGB satellite image to identify its land-cover category.")

    uploaded = st.file_uploader(
        "Upload your satellite image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded is not None:

        try:
            image = Image.open(uploaded).convert("RGB")

            left, right = st.columns([1.1, 1])

            with left:
                st.image(image, caption="Uploaded satellite image", width="stretch")

            with right:
                st.markdown("### Image Details")
                st.write(f"**Filename:** {uploaded.name}")
                st.write(f"**Image dimensions:** {image.width} × {image.height}")
                st.write("**Image format:** RGB")
                st.write("**Model input:** 64 × 64 pixels")

            if st.button("🛰️ ANALYZE IMAGE", width="stretch"):

                with st.spinner("Processing satellite image..."):

                    model = load_model()

                    resized = image.resize((64, 64))
                    array = np.array(resized, dtype=np.float32)
                    array = np.expand_dims(array, axis=0)

                    predictions = model.predict(array, verbose=0)[0]

                    index = int(np.argmax(predictions))
                    predicted_class = CLASSES[index]
                    confidence = float(predictions[index]) * 100

                st.markdown("---")
                st.markdown('<div class="eyebrow">CLASSIFICATION RESULT</div>', unsafe_allow_html=True)

                st.markdown(f"""
                <div class="hero" style="padding:35px;">
                    <div class="eyebrow">DETECTED LAND COVER</div>
                    <h1 style="font-size:42px;">{predicted_class}</h1>
                    <p>{DESCRIPTIONS[predicted_class]}</p>
                    <div style="font-size:34px;color:#59ccf5;font-weight:800;">
                        {confidence:.2f}%
                    </div>
                    <p>Prediction confidence</p>
                </div>
                """, unsafe_allow_html=True)

                st.write("")
                st.subheader("Prediction Probabilities")

                df = pd.DataFrame({
                    "Land Cover": CLASSES,
                    "Probability (%)": predictions * 100
                }).sort_values("Probability (%)", ascending=False)

                st.bar_chart(df.set_index("Land Cover"), horizontal=True)

                report = (
                    "EUROSAT CLASSIFICATION REPORT\n\n"
                    f"Image: {uploaded.name}\n"
                    f"Predicted class: {predicted_class}\n"
                    f"Confidence: {confidence:.2f}%\n\n"
                    f"{df.to_string(index=False)}"
                )

                st.download_button(
                    "Download Classification Report",
                    report,
                    file_name="eurosat_report.txt",
                    mime="text/plain"
                )

                st.caption(
                    "The model predicts one of its ten trained EuroSAT classes. "
                    "Images outside these categories may be misclassified."
                )

        except Exception as error:
            st.error(f"Image analysis failed: {error}")

# ---------------- EXPLORE DATA ----------------

elif page == "Explore Data":

    st.markdown('<div class="eyebrow">EUROSAT IMAGE COLLECTION</div>', unsafe_allow_html=True)
    st.title("Explore Satellite Images")
    st.write("Browse the ten land-cover categories in the EuroSAT RGB dataset.")

    available = [
        category for category in CLASSES
        if (BASE_DIR / category).is_dir()
    ]

    if not available:
        st.info("The full EuroSAT image collection is available in the local project dataset. Online deployment provides image classification using the trained CNN model.")
    else:

        selected = st.selectbox("Select a category", available)

        folder = BASE_DIR / selected

        images = sorted([
            f for f in folder.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        st.markdown(f"### {selected}")
        st.write(DESCRIPTIONS[selected])
        st.write(f"**Total images in this category:** {len(images)}")

        if images:
            columns = st.columns(3)

            for i, path in enumerate(images[:15]):
                with columns[i % 3]:
                    st.image(str(path), caption=path.name, width="stretch")

        st.markdown("---")
        st.subheader("Dataset Distribution")

        counts = {}

        for category in available:
            category_folder = BASE_DIR / category
            counts[category] = len([
                f for f in category_folder.iterdir()
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ])

        df = pd.DataFrame({
            "Category": list(counts.keys()),
            "Image Count": list(counts.values())
        })

        st.bar_chart(df.set_index("Category"), horizontal=True)

# ---------------- MODEL ----------------

elif page == "Model":

    st.markdown('<div class="eyebrow">DEEP LEARNING SYSTEM</div>', unsafe_allow_html=True)
    st.title("CNN Model Architecture")

    st.write("""
    This project uses a Convolutional Neural Network (CNN) to classify
    satellite images into ten land-cover categories. The model learns
    visual patterns such as texture, shapes and vegetation features.
    """)

    st.markdown("### Model Specifications")

    details = {
        "Dataset": "EuroSAT RGB",
        "Total images": "27,000",
        "Classification categories": "10",
        "Input dimensions": "64 × 64 × 3",
        "Model architecture": "Custom CNN",
        "Test accuracy": "88.47%",
        "Training images": "18,900",
        "Validation images": "4,050",
        "Testing images": "4,050"
    }

    for key, value in details.items():
        st.write(f"**{key}:** {value}")

    st.markdown("---")
    st.markdown("### Classification Workflow")

    workflow = [
        ("01", "Satellite Image Input", "Upload an RGB satellite image."),
        ("02", "Image Preprocessing", "Resize the image to 64 × 64 pixels."),
        ("03", "Feature Extraction", "CNN learns patterns from the image."),
        ("04", "Image Classification", "Predict one of the ten land-cover classes."),
        ("05", "Prediction Output", "Display the predicted class and probabilities.")
    ]

    for number, title, description in workflow:
        st.markdown(f"""
        <div class="feature-card">
            <div class="eyebrow">STEP {number}</div>
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Land Cover Classes")

    for category in CLASSES:
        with st.expander(category):
            st.write(DESCRIPTIONS[category])

# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
REMOTE SENSING IMAGE CLASSIFICATION USING EUROSAT
<br><br>
SATELLITE IMAGERY · DEEP LEARNING · LAND USE AND LAND COVER
</div>
""", unsafe_allow_html=True)
