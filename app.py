
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(
    page_title="Object Detection",
    page_icon="🔍",
    layout="wide"
)

# Custom design
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f2edff, #e8e3ff);
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
}

h1, h2, h3, p, label {
    color: #30234b !important;
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #603a9b !important;
}

.subtitle {
    text-align: center;
    color: #65547d !important;
    font-size: 17px;
}

div[data-testid="stFileUploader"] {
    background-color: white;
    border: 2px dashed #a98bd5;
    border-radius: 15px;
    padding: 15px;
}

.stButton > button {
    background: #7044ad;
    color: white !important;
    border-radius: 12px;
    border: none;
    width: 100%;
    padding: 12px;
    font-weight: bold;
}

.stButton > button:hover {
    background: #4e2c80;
    color: white !important;
}

.result-box {
    background: white;
    border: 2px solid #c9b5ee;
    border-radius: 12px;
    padding: 18px;
    color: #30234b;
}

</style>
""", unsafe_allow_html=True)


# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


# Heading
st.markdown(
    '<h1 class="title">🔍 Object Detection System</h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Upload an image and let AI identify objects in it!</p>',
    unsafe_allow_html=True
)

st.write("")

st.markdown("### 📤 Upload Your Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("### 🖼️ Original Image")
    st.image(image, use_container_width=True)

    if st.button("✨ Detect Objects"):

        try:
            with st.spinner("AI is detecting objects..."):

                model = load_model()

                with tempfile.NamedTemporaryFile(
                    suffix=".jpg", delete=False
                ) as temp_file:

                    image.save(temp_file.name)
                    temp_path = temp_file.name

                results = model(temp_path)

                os.remove(temp_path)

                result = results[0]

                detected_image = result.plot()

                detected_image = Image.fromarray(
                    detected_image[:, :, ::-1]
                )

            st.success("Object detection completed!")

            st.markdown("### 🎯 Detection Result")

            st.image(
                detected_image,
                use_container_width=True
            )

            boxes = result.boxes

            st.markdown("### 📊 Detection Summary")

            if boxes is not None and len(boxes) > 0:

                st.metric(
                    "Total Objects Detected",
                    len(boxes)
                )

                object_counts = {}

                for box in boxes:
                    class_id = int(box.cls[0])
                    object_name = result.names[class_id]

                    object_counts[object_name] = (
                        object_counts.get(object_name, 0) + 1
                    )

                for name, count in object_counts.items():
                    st.write(f"**{name.capitalize()}:** {count}")

            else:
                st.info("No objects were detected in this image.")

        except Exception as error:
            st.error("Something went wrong.")
            st.write(str(error))

st.write("---")

st.markdown(
    "<p style='text-align:center;'>Made with Python, Streamlit and YOLO 💜</p>",
    unsafe_allow_html=True
)
