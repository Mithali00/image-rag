import os
import shutil

import streamlit as st
import numpy as np
import pytesseract

from PIL import Image
from sentence_transformers import SentenceTransformer


# Configure the Streamlit page
st.set_page_config(
    page_title="Image-Based RAG Question Answering System",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Image-Based RAG Question Answering System")
st.write(
    "Upload an image containing text, ask a question, "
    "and find the most relevant answer."
)


# Configure Tesseract for Windows
if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )


# Load the embedding model once and reuse it
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# Split extracted text into sentences
def split_text(text):
    return [
        sentence.strip()
        for sentence in text.replace("\n", " ").split(".")
        if sentence.strip()
    ]


# Find the most relevant sentence
def find_answer(question, chunks, model):
    if not chunks:
        return "No readable text was found in the image."

    question_embedding = model.encode(question)
    chunk_embeddings = model.encode(chunks)

    similarities = np.dot(
        chunk_embeddings,
        question_embedding
    ) / (
        np.linalg.norm(chunk_embeddings, axis=1)
        * np.linalg.norm(question_embedding)
        + 1e-10
    )

    best_index = int(np.argmax(similarities))

    return chunks[best_index]


# Image upload
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg", "bmp", "tiff"]
)

# Question input
question = st.text_input(
    "Ask a question about the image"
)

# Process the image
if st.button("Get Answer"):
    if uploaded_file is None:
        st.warning("Please upload an image first.")

    elif not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            # Check that Tesseract is available
            if (
                os.name != "nt"
                and shutil.which("tesseract") is None
            ):
                st.error(
                    "Tesseract OCR is not installed on the server. "
                    "Please check packages.txt."
                )
                st.stop()

            # Display the uploaded image
            image = Image.open(uploaded_file).convert("RGB")

            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)

            # Extract text using OCR
            with st.spinner("Extracting text from the image..."):
                extracted_text = pytesseract.image_to_string(image)

            st.subheader("Extracted Text")

            if extracted_text.strip():
                st.text_area(
                    "Text detected in your image",
                    extracted_text,
                    height=200
                )
            else:
                st.warning(
                    "No readable text was found in this image."
                )

            # Retrieve the relevant answer
            chunks = split_text(extracted_text)

            if chunks:
                with st.spinner("Loading AI model and finding the answer..."):
                    model = load_model()
                    answer = find_answer(
                        question.strip(),
                        chunks,
                        model
                    )

                st.subheader("💡 Answer")
                st.success(answer)

            else:
                st.info(
                    "Try uploading a clearer image containing readable text."
                )

        except Exception as error:
            st.error(
                "An error occurred while processing the image."
            )
            st.exception(error)
