from flask import Flask, request, render_template_string
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)

# Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded successfully!")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Image-Based RAG Question Answering System</title>
</head>
<body>
    <h1>Image-Based RAG Question Answering System</h1>

    <form method="POST" enctype="multipart/form-data">
        <p>Upload an image:</p>
        <input type="file" name="image" accept="image/*" required>

        <p>Ask a question:</p>
        <input type="text" name="question" required>

        <button type="submit">Get Answer</button>
    </form>

    {% if extracted_text %}
        <h2>Extracted Text</h2>
        <pre>{{ extracted_text }}</pre>
    {% endif %}

    {% if answer %}
        <h2>Answer</h2>
        <p>{{ answer }}</p>
    {% endif %}
</body>
</html>
"""

# Split extracted text into sentences
def split_text(text):
    return [
        sentence.strip()
        for sentence in text.replace("\n", " ").split(".")
        if sentence.strip()
    ]

# Find the most relevant sentence
def find_answer(question, chunks):
    if not chunks:
        return "No readable text found."

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


@app.route("/", methods=["GET", "POST"])
def home():
    extracted_text = ""
    answer = ""

    if request.method == "POST":
        image_file = request.files.get("image")
        question = request.form.get("question", "").strip()

        if image_file and question:
            try:
                image = Image.open(image_file.stream).convert("RGB")

                # Extract text using OCR
                extracted_text = pytesseract.image_to_string(image)

                # Split text into chunks
                chunks = split_text(extracted_text)

                # Retrieve answer
                answer = find_answer(question, chunks)

            except Exception as error:
                answer = f"Error: {error}"

    return render_template_string(
        HTML,
        extracted_text=extracted_text,
        answer=answer
    )


if __name__ == "__main__":
    app.run(debug=True)
