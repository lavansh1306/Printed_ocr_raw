from flask import Flask, request, jsonify, render_template
import paddle
from paddleocr import PaddleOCR, draw_ocr
from pdf2image import convert_from_path
import cv2
import os
import matplotlib.pyplot as plt

# Initialize the Flask app
app = Flask(__name__)

# Initialize the OCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def process_image(image_path):
    """Perform OCR on a single image."""
    result = ocr.ocr(image_path, cls=True)
    extracted_text = []

    for line in result[0]:
        text = line[1][0]
        confidence = line[0]
        extracted_text.append(f"{text} (Confidence: {confidence})")
    
    return extracted_text

def process_pdf(pdf_path):
    """Perform OCR on each page of a PDF."""
    pages = convert_from_path(pdf_path, 300)
    all_text = []

    for page_num, page in enumerate(pages, start=1):
        image_path = f"page_{page_num}.png"
        page.save(image_path, 'PNG')
        page_text = process_image(image_path)
        all_text.extend(page_text)
        os.remove(image_path)  # Clean up

    return all_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)

    file_path = os.path.join('uploads', file.filename)

    file.save(file_path)

    ext = os.path.splitext(file.filename)[1].lower()
    output_file = os.path.join('uploads', "extracted_text.txt")

    try:
        if ext == '.pdf':
            text = process_pdf(file_path)
        elif ext in ['.jpg', '.jpeg', '.png']:
            text = process_image(file_path)
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        # Save extracted text to a file
        with open(output_file, "w") as f:
            for line in text:
                f.write(f"{line}\n")

        return jsonify({"message": "OCR successful", "text": text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
