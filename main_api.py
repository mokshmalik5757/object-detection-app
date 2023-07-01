from flask import Flask, request
from PIL import Image
from transformers import pipeline

app = Flask(__name__)

def image_classification_single(image_path):
    two_results = []
    image = Image.open(image_path)
    classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
    result = classifier(image)
    for i in range(0, 2):
        two_results.append(result[i]["label"])
    return two_results

@app.route("/classify", methods=["POST"])
def classify_image():
    if "image" not in request.files:
        return {"error": "No image uploaded."}, 400

    image = request.files["image"]
    if image.filename == "":
        return {"error": "Empty image filename."}, 400

    image.save("uploaded_image.jpg")
    classification_results = image_classification_single("uploaded_image.jpg")

    return {"classification_results": classification_results}

if __name__ == "__main__":
    app.run(debug=True)
