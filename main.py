from flask import Flask, render_template, request, redirect, url_for, jsonify
from PIL import Image
from transformers import ViTImageProcessor
from optimum.pipelines import pipeline
import easyocr
import matplotlib.pyplot as plt

app = Flask(__name__)

feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')


def image_classification_single(image_path):
    two_results = []
    image = Image.open(image_path)
    classifier = pipeline("image-classification", model="google/vit-base-patch16-224",  accelerator = "bettertransformer", feature_extractor = feature_extractor)
    result = classifier(image)
    for i in range(0, 2):
        two_results.append(result[i]["label"])
    return two_results

def image_ocr(image_path, lang):
    lang_list = lang.strip("()").split(",")
    obj = easyocr.Reader(lang_list=lang_list, gpu= False)
    ocr_text = []
    img = plt.imread(image_path)
    result = obj.readtext(img)
    for index in range(len(result)):
        ocr_text.append(result[index][1])
    return ocr_text

def detect_text(path, locale = "en"):
    """Detects text in the file."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image, image_context = {"language_hints": locale})
    texts = response.text_annotations

    if texts:
        return texts[0].description

    if response.error.message:
        raise Exception(
            "{}\n".format(response.error.message)
        )


def detect_labels(path):
    """Detects labels in the file."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.label_detection(image=image)
    labels = response.label_annotations

    tags = []

    for label in labels:
        tags.append(label.description)

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    return tags

@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)
        image = request.files["image"]
        if image.filename == "":
            return redirect(request.url)
        image.save("static/uploaded_image.jpg")
        return redirect(url_for("results"))
    return render_template("upload.html")


@app.route("/results")
def results():
    image_path = "static/uploaded_image.jpg"
    classification_results = image_classification_single(image_path)
    return render_template("final.html", image_path=image_path, classification_results=classification_results)

@app.route("/api/v1/model", methods = ["POST"])
def callModel():
    locales = None
    apiResult = {"Message": "placeholder",
                 "Data": {"result": [{"tags": [], "text": []}]},
                 "Status": "placeholder"}
    if "image" in request.files:
        images = request.files.getlist("image")# Get a list of uploaded image files
        locales = request.form.get("locale")
        # Remove the "global" keyword from here

        apiResult["Message"] = "Tags added successfully"

        for image in images:
            image.save("static/uploaded_image.jpg")  # Save each image file
            image_path = "static/uploaded_image.jpg"
            classification_results = image_classification_single(image_path)
            apiResult['Data']['result'][0]['tags'].append(classification_results)
            if locales is not None:
                ocr_results = image_ocr(image_path, locales)
                apiResult['Data']['result'][0]['text'].append(ocr_results)

        apiResult["Status"] = "Ok"

        return jsonify(apiResult)
    else:
        apiResult["Message"] = "Some error occurred"
        apiResult["Data"]["result"][0]["tags"].append("No image found")
        if locales is not None:
            apiResult['Data']['result'][0]['text'].append("No text found")
        apiResult["Status"] = "Error"
        return jsonify(apiResult)

@app.route("/api/v1/google-model", methods = ["POST"])
def google_ocr():
    locales = None

    apiResult = {"Message": "placeholder",
                 "Data": {"result": [{"tags": [], "text": []}]},
                 "Status": "placeholder"}
    if "image" in request.files:
        images = request.files.getlist("image")  # Get a list of uploaded image files
        locales = request.form.get("locale")

        apiResult["Message"] = "Tags added successfully"

        for image in images:
            image.save("static/uploaded_image.jpg")  # Save each image file
            image_path = "static/uploaded_image.jpg"
            classification_results = image_classification_single(image_path)
            apiResult['Data']['result'][0]['tags'].append(classification_results)
            if locales is not None:
                ocr_results = detect_text(image_path, locales)
                apiResult['Data']['result'][0]['text'].append(ocr_results)

        apiResult["Status"] = "Ok"

        return jsonify(apiResult)
    else:
        apiResult["Message"] = "Some error occurred"
        apiResult["Data"]["result"][0]["tags"].append("No image found")
        if locales is not None:
            apiResult['Data']['result'][0]['text'].append("No text found")
        apiResult["Status"] = "Error"
        return jsonify(apiResult)


@app.route("/api/v1/google-model-all", methods = ["POST"])
def google_text_and_ocr():
    locales = None

    apiResult = {"Message": "placeholder",
                 "Data": {"result": [{"tags": [], "text": []}]},
                 "Status": "placeholder"}
    if "image" in request.files:
        images = request.files.getlist("image")  # Get a list of uploaded image files
        locales = request.form.get("locale")

        apiResult["Message"] = "Tags added successfully"

        for image in images:
            image.save("static/uploaded_image.jpg")  # Save each image file
            image_path = "static/uploaded_image.jpg"
            classification_results = detect_labels(image_path)
            apiResult['Data']['result'][0]['tags'].append(classification_results)
            if locales is not None:
                ocr_results = detect_text(image_path, locales)
                apiResult['Data']['result'][0]['text'].append(ocr_results)

        apiResult["Status"] = "Ok"

        return jsonify(apiResult)
    else:
        apiResult["Message"] = "Some error occurred"
        apiResult["Data"]["result"][0]["tags"].append("No image found")
        if locales is not None:
            apiResult['Data']['result'][0]['text'].append("No text found")
        apiResult["Status"] = "Error"
        return jsonify(apiResult)


if __name__ == "__main__":
    app.run(debug=True)
