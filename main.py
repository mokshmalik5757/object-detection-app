from flask import Flask, render_template, request, redirect, url_for, jsonify
from PIL import Image
from transformers import pipeline
import easyocr
import matplotlib.pyplot as plt

app = Flask(__name__)

def image_classification_single(image_path):
    two_results = []
    image = Image.open(image_path)
    classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
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
    apiResult = {"Message": "placeholder",
                 "Data": {"result": [{"tags": [], "text": []}]},
                 "Status": "placeholder"}
    if "image" in request.files and "locale" in request.form:
        images = request.files.getlist("image")  # Get a list of uploaded image files
        locales = request.form.get("locale")

        apiResult["Message"] = "Tags added successfully"

        for image in images:
            image.save("static/uploaded_image.jpg")  # Save each image file
            image_path = "static/uploaded_image.jpg"
            classification_results = image_classification_single(image_path)
            apiResult['Data']['result'][0]['tags'].append(classification_results)
            ocr_results = image_ocr(image_path, locales)
            apiResult['Data']['result'][0]['text'].append(ocr_results)

        apiResult["Status"] = "Ok"

        return jsonify(apiResult)
    else:
        apiResult["Message"] = "Some error occurred"
        apiResult["Data"]["result"][0]["tags"].append("No image found")
        apiResult['Data']['result'][0]['text'].append("No text found")
        apiResult["Status"] = "Error"
        return jsonify(apiResult)

if __name__ == "__main__":
    app.run(debug=True)


