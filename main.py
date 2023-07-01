from flask import Flask, render_template, request, redirect, url_for, jsonify
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
    apiResult = {"Message": [],
                 "Data": {"result": [{"tags": []}]},
                 "Status": []}
    if "image" in request.files:
        images = request.files.getlist("image")  # Get a list of uploaded image files


        for image in images:
            image.save("static/uploaded_image.jpg")  # Save each image file
            image_path = "static/uploaded_image.jpg"
            classification_results = image_classification_single(image_path)
            apiResult["Message"].append("Tags added successfully")
            apiResult['Data']['result'][0]['tags'].append(classification_results)
            apiResult["Status"].append("Ok")

        return jsonify(apiResult)
    else:
        apiResult["Message"].append("Some error occourred")
        apiResult["Data"]["result"].append("No image found")
        apiResult["Status"].append("Error")
        return jsonify(apiResult)


if __name__ == "__main__":
    app.run(debug=True)


