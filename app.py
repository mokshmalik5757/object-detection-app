from dotenv import load_dotenv
import os
from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from pyngrok import ngrok
import base64


app = Flask(__name__)
env_config = os.getenv("PROD_APP_SETTINGS", "config.DevelopmentConfig")
app.config.from_object(env_config)

app.config['UPLOAD_FOLDER'] = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def perform_instance_segmentation(image_path):
    lvis_path = "LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(lvis_path))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(lvis_path)
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)
    im = cv2.imread(image_path)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result_path = os.path.join("uploads", 'result.jpg')
    cv2.imwrite(result_path, out.get_image()[:, :, ::-1])
    return result_path


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('file')  # Get a list of uploaded files
        result_base64_list = []  # List to store the base64-encoded result images
        original_base64_list = []  # List to store the base64-encoded original images
        show_result = False  # Flag to determine if result images should be displayed

        for file in files:
            if file.filename == '':
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                result_path = perform_instance_segmentation(file_path)

                with open(result_path, 'rb') as f:
                    result_data = f.read()

                result_base64 = base64.b64encode(result_data).decode('utf-8')
                result_base64_list.append(result_base64)

                with open(file_path, 'rb') as f:
                    original_data = f.read()

                original_base64 = base64.b64encode(original_data).decode('utf-8')
                original_base64_list.append(original_base64)

        show_result = len(result_base64_list) > 0  # Check if there are any result images

        return render_template('result.html', original_base64_list=original_base64_list, result_base64_list=result_base64_list, show_result=show_result)

    return render_template('index.html')


if __name__ == '__main__':
    # Start ngrok tunnel
    public_url = ngrok.connect(port=5000).public_url
    print(f"Running Flask app on {public_url}")

    # Run the Flask app
    app.run()

