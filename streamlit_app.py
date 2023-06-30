import streamlit as st
from PIL import Image
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


def perform_instance_segmentation(image):
    lvis_path = "LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(lvis_path))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(lvis_path)
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)

    try:
        outputs = predictor(image)

        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        result_image = out.get_image()[:, :, ::-1]
        return result_image
    except:
        return None

def main():
    st.set_page_config(page_title="Object Detection", page_icon="📷")
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                [data-testid="stForm"] {border: 0px}
                .css-10trblm.eqr7zpz0{
                text-align:center;
                }
                </style>
                """

    st.markdown(hide_st_style, unsafe_allow_html=True)
    st.title("Object Classification App"+ " "+ "🕵️‍♀️")


    with st.form(clear_on_submit=False, key="form_1"):
        global uploaded_images
        uploaded_images = st.file_uploader("Upload images", accept_multiple_files=True, type=["jpg", "jpeg", "png", "webp", "gif", "svg"])
        submit_button = st.form_submit_button("Submit")

    show_original = st.checkbox("Show Original Image")
    clear_button = st.button("Clear")

    if submit_button:
        with st.spinner("Processing..."):
            if uploaded_images:
                for uploaded_image in uploaded_images:
                    image = Image.open(uploaded_image)
                    image = np.array(image)

                    if image is not None:
                        result_image = perform_instance_segmentation(image)

                        if result_image is not None:
                            st.subheader("Processed Image")
                            st.image(result_image, caption="Instance Segmentation Result", use_column_width=True)

                            if show_original:
                                if st.session_state.get("result_image") is not None:
                                    st.subheader("Original Image")
                                    st.image(st.session_state["result_image"], caption="Original Image", use_column_width=True)
                                else:
                                    st.session_state["result_image"] = image
                                    st.subheader("Original Image")
                                    st.image(st.session_state["result_image"], caption="Original Image", use_column_width=True)
                            else:
                                st.session_state["result_image"] = None
            else:
                st.error("No image provided")

    if clear_button:
        st.session_state.clear()

if __name__ == "__main__":
    main()
