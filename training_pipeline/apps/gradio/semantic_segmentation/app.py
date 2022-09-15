import gradio as gr
import numpy as np
import tensorflow as tf
from huggingface_hub import from_pretrained_keras
from PIL import Image

MODEL_CKPT = "$MODEL_REPO_ID@$MODEL_VERSION"
MODEL = from_pretrained_keras(MODEL_CKPT)


RESOLTUION = 128

ADE_PALETTE = []
with open(r"./palette.txt", "r") as fp:
    for line in fp:
        tmp_list = list(map(int, line[:-1].strip("][").split(", ")))
        ADE_PALETTE.append(tmp_list)


def preprocess_input(image: Image) -> tf.Tensor:
    image = np.array(image)
    image = tf.convert_to_tensor(image)

    image = tf.image.resize(image, (RESOLTUION, RESOLTUION))
    image = image / 255

    return tf.expand_dims(image, 0)


# The below utility get_seg_overlay() are from:
# https://github.com/deep-diver/semantic-segmentation-ml-pipeline/blob/main/notebooks/inference_from_SavedModel.ipynb


def get_seg_overlay(image, seg):
    color_seg = np.zeros(
        (seg.shape[0], seg.shape[1], 3), dtype=np.uint8
    )  # height, width, 3
    palette = np.array(ADE_PALETTE)

    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    # Show image + mask
    img = np.array(image) * 0.5 + color_seg * 0.5

    img *= 255
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return img


def run_model(image: Image) -> tf.Tensor:
    preprocessed_image = preprocess_input(image)
    prediction = MODEL.predict(preprocessed_image)

    seg_mask = tf.math.argmax(prediction, -1)
    seg_mask = tf.squeeze(seg_mask)
    return seg_mask


def get_predictions(image: Image):
    predicted_segmentation_mask = run_model(image)
    preprocessed_image = preprocess_input(image)
    preprocessed_image = tf.squeeze(preprocessed_image, 0)

    pred_img = get_seg_overlay(
        preprocessed_image.numpy(), predicted_segmentation_mask.numpy()
    )
    return Image.fromarray(pred_img)


title = "Simple demo for a semantic segmentation model trained on the Sidewalks dataset."

description = """

Note that the outputs obtained in this demo won't be state-of-the-art. The underlying project has a different objective focusing more on the ops side of
deploying a semantic segmentation model. For more details, check out the repository: https://github.com/deep-diver/semantic-segmentation-ml-pipeline/.

"""

demo = gr.Interface(
    get_predictions,
    gr.inputs.Image(type="pil"),
    "pil",
    allow_flagging="never",
    title=title,
    description=description,
    examples=[["test-image.jpg"]],
)

demo.launch()
