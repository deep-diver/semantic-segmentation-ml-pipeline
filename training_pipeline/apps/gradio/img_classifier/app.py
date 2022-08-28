import gradio as gr
from huggingface_hub import snapshot_download
from tensorflow import keras
from tensorflow.keras.applications import resnet50

model_path = snapshot_download(repo_id="$MODEL_REPO_ID", revision="$MODEL_VERSION")
model = keras.models.load_model(model_path)
labels = []

with open(r"labels.txt", "r") as fp:
    for line in fp:
        labels.append(line[:-1])


def classify_image(inp):
    inp = inp.reshape((-1, 224, 224, 3))
    inp = resnet50.preprocess_input(inp)

    prediction = model.predict(inp).flatten()
    confidences = {labels[i]: float(prediction[i]) for i in range(10)}
    return confidences


iface = gr.Interface(
    fn=classify_image,
    inputs=gr.inputs.Image(shape=(224, 224)),
    outputs=gr.outputs.Label(num_top_classes=3),
)
iface.launch()
