---
# For reference on model card metadata, refer to: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
{{card_data}}
---

# Model Card for {{ model_id | default("Model ID", true) }}

# Model Details

## Model Description

This is a {{ task | default("segmentation")}} model trained using Tensorflow Extended.


# Training Details

## Training Data

{{ training_data | default("[More Information Needed]", true)}}

## Training Procedure

# Evaluation

## Results

{{ results | default("[More Information Needed]", true)}}
