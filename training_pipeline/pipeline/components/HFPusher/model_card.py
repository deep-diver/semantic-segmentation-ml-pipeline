from huggingface_hub import ModelCard, ModelCardData


def create_card(template_path, model_metadata, **template_kwargs):
    """Creates model card.

    Args:
        template_path (str): Path to the jinja template model is based on.
        model_metadata (dict): Dict of card metadata.
        Refer to the link to know what you can pass to the metadata section:
        https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md

    Returns:
        Model card (ModelCard): instantiated and filled ModelCard.
    """
    if model_metadata is None:
        model_metadata = {}
    model_metadata["library_name"] = "tfx"
    if "tags" in model_metadata:
        model_metadata["tags"].append("keras")
    else:
        model_metadata["tags"] = "keras"
    card_data = ModelCardData(**{v: k for k, v in model_metadata.items()})
    model_card = ModelCard.from_template(card_data,
                                        template_path,
                                        **template_kwargs)
    return model_card
    
