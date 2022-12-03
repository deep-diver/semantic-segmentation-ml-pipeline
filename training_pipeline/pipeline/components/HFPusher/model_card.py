from huggingface_hub import ModelCard, ModelCardData


def create_card(card_data, template_path, model_metadata, 
                placeholders, **template_kwargs):
    """Creates model card.

    Args:
        model_metadata (dict): Dict of card metadata.
        see here for what you can pass to the metadata section:
        https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md

    Returns:
        Model card (ModelCard): instantiated and filled ModelCard.
    """
    if model_metadata is None:
        model_metadata = {}
    model_metadata["library_name"] = "vertex"
    if model_metadata["tags"]:
        model_metadata["tags"].append("keras")
    else:
        model_metadata["tags"] = "keras"
    card_data = ModelCardData(**{model_metadata})
    model_card = ModelCard.from_template(card_data,
                                        template_path,
                                        **template_kwargs)
    return model_card
    
