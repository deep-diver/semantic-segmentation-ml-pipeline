import absl


def INFO(text: str):
    absl.logging.info(text)


def transformed_name(key: str) -> str:
    return key + "_xf"
