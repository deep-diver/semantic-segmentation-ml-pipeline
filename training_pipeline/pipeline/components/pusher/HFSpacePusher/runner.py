from typing import Any, Dict

import os
from absl import logging

import tensorflow as tf

from huggingface_hub import HfApi
from huggingface_hub import Repository
from requests.exceptions import HTTPError

from pipeline.components.pusher.HFSpacePusher import constants


def release_model_for_hf_space(
    model_repo_id: str,
    model_repo_url: str,
    model_version: str,
    hf_release_args: Dict[str, Any],
):
    access_token = hf_release_args[constants.ACCESS_TOKEN_KEY]

    username = hf_release_args[constants.USERNAME_KEY]
    reponame = hf_release_args[constants.REPONAME_KEY]

    repo_type = "space"

    repo_url_prefix = "https://huggingface.co"
    space_repo_id = f"{username}/{reponame}-{repo_type}"
    space_repo_url = f"{repo_url_prefix}/{space_repo_id}"

    app_path = hf_release_args[constants.APP_PATH_KEY]
    app_path = app_path.replace(".", "/")

    placeholder_to_replace = {
        hf_release_args[constants.MODEL_HUB_REPO_PLACEHOLDER_KEY]: model_repo_id,
        hf_release_args[constants.MODEL_HUB_URL_PLACEHOLDER_KEY]: model_repo_url,
        hf_release_args[constants.MODEL_VERSION_PLACEHOLDER_KEY]: model_version,
    }

    replace_placeholders_in_files(app_path, placeholder_to_replace)

    try:
        HfApi().create_repo(
            token=access_token,
            repo_id=space_repo_id,
            repo_type=repo_type,
            space_sdk="gradio",
        )
    except HTTPError:
        logging.warning(f"{space_repo_id} repository may already exist")
        logging.warning("this is expected behaviour if you overwrite with a new branch")

    repository = Repository(
        local_dir="hf-space-repo/",
        clone_from=space_repo_url,
        use_auth_token=access_token,
        repo_type=repo_type,
    )
    # repository.git_checkout(revision=model_version, create_branch_ok=True)

    root_dir = "hf-space-repo"
    blobnames = tf.io.gfile.listdir(app_path)

    for blobname in blobnames:
        blob = f"{app_path}/{blobname}"

        if tf.io.gfile.isdir(blob):
            sub_dir = f"{root_dir}/{blobname}"

            try:
                os.mkdir(sub_dir)
            except FileExistsError as error:
                logging.warning(
                    f"folder {sub_dir} already exists, this is expected behaviour if you overwrite with a new branch"
                )
                logging.warning(error)

            sub_blobnames = tf.io.gfile.listdir(blob)
            for sub_blobname in sub_blobnames:
                sub_blob = f"{blob}{sub_blobname}"

                logging.warning(f"{sub_dir}/{sub_blobname}")
                tf.io.gfile.copy(sub_blob, f"{sub_dir}{sub_blobname}", overwrite=True)
        else:
            logging.warning(f"{root_dir}/{blobname}")
            tf.io.gfile.copy(blob, f"{root_dir}/{blobname}", overwrite=True)

    repository.git_add(pattern=".", auto_lfs_track=True)
    repository.git_commit(commit_message=f"updload {model_version} model")
    repository.git_push(upstream="origin main")

    return (space_repo_id, space_repo_url)


def replace_placeholders_in_files(
    root_dir: str, placeholder_to_replace: Dict[str, str]
):
    files = tf.io.gfile.listdir(root_dir)
    for file in files:
        path = tf.io.gfile.join(root_dir, file)

        if tf.io.gfile.isdir(path):
            replace_placeholders_in_files(path, placeholder_to_replace)
        else:
            replace_placeholders_in_file(path, placeholder_to_replace)


def replace_placeholders_in_file(filepath: str, placeholder_to_replace: Dict[str, str]):
    with tf.io.gfile.GFile(filepath, "r") as f:
        source_code = f.read()

    for placeholder in placeholder_to_replace:
        source_code = source_code.replace(
            placeholder, placeholder_to_replace[placeholder]
        )

    with tf.io.gfile.GFile(filepath, "w") as f:
        f.write(source_code)
