from typing import Any, Dict

import os
from absl import logging

import tensorflow as tf

from huggingface_hub import Repository
from huggingface_hub import HfApi
from requests.exceptions import HTTPError

from pipeline.components.pusher.HFModelPusher import constants


def release_model_for_hf_model(
    model_path: str,
    model_version_name: str,
    hf_release_args: Dict[str, Any],
):
    access_token = hf_release_args[constants.ACCESS_TOKEN_KEY]

    username = hf_release_args[constants.USERNAME_KEY]
    reponame = hf_release_args[constants.REPONAME_KEY]

    repo_type = "model"

    repo_id = f"{username}/{reponame}-{repo_type}"
    repo_url_prefix = "https://huggingface.co"
    repo_url = f"{repo_url_prefix}/{repo_id}"

    logging.warning(f"model_path: {model_path}")
    logging.warning("download pushed model")

    try:
        HfApi().create_repo(token=access_token, repo_id=repo_id, repo_type=repo_type)
    except HTTPError:
        logging.warning(f"{repo_id}-model repository may already exist")
        logging.warning("this is expected behaviour if you overwrite with a new branch")

    repository = Repository(
        local_dir="hf-model-repo/", clone_from=repo_url, use_auth_token=access_token
    )
    repository.git_checkout(revision=model_version_name, create_branch_ok=True)

    root_dir = "hf-model-repo"
    blobnames = tf.io.gfile.listdir(model_path)

    for blobname in blobnames:
        blob = f"{model_path}/{blobname}"

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
    repository.git_commit(commit_message="updload new version of the model")
    repository.git_push(upstream=f"origin {model_version_name}")

    return (repo_id, repo_url)
