from typing import Any, Dict

import os
import tarfile
from absl import logging

from github import Github
import tensorflow as tf

from pipeline.components.pusher.GHReleasePusher import constants


def release_model_for_github(
    model_path: str,
    model_version_name: str,
    gh_release_args: Dict[str, Any],
) -> str:
    access_token = gh_release_args[constants.ACCESS_TOKEN_KEY]

    username = gh_release_args[constants.USERNAME_KEY]
    reponame = gh_release_args[constants.REPONAME_KEY]
    repo_uri = f"{username}/{reponame}"

    branch_name = gh_release_args[constants.BRANCH_KEY]

    model_archive = gh_release_args[constants.ASSETNAME_KEY]

    gh = Github(access_token)
    repo = gh.get_repo(repo_uri)
    branch = repo.get_branch(branch_name)

    release = repo.create_git_release(
        model_version_name,
        f"model release {model_version_name}",
        "",
        draft=False,
        prerelease=False,
        target_commitish=branch,
    )

    logging.warning(f"model_path: {model_path}")
    if model_path.startswith("gs://"):
        logging.warning("download pushed model")
        root_dir = "saved_model"
        os.mkdir(root_dir)

        blobnames = tf.io.gfile.listdir(model_path)

        for blobname in blobnames:
            blob = f"{model_path}/{blobname}"

            if tf.io.gfile.isdir(blob):
                sub_dir = f"{root_dir}/{blobname}"
                os.mkdir(sub_dir)

                sub_blobnames = tf.io.gfile.listdir(blob)
                for sub_blobname in sub_blobnames:
                    sub_blob = f"{blob}{sub_blobname}"

                    logging.warning(f"{sub_dir}/{sub_blobname}")
                    tf.io.gfile.copy(sub_blob, f"{sub_dir}{sub_blobname}")
            else:
                logging.warning(f"{root_dir}/{blobname}")
                tf.io.gfile.copy(blob, f"{root_dir}/{blobname}")

        model_path = root_dir

    logging.warning("compress the model")
    with tarfile.open(model_archive, "w:gz") as tar:
        tar.add(model_path)

    logging.warning("upload the model")
    release.upload_asset(model_archive, name=model_archive)
    return f"https://github.com/{username}/{reponame}/releases/tag/{model_version_name}"
