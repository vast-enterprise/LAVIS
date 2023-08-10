import torch
from typing import List
import os
import numpy as np
from json import loads

from baidubce.services.bos.bos_client import BosClient
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.bce_client_configuration import BceClientConfiguration

BOS_ACCESS_KEY_ID = os.environ.get("BOS_ACCESS_KEY_ID")
BOS_SECRET_ACCESS_KEY = os.environ.get("BOS_SECRET_ACCESS_KEY")
BOS_ENDPOINT = os.environ.get("BOS_ENDPOINT")
BOS_TASK_BUCKET = os.environ.get("BOS_TASK_BUCKET", "conductor-task")

error_list = []
if BOS_ACCESS_KEY_ID is None:
    error_list.append("BOS_ACCESS_KEY_ID")
if BOS_SECRET_ACCESS_KEY is None:
    error_list.append("BOS_SECRET_ACCESS_KEY")
if BOS_ENDPOINT is None:
    error_list.append("BOS_ENDPOINT")
assert len(error_list) == 0, "Please set the following environment variables: {}".format(", ".join(error_list))

BOS_CONFIG = BceClientConfiguration(
    credentials=BceCredentials(
        access_key_id=BOS_ACCESS_KEY_ID,
        secret_access_key=BOS_SECRET_ACCESS_KEY
    ),
    endpoint=BOS_ENDPOINT
)


bos_client = None
def get_bos_client():
    global bos_client, BOS_CONFIG
    if bos_client is None:
        bos_client = BosClient(BOS_CONFIG)
    return bos_client


def save_to_bos(embeddings:List[torch.Tensor], bucket_name, uri_list):
    """
    make sure the idempotence of saving progress
    """
    bos_client = get_bos_client()
    try: 
        bos_client.does_bucket_exist(bucket_name)
    except Exception as e:
        # FIXME: 使用前得手动创建bucket
        raise Exception("bucket不存在，请先创建bucket", e)
    try:
        for i, embed in enumerate(embeddings):
            embed_numpy = embed.cpu().numpy()
            np.save('temp', embed_numpy)
            bos_client.put_object_from_file(bucket_name, uri_list[i], 'temp.npy')
    finally:
        if os.path.exists('temp.npy'):
            os.remove('temp.npy')

def get_task_from_bos(uri: str):
    bos_client = get_bos_client()
    result = bos_client.get_object_as_string(BOS_TASK_BUCKET, uri).decode("utf-8")
    if result:
        for line in result.strip(' \n').split("\n"):
            yield loads(line)