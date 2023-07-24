import os
from pprint import pprint

os.environ['HUGGINGFACE_HUB_CACHE'] = "/mnt/pfs/share/pretrained_model/.cache/huggingface/hub"
os.environ['TRANSFORMERS_OFFLINE'] = "1"

# os.environ['http_proxy'] = "http://192.168.48.17:18000"
# os.environ['https_proxy'] = "http://192.168.48.17:18000"

from torch.hub import set_dir
set_dir("/mnt/pfs/share/pretrained_model/.cache/torch/hub")

# import sys
# sys.path.append('/mnt/pfs/share/yuanze/LAVIS/lavis')

from lavis import Blip2T5
from lavis.models import load_preprocess, load_model_and_preprocess
# # pip install accelerate
from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Block
import torch
from PIL import Image
from omegaconf import OmegaConf
from deepspeed import init_inference
from time import time
import argparse
from typing import List
from baidubce.services.bos.bos_client import BosClient
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.bce_client_configuration import BceClientConfiguration
import numpy as np

parser = argparse.ArgumentParser()
argparse.add_argument("--db_host", type=str, default="mysql-test", help="database host")
argparse.add_argument("--db_user", type=str, default="root", help="database user")
argparse.add_argument("--db_password", type=str, default="123456", help="database password")
argparse.add_argument("--db_database", type=str, default="test_caption", help="database name")
argparse.add_argument("--bos_access_key_id", type=str, default="ALTAKQu6BVji8Rg3GdGEHmYo25", help="bos access key id")
argparse.add_argument("--bos_secret_access_key", type=str, default="9eace348d4a3483c9efd48c819f34e93", help="bos secret access key")
argparse.add_argument("--bos_endpoint", type=str, default="bj.bcebos.com", help="bos endpoint")
argparse.add_argument("--blip2_generate_batchsize", type=int, default=5, help="blip2 t5 generate batchsize, must not be bigger than 6, otherwise OOM")
FLAGS = parser.parse_args()

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
BATCH_SIZE = FLAGS.blip2_generate_batchsize
DATABASE_CONFIG = {
    "host": FLAGS.db_host,
    "user": FLAGS.db_user,
    "password": FLAGS.db_password,
    "database": FLAGS.db_database
}
BOS_CONFIG = BceClientConfiguration(
    credentials=BceCredentials(
        access_key_id=FLAGS.bos_access_key_id,
        secret_access_key=FLAGS.bos_secret_access_key
    ),
    endpoint=FLAGS.bos_endpoint
)

def get_caption_blip2_model(try_parellel=False):
    if try_parellel:
        t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", 
                                                        #    device_map="auto", 
                                                        torch_dtype=torch.float16,
                                                        low_cpu_mem_usage=True,
                                                        )

        t5_model = init_inference(
            t5_model,
            mp_size=2,
            # 单纯fp16可能会丢失很多信息
            dtype=torch.bfloat16,
            # 让cross attention和residual能跨cuda通信
            injection_policy={T5Block: ('SelfAttention.o', 'EncDecAttention.o', 'DenseReluDense.wo')}
        )

        # load model
        model = Blip2T5.from_pretrained(model_type="pretrain_flant5xxl")
        # 如果不在外面实例化，不知道为什么会在两个cuda上都加载。
        model.t5_model = t5_model
        model.eval()

        # load preprocess
        cfg = OmegaConf.load(Blip2T5.default_config_path("pretrain_flant5xxl"))
        preprocess_cfg = cfg.preprocess
        vis_processors, _ = load_preprocess(preprocess_cfg)

        for name, parameter in model.named_parameters():
            if 't5_model' not in name:
                parameter.data = parameter.data.to(device)
    else:
        model, vis_processors, _ = load_model_and_preprocess(name='blip2_t5', model_type='pretrain_flant5xxl', is_eval=True, device=device)
        # import ipdb; ipdb.set_trace()
    return model, vis_processors

image_dict = {}
def get_image_from_path(image_path):
    global image_dict
    if not image_dict.__contains__(image_path):
        image_dict[image_path] = Image.open(image_path).convert('RGB')
    return image_dict[image_path]

def get_images_path(test=True, path=None) -> dict:
    """
    TODO:
    get imgs paths, imgs: (N, 22, 3, H, W)                                               
    where N denotes the total num of a batch images
    the amount may be enormous, so just keep the path instead of the raw imgs

    return: [(image_id, pic_absolute_path)]
    """
    ret_list = []
    import ipdb; ipdb.set_trace()
    if test:
        pic_folder = "/mnt/pfs/users/wangdehu/tmp_dir"
        sub_folder_list = os.listdir(pic_folder)
        for sub_folder in sub_folder_list:
            sub_folder = os.path.join(pic_folder, sub_folder)
            if not os.path.isdir(sub_folder):
                continue
            imgs = os.listdir(sub_folder)
            for img in imgs:
                if not img.endswith('.png'):
                    continue
                img = os.path.join(sub_folder, img)
                ret_list.append((img, img))
    elif path is not None and os.path.isdir(path):
        import json
        with open(os.path.join(path, "img_path.json"), "r") as f:
            # 每行是一个json object：
            # {"image_id": 0, "image_path": "path/to/image"}
            test_json = json.load(f)
        for item in test_json:
            ret_list.append((item["image_id"], item["image_path"]))
    return ret_list

time_generate1 = 0
time_generate2 = 0
model, vis_processors = None, None

def generate_coarse_captions(images_path_list) -> torch.Tensor():
    """
    TODO:
    generate coarse captions
    return list: [(img_id, image_path, coarse_captions[])]
    """
    global time_generate1, time_generate2, BATCH_SIZE, model, vis_processors
    if model is None:
        model, vis_processors = get_caption_blip2_model()
    prompt = "Question: what object is in this image? Answer:"
    full_prompt = "Question: what is the structure and geometry of this %s?"
    ret_list = []
    while len(images_path_list) != 0:
        if len(images_path_list) > BATCH_SIZE:
            tmp_img_path_list = images_path_list[:BATCH_SIZE]
            images_path_list = images_path_list[BATCH_SIZE:]
        else:
            tmp_img_path_list = images_path_list
            images_path_list = []
        image_list = [get_image_from_path(image_tuple[1]) for image_tuple in tmp_img_path_list]
        image_list = [vis_processors["eval"](image).unsqueeze(0).to(device) for image in image_list]
        batch_image = torch.cat(image_list, dim=0)
        tic = time()
        object_list = model.generate({"image": batch_image, "prompt": [prompt for _ in range(batch_image.shape[0])]}, max_length=5) # 如果不设置max_length，可能会由于prompt的token太长爆显存
        time_generate1 += time() - tic
        print(len(object_list))
        tic = time()
        coarse_caption_list = model.generate({"image": batch_image, "prompt": [full_prompt % object for object in object_list]}, use_nucleus_sampling=True, num_captions=5)
        time_generate2 += time() - tic
        for i in range(len(tmp_img_path_list)):
            ret_list.append((*tmp_img_path_list[i], coarse_caption_list[i * 5: (i + 1) * 5]))
        # what is coarse caption's length?
    return ret_list

import clip
from torch.nn import CosineSimilarity

cos, clip_model, clip_preprocess = None, None, None

def select_captions(coarse_caption_list) -> dict:
    global cos, clip_model, clip_preprocess
    if cos is None:
        # set up CLIP
        cos = CosineSimilarity(dim=1, eps=1e-6)
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, download_root='/mnt/pfs/share/pretrained_model/.cache')
        clip_model.eval()
    selected_caption_list = []
    for img_id, img_path, coarse_caption_list in coarse_caption_list:
        img = get_image_from_path(img_path)
        img_input = clip_preprocess(img).unsqueeze(0).to(device)
        text = clip.tokenize(coarse_caption_list).to(device)
        image_features = clip_model.encode_image(img_input)
        text_features = clip_model.encode_text(text)
        score = cos(image_features, text_features)
        selected_caption_list.append((img_id, img_path, coarse_caption_list[torch.argmax(score)]))
    return selected_caption_list

def extract_features(img_id_list, img_path_list, caption_list):
    BS_IMAGE, BS_TEXT = 8, 256
    # model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="coco", is_eval=True, device=device)
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)
    image_input_list = [vis_processors["eval"](get_image_from_path(image_path)).unsqueeze(0).to(device) for image_path in img_path_list]
    text_input_list = [txt_processors["eval"](caption) for caption in caption_list]
    image_feats = []
    text_feats = []
    sample = {}
    for i in range(0, len(img_id_list), BS_IMAGE):
        sample["image"] = torch.cat(image_input_list[i: min(len(img_id_list), i + BS_IMAGE)], dim=0)
        tmp = model.extract_features(sample, mode="image")['image_embeds_proj']
        image_feats.extend(tmp.chunk(tmp.size()[0], dim=0))
    sample = {}
    for i in range(0, len(img_id_list), BS_TEXT):
        sample["text_input"] = text_input_list[i: min(len(img_id_list), i + BS_TEXT)]
        tmp = model.extract_features(sample, mode="text")['text_embeds_proj'][:, 0, :]
        text_feats.extend(tmp.chunk(tmp.size()[0], dim=0))
    return image_feats, text_feats

import mysql.connector as connector
def write_database(store_caption_list):
    """
    store_caption_list: [(image_id, image_embedding_uri, image_caption, image_caption_embedding_uri)]
    """
    # write dict into mysql database
    # make new connection to database:
    global DATABASE_CONFIG
    database_config = DATABASE_CONFIG.copy()
    host = database_config.pop("host")
    user = database_config.pop("user")
    password = database_config.pop("password")
    database = database_config.pop("database")
    table = database_config.pop("table", "model_image_caption")
    conn = connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    conn.start_transaction()
    cursor = conn.cursor()
    sql = f"INSERT INTO `{database}`.`{table}` (image_id, cap_model_tag, extra, \
        image_embedding_bucket, image_embedding_uri, image_embedding_shape, image_caption, \
        image_caption_embedding_bucket, image_caption_embedding_uri, image_caption_embedding_shape) \
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    try:
        for image_id, image_embedding_uri, image_caption, image_caption_embedding_uri in store_caption_list:
            cursor.execute(sql % (image_id, "LAVIS-blip2_t5-xxl,CLIP-ViT-B/32", '', 'image_embedding_bucket',\
                                image_embedding_uri, '(32, 256)', image_caption, 'image_caption_embedding_bucket',\
                                image_caption_embedding_uri, '(256,)'))
        conn.commit()
        print("write to database successfully")
    except Exception as e:
        print("write to database failed, reason: ")
        print(e)
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

bos_client = None
def get_bos_client():
    global bos_client, BOS_CONFIG
    if bos_client is None:
        bos_client = BosClient(BOS_CONFIG)
    return bos_client

import hashlib
def calculate_md5_numpy_array(numpy_array):
    array_bytes = numpy_array.tobytes()
    md5_hash = hashlib.md5()
    md5_hash.update(array_bytes)
    md5_hexdigest = md5_hash.hexdigest()
    return md5_hexdigest

# ############################################################################
# # TODO:
# # Save image and caption embeddings to files
# ############################################################################
def save_to_file(embeddings:List[torch.Tensor], bucket_name, file_uri_list):
    """
    make sure the idempotence of saving progress
    """
    bos_client = get_bos_client()
    try: 
        bos_client.does_bucket_exist(bucket_name)
    except Exception as e:
        raise Exception("bucket不存在，请先创建bucket", e)
    for i, embed in enumerate(embeddings):
        embed_numpy = embed.numpy()
        np.save('temp', embed_numpy)
        bos_client.put_object_from_file(bucket_name, file_uri_list[i], 'temp.npy')
    if os.path.exists('temp.npy'):
        os.remove('temp.npy')

# ############################################################################
# # TODO:
# # send message to mq
# ############################################################################
def send_message_to_mq():
    pass

if __name__ == "__main__":
    images_path_list = get_images_path(False, '/mnt/pfs/share/yuanze/LAVIS')
    coarse_caption_list = generate_coarse_captions(images_path_list)
    model, vis_processors = None, None
    print(f'time_generate1 is {time_generate1}')
    print(f'time_generate2 is {time_generate2}')
    # pprint(coarse_caption_list)
    tic = time()
    selected_caption_list = select_captions(coarse_caption_list)
    print(f'time_select is {time() - tic}')
    tic = time()
    image_feats, text_feats = extract_features(*zip(*selected_caption_list)) # image_feats: [(32, 256) * bs], text_feats: [(256,) * bs]
    print(f'time_extract is {time() - tic}')
    # bs = image_feats.shape[0]
    # sim = image_feats @ text_feats.T # (bs, 32, bs)
    # sim = sim.reshape(bs, -1)
    # cos_sim = sim.max(1)
    # FIXME: 这里暂时用image_path作为桶内的uri，之后改为上游任务传来的uri
    image_uri_list = save_to_file(image_feats, 'image_embedding_bucket', zip(*selected_caption_list)[1]) 
    caption_uri_list = save_to_file(text_feats, 'image_caption_embedding_bucket', zip(*selected_caption_list)[1])
    print('embedding存bos成功')
    write_database([(image_id, image_uri, caption, caption_uri) for \
                    (image_id, _, caption), image_uri, caption_uri in zip(selected_caption_list, image_uri_list, caption_uri_list)])
    send_message_to_mq()

# batchsize 2:
# time_generate1 is 12.068463802337646
# time_generate2 is 155.86847376823425
# time_select is 4.8412981033325195
# time_extract is 20.37248706817627