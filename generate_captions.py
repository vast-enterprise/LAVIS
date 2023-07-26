import os

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
parser.add_argument("--db_host", type=str, default="vast-db.rdsm7toamklwzxh.rds.bj.baidubce.com", help="database host")
parser.add_argument("--db_user", type=str, default="root", help="database user")
parser.add_argument("--db_port", type=int, default=28564, help="database port")
parser.add_argument("--db_password", type=str, default="VcUFNW5Kg2", help="database password")
parser.add_argument("--db_database", type=str, default="vast_data_platform_test", help="database name")
parser.add_argument("--bos_access_key_id", type=str, default="ALTAKQu6BVji8Rg3GdGEHmYo25", help="bos access key id")
parser.add_argument("--bos_secret_access_key", type=str, default="9eace348d4a3483c9efd48c819f34e93", help="bos secret access key")
parser.add_argument("--bos_endpoint", type=str, default="bj.bcebos.com", help="bos endpoint")
parser.add_argument("--bos_parent_folder", type=str, default="zero_render", help="bos parent folder which denotes the data source")
# parser.add_argument("--mq_xxx", type=str, default="bj.bcebos.com", help="bos endpoint")
parser.add_argument("--image_json_path", type=str, default="/mnt/pfs/share/yuanze/LAVIS/img_path.json", help="path to json file which contains [{img_id, path}]")
parser.add_argument("--blip2_generate_batchsize", type=int, default=5, help="blip2 t5 generate batchsize, better not be bigger than 6, otherwise may OOM")
parser.add_argument("--blip2_generate_num_captions", type=int, default=5, help="blip2 t5 generate num_captions")
FLAGS = parser.parse_args()

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
BATCH_SIZE = FLAGS.blip2_generate_batchsize
NUM_CAPTIONS = FLAGS.blip2_generate_num_captions
IMAGE_JSON_PATH = FLAGS.image_json_path
DATABASE_CONFIG = {
    "host": FLAGS.db_host,
    "user": FLAGS.db_user,
    "port": FLAGS.db_port,
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
DATA_SOURCE = FLAGS.bos_parent_folder

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

def get_images_path(test=True, jsonpath=None) -> dict:
    """
    TODO:
    get imgs paths, imgs: (N, 24, 3, H, W)                                               
    where N denotes the total num of a batch images
    the amount may be enormous, so just keep the path instead of the raw imgs

    return: [(image_id, pic_absolute_path)]
    """
    ret_list = []
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
    elif jsonpath is not None and os.path.isfile(jsonpath):
        import json
        img_id = 0
        with open(jsonpath, "r") as f:
            # 每行是一个json object：
            # {"image_id": 0, "image_path": "path/to/image"}
            test_json = json.load(f)
        for item in test_json:
            # FIXME: 这里暂时用cnt作为img_id，之后改为上游任务传来的img_id
            # img_id = item["image_id"]
            img_path = item["image_path"]
            try:
                for img in os.listdir(img_path):
                    if not img.startswith("render"):
                        continue
                    ret_list.append((img_id, os.path.join(img_path, img)))
                    img_id += 1
            except FileNotFoundError as e:
                print(f"image_path {img_path} not found")
    return ret_list

time_generate1 = 0
time_generate2 = 0
model, vis_processors = None, None

def generate_coarse_captions(images_path_list) -> torch.Tensor():
    """
    generate coarse captions
    return list: [(img_id, image_path, coarse_captions[])]
    """
    global time_generate1, time_generate2, BATCH_SIZE, model, vis_processors, NUM_CAPTIONS
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
        if model is None:
            model, vis_processors = get_caption_blip2_model()
        image_list = [vis_processors["eval"](image).unsqueeze(0).to(device) for image in image_list]
        batch_image = torch.cat(image_list, dim=0)
        tic = time()
        object_list = model.generate({"image": batch_image, "prompt": [prompt for _ in range(batch_image.shape[0])]}, max_length=5) # 如果不设置max_length，可能会由于prompt的token太长爆显存
        time_generate1 += time() - tic
        print(len(object_list))
        tic = time()
        coarse_caption_list = model.generate({"image": batch_image, "prompt": [full_prompt % object for object in object_list]}, use_nucleus_sampling=True, num_captions=NUM_CAPTIONS)
        time_generate2 += time() - tic
        for i in range(len(tmp_img_path_list)):
            ret_list.append((*tmp_img_path_list[i], coarse_caption_list[i * NUM_CAPTIONS: (i + 1) * NUM_CAPTIONS]))
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
    port = database_config.pop("port")
    password = database_config.pop("password")
    database = database_config.pop("database")
    table = database_config.pop("table", "model_image_caption")
    conn = connector.connect(
        host=host,
        user=user,
        port=port,
        password=password,
        database=database
    )
    conn.start_transaction()
    cursor = conn.cursor()
    select_sql = f"SELECT * FROM `{database}`.`{table}` WHERE image_id = '%s' and is_delete = 0"
    insert_sql = f"INSERT INTO `{database}`.`{table}` (image_id, cap_model_tag, extra, \
        image_embedding_bucket, image_embedding_uri, image_embedding_shape, image_caption, \
        image_caption_embedding_bucket, image_caption_embedding_uri, image_caption_embedding_shape) \
        VALUES ('%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s')"
    update_sql = "UPDATE `" + database + "`.`" + table + "` SET cap_model_tag = '{1}', extra = '{2}', \
        image_embedding_bucket = '{3}', image_embedding_uri = '{4}', image_embedding_shape = '{5}', image_caption = '{6}', \
        image_caption_embedding_bucket = '{7}', image_caption_embedding_uri = '{8}', image_caption_embedding_shape = '{9}' \
        WHERE image_id = '{0}' and is_delete = 0"
    try:
        insert_cnt, update_cnt, skip_cnt = 0, 0, 0
        for image_id, image_embedding_uri, image_caption, image_caption_embedding_uri in store_caption_list:
            cursor.execute(select_sql % image_id)
            record = cursor.fetchall()
            assert len(record) == 0 or len(record) == 1, "valid image_id should be unique"
            # TODO: 用配置文件管理固定的值
            tmp = (image_id, "LAVIS-blip2_t5-xxl,CLIP-ViT-B/32", '{}', 'image_embedding_bucket',\
                                image_embedding_uri, '(32, 256)', image_caption.replace("'", "\\'").replace('"', '\\"'), \
                                'image_caption_embedding_bucket', image_caption_embedding_uri, '(256,)')
            if len(record) == 1:
                if record[0][1:11] == tmp:
                    print(f"image_id {image_id} already exists in database, and there is no change, so skip...")
                    skip_cnt += 1
                else:
                    print(f"image_id {image_id} already exists in database, but there are changes, so update...")
                    cursor.execute(update_sql.format(*tmp))
                    update_cnt += 1
            else:
                tmp = insert_sql % tmp
                cursor.execute(tmp)
                insert_cnt += 1
        conn.commit()
        print(f"write to database successfully, insert {insert_cnt} rows, update {update_cnt} rows, skip {skip_cnt} rows")
    except Exception:
        conn.rollback()
        print("fail to write to database")
        raise
    finally:
        cursor.close()
        conn.close()

bos_client = None
def get_bos_client():
    global bos_client, BOS_CONFIG
    if bos_client is None:
        bos_client = BosClient(BOS_CONFIG)
    return bos_client

# ############################################################################
# # TODO:
# # Save image and caption embeddings to files
# ############################################################################
def save_to_file(embeddings:List[torch.Tensor], bucket_name, local_img_path_list):
    """
    make sure the idempotence of saving progress
    """
    bos_client = get_bos_client()
    try: 
        bos_client.does_bucket_exist(bucket_name)
    except Exception as e:
        # FIXME: 使用前得手动创建bucket
        raise Exception("bucket不存在，请先创建bucket", e)
    bos_img_path = []
    for img_path in local_img_path_list:
        img_path = os.path.splitext(img_path)[0] + '.npy'
        bos_img_path.append(os.path.join(DATA_SOURCE, *img_path.split(os.path.sep)[-3:]))
    for i, embed in enumerate(embeddings):
        embed_numpy = embed.cpu().numpy()
        np.save('temp', embed_numpy)
        bos_client.put_object_from_file(bucket_name, bos_img_path[i], 'temp.npy')
    if os.path.exists('temp.npy'):
        os.remove('temp.npy')
    return bos_img_path

# ############################################################################
# # TODO:
# # send message to mq
# ############################################################################
def send_message_to_mq():
    pass

if __name__ == "__main__":
    images_path_list = get_images_path(False, IMAGE_JSON_PATH)
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
    img_path_list = [*zip(*selected_caption_list)][1]
    image_uri_list = save_to_file(image_feats, 'image-embedding-bucket', img_path_list)
    caption_uri_list = save_to_file(text_feats, 'image-caption-embedding-bucket', img_path_list)
    print('embedding存bos成功')
    tic = time()
    write_database([(image_id, image_uri, caption, caption_uri) for \
                    (image_id, _, caption), image_uri, caption_uri in zip(selected_caption_list, image_uri_list, caption_uri_list)])
    print(f'time_write_database is {time() - tic}')
    # send_message_to_mq()


# batchsize 2:
# time_generate1 is 12.068463802337646
# time_generate2 is 155.86847376823425
# time_select is 4.8412981033325195
# time_extract is 20.37248706817627
