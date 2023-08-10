import os

os.environ['HUGGINGFACE_HUB_CACHE'] = \
    os.environ.get("HUGGINGFACE_HUB_CACHE", "/mnt/pfs/share/pretrained_model/.cache/huggingface/hub")
os.environ['TRANSFORMERS_OFFLINE'] = \
    os.environ.get("TRANSFORMERS_OFFLINE", "1")

from torch.hub import set_dir
set_dir(os.environ.get("TORCH_HUB_DIR", "/mnt/pfs/share/pretrained_model/.cache/torch/hub"))

from json import dumps
import traceback

from select_captions import select_captions
from feature_extraction import extract_features
from client.bos import save_to_bos, get_task_from_bos
from get_coarse_captions import generate_coarse_captions
from client.database import write_image_caption_to_database, get_image_paths_from_model_id
from client.message_queue import produce_message
import image_fetch

def get_bos_uri(datasource, img_path):
    img_path = os.path.splitext(img_path)[0] + '.npy'
    return os.path.join(datasource, *img_path.split(os.path.sep)[-3:])

def get_task_from_json(json_path):
    if json_path is None:
        json_path = ""
    from json import load
    with open(json_path, 'r') as f:
        return load(f)

if __name__ == "__main__":
    task_path = os.environ.get("CONDUCTOR_TASK_ID_KEY")
    task_key = os.environ.get("CONDUCTOR_TASK_STREAM_KEY")
    is_test = os.environ.get("TEST", "false").lower() == "true"
    if not is_test and (not task_path or not task_key):
        raise RuntimeError(
            "CONDUCTOR_TASK_ID_KEY nor CONDUCTOR_TASK_STREAM_KEY not found in environment variables"
        )
    if is_test:
        task_path = "20230731/transcode/task-runw3idfmlfdl1exasn.txt"
    for i, task in enumerate(get_task_from_bos(task_path)):
        print("执行第{}个任务".format(i + 1))
        try:
            image_path_list = get_image_paths_from_model_id(task['id'])
            print(f'\n获得模型图片路径成功，读取该模型\"{task["name"]}\"共{len(image_path_list)}张图片\n')
            coarse_caption_list = generate_coarse_captions(image_path_list)
            print(f'\n获得粗糙描述成功\n')
            # pprint(coarse_caption_list)
            selected_caption_list = select_captions(coarse_caption_list)
            image_feats, text_feats = extract_features(*zip(*selected_caption_list)) # image_feats: [(32, 256) * bs], text_feats: [(256,) * bs]
            img_path_list = [*zip(*selected_caption_list)][1]
            uri_list = [get_bos_uri(task['source'], img_path) for img_path in img_path_list]
            save_to_bos(image_feats, 'image-embedding-bucket', uri_list)
            save_to_bos(text_feats, 'image-caption-embedding-bucket', uri_list)
            print('embedding存bos成功')
            write_image_caption_to_database([(image_id, bos_uri, caption, bos_uri) for \
                            (image_id, _, caption), bos_uri in zip(selected_caption_list, uri_list)])
            # write_image_caption_to_database([(image_id, 'test_uri', 'test', 'test_uri') for \
            #                 (image_id, _) in image_path_list])
            print('写数据库成功')
            if not is_test:
                produce_message(
                    dumps(
                        {
                            "code": 0,
                            "msg": "",
                            "key": task_key,
                            "current_id": str(task["id"]),
                            "payload": {},
                        }
                    ).encode("utf-8")
                )
        except Exception as e:
            if not is_test:
                produce_message(
                    dumps(
                        {
                            "code": -1,
                            "msg": str(e),
                            "key": task_key,
                            "current_id": str(task["id"]),
                            "payload": {},
                        }
                    ).encode("utf-8")
                )
            traceback.print_exc()
        finally:
            image_fetch.image_dict = dict()
