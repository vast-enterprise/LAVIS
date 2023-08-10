import os 
from PIL import Image

image_dict = {}
def get_image_from_path(image_path):
    global image_dict
    if not image_dict.__contains__(image_path):
        if os.path.isfile(image_path):
            image_dict[image_path] = Image.open(image_path).convert('RGB')
        else:
            raise FileNotFoundError(f"image_path {image_path} not found")
    return image_dict[image_path]

# def test_get_images_path(test=False, jsonpath=None) -> dict:
#     """
#     TODO:
#     get imgs paths, imgs: (N, 24, 3, H, W)                                               
#     where N denotes the total num of a batch images
#     the amount may be enormous, so just keep the path instead of the raw imgs

#     return: [(image_id, pic_absolute_path)]
#     """
#     ret_list = []
#     if test:
#         pic_folder = "/mnt/pfs/users/wangdehu/tmp_dir"
#         sub_folder_list = os.listdir(pic_folder)
#         for sub_folder in sub_folder_list:
#             sub_folder = os.path.join(pic_folder, sub_folder)
#             if not os.path.isdir(sub_folder):
#                 continue
#             imgs = os.listdir(sub_folder)
#             for img in imgs:
#                 if not img.endswith('.png'):
#                     continue
#                 img = os.path.join(sub_folder, img)
#                 ret_list.append((img, img))
#     elif jsonpath is not None and os.path.isfile(jsonpath):
#         import json
#         img_id = 0
#         with open(jsonpath, "r") as f:
#             # 每行是一个json object：
#             # {"image_id": 0, "image_path": "path/to/image"}
#             test_json = json.load(f)
#         for item in test_json:
#             # FIXME: 这里暂时用cnt作为img_id，之后改为上游任务传来的img_id
#             # img_id = item["image_id"]
#             img_path = item["image_path"]
#             try:
#                 for img in os.listdir(img_path):
#                     if not img.startswith("render"):
#                         continue
#                     ret_list.append((img_id, os.path.join(img_path, img)))
#                     img_id += 1
#             except FileNotFoundError as e:
#                 print(f"image_path {img_path} not found")
#     return ret_list