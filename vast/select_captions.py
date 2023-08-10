import torch
import clip
from torch.nn import CosineSimilarity
from tqdm import tqdm
from pprint import pprint

from image_fetch import get_image_from_path

cos, clip_model, clip_preprocess = None, None, None
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def select_captions(coarse_caption_list) -> dict:
    global cos, clip_model, clip_preprocess
    if cos is None:
        # set up CLIP
        cos = CosineSimilarity(dim=1, eps=1e-6)
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, download_root='/mnt/pfs/share/pretrained_model/.cache')
        clip_model.eval()
    selected_caption_list = []
    bucket = {}
    for img_id, img_path, coarse_caption_list in tqdm(coarse_caption_list, desc="clip对每个image选择最佳caption中："):
        img = get_image_from_path(img_path)
        img_input = clip_preprocess(img).unsqueeze(0).to(device)
        text = clip.tokenize(coarse_caption_list).to(device)
        image_features = clip_model.encode_image(img_input)
        text_features = clip_model.encode_text(text)
        score = cos(image_features, text_features)
        indices = torch.argmax(score)
        selected_caption_list.append((img_id, img_path, coarse_caption_list[indices]))
        print('\n')
        pprint(f'{coarse_caption_list}')
        print()
        print(f"选择了第{indices.item()}个caption: {coarse_caption_list[indices]}")
        if bucket.__contains__(indices.item()):
            bucket[indices.item()] += 1
        else:
            bucket[indices.item()] = 1
    print("clip选择情况:")
    print(bucket)
    return selected_caption_list, bucket

