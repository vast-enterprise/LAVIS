import torch
from lavis.models import load_model_and_preprocess
from tqdm import tqdm

from image_fetch import get_image_from_path

assert torch.cuda.is_available(), "cuda is not available"
device = torch.device("cuda")

model, vis_processors, txt_processors = None, None, None

def extract_features(img_id_list, img_path_list, caption_list):
    global model, vis_processors, txt_processors
    BS_IMAGE, BS_TEXT = 8, 256
    # model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="coco", is_eval=True, device=device)
    if model is None:
        model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=device)
    image_input_list = [vis_processors["eval"](get_image_from_path(image_path)).unsqueeze(0).to(device) for image_path in img_path_list]
    text_input_list = [txt_processors["eval"](caption) for caption in caption_list]
    image_feats = []
    text_feats = []
    
    sample = {}
    for i in tqdm(range(0, len(img_id_list), BS_IMAGE), desc="图片特征抽取中："):
        sample["image"] = torch.cat(image_input_list[i: min(len(img_id_list), i + BS_IMAGE)], dim=0)
        tmp = model.extract_features(sample, mode="image")['image_embeds_proj']
        image_feats.extend(tmp.chunk(tmp.size()[0], dim=0))
    sample = {}
    for i in tqdm(range(0, len(img_id_list), BS_TEXT), desc="文本特征抽取中："):
        sample["text_input"] = text_input_list[i: min(len(img_id_list), i + BS_TEXT)]
        tmp = model.extract_features(sample, mode="text")['text_embeds_proj'][:, 0, :]
        text_feats.extend(tmp.chunk(tmp.size()[0], dim=0))
    return image_feats, text_feats


