from lavis import Blip2T5
from lavis.models import load_preprocess, load_model_and_preprocess
# # pip install accelerate
from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Block
import torch
from omegaconf import OmegaConf
from deepspeed import init_inference
from time import time
from os import environ
from tqdm import tqdm

from image_fetch import get_image_from_path

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
BATCH_SIZE = int(environ.get("blip2_generate_batchsize", 5))
NUM_CAPTIONS = int(environ.get("blip2_generate_num_captions", 5))

model, vis_processors = None, None

def get_caption_blip2_model(try_parellel=False):
    global model, vis_processors
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
        if model is None:
            time_load_model = time()
            model, vis_processors, _ = load_model_and_preprocess(name='blip2_t5', model_type='pretrain_flant5xxl', is_eval=True, device=device)
            print("读取模型完毕，耗时%.2f秒" % (time() - time_load_model))
        # import ipdb; ipdb.set_trace()
    return model, vis_processors



def generate_coarse_captions(images_path_list) -> torch.Tensor():
    """
    generate coarse captions
    return list: [(img_id, image_path, coarse_captions[])]
    """
    global BATCH_SIZE, NUM_CAPTIONS
    prompt = "Question: what object is in this image? Answer:"
    full_prompt = "Question: what is the structure and geometry of this %s?"
    ret_list = []
    print("读取BLIP2模型中...")
    time_generate1 ,time_generate2 = 0, 0
    model, vis_processors = get_caption_blip2_model()
    with tqdm(total=len(images_path_list), desc="caption生成进度（图片）：") as pbar:
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
            tic = time()
            coarse_caption_list = model.generate({"image": batch_image, "prompt": [full_prompt % object for object in object_list]}, use_nucleus_sampling=True, num_captions=NUM_CAPTIONS)
            time_generate2 += time() - tic
            coarse_caption_list = [caption.capitalize() for caption in coarse_caption_list]
            for i in range(len(tmp_img_path_list)):
                ret_list.append((*tmp_img_path_list[i], coarse_caption_list[i * NUM_CAPTIONS: (i + 1) * NUM_CAPTIONS]))
            pbar.update(len(tmp_img_path_list))
    print(f'生成coarse caption结束，其中第一步生成耗时{time_generate1:.2f}秒，第二步生成耗时{time_generate2:.2f}秒')
    return ret_list