import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
import torchvision
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    """
    构建图像预处理的变换管道。
    Args:
        input_size (int): 输入图像的目标尺寸。
    Returns:
        transform (callable): 图像预处理的变换管道。
    """
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    该函数用于自适应图片切块时，自动选择最合适的目标宽高比。
    Args:
        aspect_ratio (float): 输入图像的宽高比。
        target_ratios (list of tuple): 目标宽高比列表，每个元素为一个 (width, height) 元组。
        width (int): 输入图像的宽度。
        height (int): 输入图像的高度。
        image_size (int): 目标图像的尺寸。
    Returns:
        tuple: 最接近的目标宽高比 (width, height)。
    """
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """
    该函数用于多块视觉推理任务（如大模型视觉输入），能让图片自适应地切分为合适数量和比例的小块，
    提升模型对不同尺寸和比例图片的适应性。
    Args:
        image (PIL.Image): 输入图像。
        min_num (int): 最小切块数量。
        max_num (int): 最大切块数量。
        image_size (int): 目标图像的尺寸。
        use_thumbnail (bool): 是否使用缩略图。
    Returns:
        processed_images: 切分后的图像块列表。
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    """
    加载并预处理输入图像。
    Args:
        image_file (str): 图像文件路径。
        input_size (int): 输入图像的目标尺寸。
        max_num (int): 最大切块数量。
    Returns:
        pixel_values (torch.Tensor): 预处理后的图像张量。
    """
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    """
    根据给定的边界、帧率和最大帧数，计算视频切分的帧索引。
    Args:
        bound (tuple or None): 视频的起始和结束时间边界 (start, end)，单位为秒。如果为 None，则使用整个视频。
        fps (float): 视频的帧率。
        max_frame (int): 视频的最大帧数。
        first_idx (int): 起始帧索引，默认为 0。
        num_segments (int): 需要切分的段数。
    Returns:
        frame_indices (np.ndarray): 计算得到的帧索引数组。
    """
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    """
    从视频文件中均匀采样指定数量的帧，对每一帧进行切块和预处理，最终返回所有帧的patch张量和每帧patch数量的列表。
    Args:
        video_path (str): 视频文件路径。
        bound (tuple or None): 视频的起始和结束时间边界 (start, end)，单位为秒。如果为 None，则使用整个视频。
        input_size (int): 输入图像的目标尺寸。
        max_num (int): 每帧图像的最大切块数量。
        num_segments (int): 需要采样的帧数。
    Returns:
        pixel_values (torch.Tensor): 预处理后的所有帧的patch张量。
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    print(world_size, 'xxx')
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL3-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

# 可视化采样到的16帧
def show_pixel_values(pixel_values, num_show=16, image_size=448):
    # pixel_values: [N, 3, H, W]，已归一化
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    plt.figure(figsize=(16, 4))
    for i in range(min(num_show, pixel_values.shape[0])):
        img = pixel_values[i].cpu() * std + mean  # 反归一化
        img = img.clamp(0, 1)
        img = torchvision.transforms.ToPILImage()(img)
        plt.subplot(1, num_show, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Frame {i+1}')
    plt.tight_layout()
    plt.show()

path = 'pretrained/InternVL3-2B'
device_map = split_model('InternVL3-2B')
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=False,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map="auto").eval()

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
generation_config = dict(max_new_tokens=1024, do_sample=True)

video_path = './data/ucf/train/videos/Abuse001_x264.mp4'
# [8, 3, 448, 448], [1, 1, 1, 1, 1, 1, 1, 1]
pixel_values, num_patches_list = load_video(video_path, num_segments=16, max_num=1)
pixel_values = pixel_values.to(torch.bfloat16).cuda()
# show_pixel_values(pixel_values, num_show=16, image_size=448)
video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])

with open('prompt_ROSES.txt', 'r', encoding='utf-8') as f:
    prompt = f.read()
# different prompts comparision
# question = video_prefix + 'Are there any anomalies in the video?'
question = video_prefix + prompt
# Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list, history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

with open('prompt_decision_ROSES.txt', 'r', encoding='utf-8') as f:
    question = f.read()
response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               num_patches_list=num_patches_list, history=history, return_history=True)
print(f'User: {question}\nAssistant: {response}')