import torch
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import pathlib
import ujson
import torchvision.transforms as transforms
import random
import cv2

def load_json(file_path: str):
    file_path = pathlib.Path(file_path)
    with open(file_path, 'rb') as f:
        content = f.read()
    return ujson.loads(content)


def generate_mask(trans_image, im_shape, resolution, polygon, location):
    if isinstance(im_shape, int): im_shape = (im_shape, im_shape)
    mask = Image.new("L", im_shape, 255)
    draw = ImageDraw.Draw(mask)
    polygon_int = [(int(x), int(y)) for x, y in polygon]
    draw.polygon(polygon_int, fill=0)
    mask = np.array(mask.convert("L"))[location[1]:location[3], location[0]:location[2]]
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((resolution, resolution))
        ])
    mask = transform(mask)
    mask = torch.where(mask < 0.5, torch.tensor(0.0), torch.tensor(1.0))
    masked_image = trans_image * mask.expand_as(trans_image)

    mask_np = mask.squeeze().byte().cpu().numpy() 
    mask_np = np.transpose(mask_np)
    points = np.column_stack(np.where(mask_np == 0))
    rect = cv2.minAreaRect(points)

    return mask, masked_image, rect


def calculate_square(full_image, polygon):
    gray = cv2.cvtColor(full_image, cv2.COLOR_RGB2GRAY)
    # Find non-white areas (i.e., pixel values less than 255)
    coords = cv2.findNonZero(255 - gray)
    x_l, y_t, w, h = cv2.boundingRect(coords)
    x_r = x_l + w
    y_b = y_t + h

    x0, y0, x1, y1 = int(min([x[0] for x in polygon])), int(min([x[1] for x in polygon])), int(max([x[0] for x in polygon])), int(max([x[1] for x in polygon]))
    x0 = max(x_l, min(x0, x_r))
    y0 = max(y_t, min(y0, y_b))
    x1 = max(x_l, min(x1, x_r))
    y1 = max(y_t, min(y1, y_b))

    width = x1 - x0
    height = y1 - y0
    L = max(width, height)

    if w < L:
        sx0, sx1 = x_l, x_r
    else:
        sx0_min = max(x_l, x1 - L)
        sx0_max = min(x_r - L, x0)  
        sx0 = random.randint(sx0_min, sx0_max+1)   
        sx1 = sx0 + L

    if h < L:
        sy0, sy1 = y_t, y_b
    else:
        sy0_min = max(y_t, y1 - L)
        sy0_max = min(y_b - L, y0)
        sy0 = random.randint(sy0_min, sy0_max+1)
        sy1 = sy0 + L
    
    return [sx0, sy0, sx1, sy1]


class AnyWordDataset(Dataset):
    def __init__(
        self,
        json_path,
        seed,
        resolution=256,
        ttf_size=64,
        max_len=25,
        language=None,
    ):
        assert isinstance(json_path, (str, list))
        if isinstance(json_path, str):
            json_path = [json_path]
        self.resolution = resolution
        self.ttf_size = ttf_size
        self.max_len = max_len
        self.language = language
        self.raw_data = []
        for jp in json_path:
            self.raw_data += self.load_data(jp)
        self._length = len(self.raw_data)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            transforms.Resize((resolution, resolution))
        ])
        random.seed(seed)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        gt = self.raw_data[index]
        img_path = gt['img_name']
        full_image = np.array(Image.open(img_path).convert('RGB'))
        height, width = full_image.shape[:2]
        im_shape = (width, height) 
        polygon = gt['polygon']
        location = calculate_square(full_image, polygon)
        crop_image = full_image[location[1]:location[3], location[0]:location[2]]
        trans_image = self.transform(crop_image)
        mask, masked_image, mask_rect = generate_mask(trans_image, im_shape, self.resolution, polygon, location)
        text = gt['text']
        draw_ttf = self.draw_text(text[:self.max_len])
        glyph = self.draw_glyph(text, mask_rect) 
        info = {
            # "full_image": torch.tensor(full_image),
            # "location": torch.tensor(location),
            "image": trans_image,
            'mask': mask,
            'masked_image': masked_image,
            'ttf_img': draw_ttf,
            'glyph': glyph,
            "text": text
            }
        return info

    def load_data(self, json_path):
        content = load_json(json_path)
        data_list = content['data_list']
        data_root = content['data_root']
        d = []
        for item in data_list:
            if 'annotations' in item:      
                img_name = os.path.join(data_root, item['img_name'])
                for annotation in item['annotations']:
                    if len(annotation['polygon']) == 0 or len(annotation['text']) < 1:
                        continue
                    if 'language' in annotation and self.language is not None and annotation['language'] != self.language:
                        continue
                    gt = {}
                    gt['img_name'] = img_name
                    gt['polygon'] = annotation['polygon']
                    gt['text'] = annotation['text']
                    d.append(gt)                    

        return d
    
    def draw_text(self, text, font_path="AlibabaPuHuiTi-3-85-Bold.ttf"):
        R = self.ttf_size
        fs = int(0.8*R)
        interval = 128 // self.max_len
        img_tensor = torch.ones((self.max_len, R, R), dtype=torch.float)
        for i, char in enumerate(text):
            img = Image.new('L', (R, R), 255)  # Create a white image
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(font_path, fs)
            text_size = font.getsize(char)
            text_position = ((R - text_size[0]) // 2, (R - text_size[1]) // 2)
            draw.text(text_position, char, font=font, fill=interval*i)
            img_tensor[i] = torch.from_numpy(np.array(img)).float() / 255.0    
        return img_tensor
    
    def draw_glyph(self, text, rect, font_path="AlibabaPuHuiTi-3-85-Bold.ttf"):
        resolution = self.resolution
        # Create a 3-channel (RGB) background image initialized to white
        bg_img = np.ones((resolution, resolution, 3), dtype=np.uint8) * 255
        font = ImageFont.truetype(font_path, self.ttf_size)
        text_img = Image.new('RGB', font.getsize(text), (255, 255, 255))  # Create an RGB image initialized to white
        draw = ImageDraw.Draw(text_img)
        draw.text((0, 0), text, font=font, fill=(127, 127, 127))  # Draw text in gray
        text_np = np.array(text_img)
        rec_h, rec_w = rect[1]
        box = cv2.boxPoints(rect)
        if rec_h > rec_w * 1.5:
            box = [box[1], box[2], box[3], box[0]]
        dst_points = np.array(box, dtype=np.float32)
        src_points = np.float32([[0, 0], [text_np.shape[1], 0], [text_np.shape[1], text_np.shape[0]], [0, text_np.shape[0]]])
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped_text_img = cv2.warpPerspective(text_np, M, (resolution, resolution))
        # Create a mask where the text is non-white (non-background)
        mask = np.any(warped_text_img == [127, 127, 127], axis=-1)
        # Use the mask to overlay the warped_text_img onto the bg_img
        bg_img[mask] = warped_text_img[mask]
        # Convert the final image to a tensor
        # Convert to float and scale to [0, 1]
        bg_img = bg_img.astype(np.float32) / 255.0
        # Convert to PyTorch tensor
        bg_img_tensor = torch.from_numpy(bg_img).permute(2, 0, 1)  # Change from HWC to CHW format
        return bg_img_tensor
