import torch
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import lmdb
import json
import pathlib
import ujson
import torchvision.transforms as transforms
import random
import cv2
from tqdm import tqdm


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

    x0, y0, x1, y1 = min([x[0] for x in polygon]), min([x[1] for x in polygon]), max([x[0] for x in polygon]), max([x[1] for x in polygon])
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


class AnyWordLmdbDataset(Dataset):
    def __init__(
        self,
        lmdb_path,
        seed,
        resolution=256,
        ttf_size=64,
        max_len=25,
        language=None,
        train_vae=False,
    ):
        self.resolution = resolution
        self.ttf_size = ttf_size
        self.max_len = max_len
        self.language = language
        self.train_vae = train_vae
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((resolution, resolution)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        random.seed(seed)
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        self.txn = self.env.begin(write=False)
        self.index_list = [] 
        self.build_index()
        self._length = len(self.index_list)

    def build_index(self):

        cursor = self.txn.cursor()
        total_items = self.txn.stat()['entries'] 
        with tqdm(total=total_items, desc="Building Index") as pbar:
            for key, value in cursor:
                if key.startswith(b'image-'):
                    id = key.decode('utf-8').replace('image-', '')
                    annotation_key = f'annotations-{id}'.encode('utf-8')
                    annotations_bin = self.txn.get(annotation_key)

                    if annotations_bin is None:
                        pbar.update(1)
                        continue

                    annotations = json.loads(annotations_bin.decode('utf-8'))
                    for sub_id in range(len(annotations)):
                        self.index_list.append((id, sub_id))
                
                pbar.update(1)


    def __len__(self):
        return self._length
    
    def _decode_image(self, image_bin):
        image_buf = np.frombuffer(image_bin, dtype=np.uint8)
        img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_pil = Image.fromarray(img)
        return img_pil
    
    def __getitem__(self, idx):
        id, sub_id = self.index_list[idx]
        
        image_key = f'image-{id}'.encode('utf-8')
        annotation_key = f'annotations-{id}'.encode('utf-8')

        image_bin = self.txn.get(image_key)
        annotations_bin = self.txn.get(annotation_key)

        image = self._decode_image(image_bin)
        annotations = json.loads(annotations_bin.decode('utf-8'))
        if len(annotations) == 0:
            raise ValueError(f"No annotations found for index {idx}")

        annotation = annotations[sub_id]
        
        full_image = np.array(image)
        height, width = full_image.shape[:2]
        im_shape = (width, height) 
        location = calculate_square(full_image, annotation['polygon'])
        crop_image = full_image[location[1]:location[3], location[0]:location[2]]
        trans_image = self.transform(crop_image)
        if self.train_vae:
            return {"image": trans_image}
        mask, masked_image, mask_rect = generate_mask(trans_image, im_shape, self.resolution, annotation['polygon'], location)
        text = annotation['text']
        draw_ttf = self.draw_text(text)
        glyph = self.draw_glyph(text, mask_rect) 
        info = {
            "image": trans_image,
            'mask': mask,
            'masked_image': masked_image,
            'ttf_img': draw_ttf,
            'glyph': glyph,
            "text": text
        }
        return info

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

