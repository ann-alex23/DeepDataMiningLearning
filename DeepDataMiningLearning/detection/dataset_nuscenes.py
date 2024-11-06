import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools import mask as coco_mask  # For RLE decoding
import numpy as np
from pathlib import Path

class NuScenesDataset(Dataset):
    def __init__(self, 
                 root: str, 
                 split: str = 'train', 
                 transform=None, 
                 version: str = 'v1.0-mini'):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.version = version
        
        # Load annotations
        annotation_path = self.root / f"{self.version}"/"object_ann_with_split.json" 
        category_path = self.root / f"{self.version}" / "category.json"
        with open(annotation_path, "r") as f:
            all_annotations = json.load(f)
            self.annotations = [ann for ann in all_annotations if ann["split"] == self.split]
        
        with open(category_path, "r") as f:
            categories = json.load(f)
        
        # Map category tokens to names
        self.category_map = {cat['token']: cat['name'] for cat in categories}
        self.numclass=len(categories)
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # Load image
        sample_data_token = annotation['sample_data_token']
        image_path = self.get_image_path(sample_data_token)
        image = Image.open(image_path).convert("RGB")
        
        # Extract bbox and convert to tensor
        bbox = torch.tensor(annotation['bbox'], dtype=torch.float32)
        
        # Get category label
        category_token = annotation['category_token']
        category_name = self.category_map.get(category_token, "__background__")
        label = torch.tensor([list(self.category_map.values()).index(category_name)], dtype=torch.int64)

        
        # Decode RLE mask (if needed)
        # mask = None
        # if 'mask' in annotation:
        #     mask_data = {
        #         "size": annotation['mask']['size'],
        #         "counts": annotation['mask']['counts'].encode('utf-8')  # Convert to bytes
        #     }
        #     mask = coco_mask.decode(mask_data)
        #     mask = torch.tensor(mask, dtype=torch.uint8)  # Convert to tensor
        area = ((bbox[3] - bbox[1]) * (bbox[2] - bbox[0])).clone().detach().to(torch.int64)
        iscrowd=torch.zeros(1, dtype=torch.int64)
        target = {
            "boxes": bbox.unsqueeze(0),  # Add batch dimension
            "labels": label,
            'image_id':sample_data_token,
            "area":area,
            "iscrowd":iscrowd
            # "masks": mask.unsqueeze(0) if mask is not None else None
        }
        
        if self.transform:
            image, target = self.transform(image, target)
        
        return image, target

    def get_image_path(self, sample_data_token):
        # Assuming the path structure is consistent and you have a "sample_data.json" file
        sample_data_path = self.root / f"{self.version}" / "sample_data.json"
        
        with open(sample_data_path, "r") as f:
            sample_data = json.load(f)
        
        for sample in sample_data:
            if sample["token"] == sample_data_token:
                return self.root / sample["filename"]
        
        raise FileNotFoundError(f"No image found for token {sample_data_token}")

# Usage Example
dataset = NuScenesDataset(root=r"C:/Users/annal/Downloads/nuimages-v1.0-mini", split="train", version="v1.0-mini")
image, target = dataset[0]  # Load first sample
print(target["area"].tolist(),type(target["area"].tolist()))
print(target["iscrowd"].tolist())
# print("Image shape:", image.size)
# print("Bounding box:", target["boxes"])
# print("Label:", target["labels"])
# print(len(dataset.category_map.values()))

# print("Mask shape:", target["masks"].shape if target["masks"] is not None else "No mask available")
