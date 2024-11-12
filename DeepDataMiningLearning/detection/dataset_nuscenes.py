import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools import mask as coco_mask
import numpy as np
from pathlib import Path
from collections import defaultdict

class NuScenesDataset(Dataset):
    def __init__(self, 
                 root: str, 
                 split: str = 'train', 
                 transform=None, 
                 version: str = 'nuimages-v1.0-all-metadata/v1.0-val'):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.version = version
        
        # Load annotations
        annotation_path = self.root / f"{self.version}"/'split_nuscenes_val'/f"{split}_annotations.json"
        category_path = self.root / f"{self.version}" / "category.json"
        
        with open(annotation_path, "r") as f:
            all_annotations = json.load(f)
            # Group annotations by sample_data_token
            self.annotations = defaultdict(list)
            for ann in all_annotations:
                self.annotations[ann["sample_data_token"]].append(ann)

        with open(category_path, "r") as f:
            categories = json.load(f)
        
        # Map category tokens to names
        self.category_map = {cat['token']: cat['name'] for cat in categories}
        self.numclass = len(categories)
        # Store all unique image keys
        self.image_keys = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx):
        sample_data_token = self.image_keys[idx]
        image_annotations = self.annotations[sample_data_token]
        
        # Load image
        image_path = self.get_image_path(sample_data_token)
        image = Image.open(image_path).convert("RGB")
        
        # Initialize lists for targets
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        # Collect all annotations for the image
        for ann in image_annotations:
            # bbox  in [xmin, ymin, xmax, ymax] format
            bbox = ann['bbox']
            boxes.append(bbox)

            # Get category label from category.json file
            category_token = ann['category_token']
            category_name = self.category_map.get(category_token, "__background__")
            label = list(self.category_map.values()).index(category_name)
            labels.append(label)
            
            # Calculate area and add iscrowd value
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            areas.append(width * height)
            iscrowd.append(0)

        # Convert lists to tensors and prepare target dictionary
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": int(idx),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }
        
        # Apply transforms
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
                return self.root / 'nuimages-v1.0-all-samples' / sample["filename"]
        
        raise FileNotFoundError(f"No image found for token {sample_data_token}")

# Example
# dataset = NuScenesDataset(root=r"C:/Users/annal/Downloads/nuscenes_dataset", split="val", version="nuimages-v1.0-all-metadata/v1.0-val")
# print(len(dataset))
# image, target = dataset[4]
# print(target)  # Load first sample
# print("Bounding boxes:", target["boxes"])
# print("Labels:", target["labels"])
