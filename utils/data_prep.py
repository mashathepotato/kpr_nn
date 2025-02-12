import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ConeDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.count_missing = 0
        self.ann_files = [f for f in os.listdir(ann_dir) if f.endswith('.json')]
        self.ann_files = [f for f in os.listdir(ann_dir) if f.endswith('.json')]
        
        self.valid_files = []

        for ann_file in os.listdir(ann_dir):
            if not ann_file.endswith('.json'):
                continue
            
            ann_path = os.path.join(self.ann_dir, ann_file)
            with open(ann_path, 'r') as f:
                ann_data = json.load(f)
            
            # Some samples are rejected
            if ann_data.get("status") == "rejected":
                continue

            img_file = ann_file.replace(".json", "")
            img_path = os.path.join(self.img_dir, img_file)
            if os.path.exists(img_path):
                self.valid_files.append(ann_file)
            else:
                self.count_missing += 1
                # print(f"Missing image for annotation {ann_file}")
        # print(self.count_missing)
    
    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        ann_file = self.valid_files[idx]
        img_id = ann_file.replace(".json", "")
        img_path = os.path.join(self.img_dir, img_id)
        ann_path = os.path.join(self.ann_dir, ann_file)

        image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = image.size

        with open(ann_path, 'r') as f:
            ann_data = json.load(f)

        keypoints = torch.tensor(ann_data.get("keypoints", []), dtype=torch.float32)

        # Rescaling into 80x80
        target_size = (80, 80)
        scale_x = target_size[0] / orig_width
        scale_y = target_size[1] / orig_height

        if keypoints.numel() > 0:
            keypoints[:, 0] *= scale_x
            keypoints[:, 1] *= scale_y

        if self.transform:
            image = self.transform(image)

        return image, keypoints