import random
from collections import defaultdict

import numpy as np
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

from utils.dataset import SyntheticImagesDS, ResetIndicesDS
from torchvision import transforms


class SyntheticImagesDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, images_dir: str, templates_dir: str, batch_size: int, num_workers: int, 
                 val_split: float, input_size: int, use_grayscale: bool = False):
        super().__init__()
        self.data_path = data_path
        self.images_dir = images_dir
        self.templates_dir = templates_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.input_size = input_size
        self.use_grayscale = use_grayscale

        self.stat_info = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 

        #input transformers
        self.transform = transforms.Compose([
                    transforms.Resize((input_size, input_size)),
                    transforms.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        self.transform_val = transforms.Compose([
                    transforms.Resize((input_size, input_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        #target transformers
        self.transform_template = transforms.Compose([
                    transforms.Resize((input_size, input_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    def setup(self, stage=None):
        if hasattr(self, "train_data") and hasattr(self, "val_data"):
            return

        full_dataset = SyntheticImagesDS(self.data_path, self.images_dir, self.templates_dir, self.transform, self.transform_template)
        val_dataset = SyntheticImagesDS(self.data_path, self.images_dir, self.templates_dir, self.transform_val, self.transform_template)

        template_groups = defaultdict(list)
        for idx in range(len(full_dataset)):
            template_name = full_dataset.get_template_name(idx)
            template_groups[template_name].append(idx)

        # Split the templates into train and validation sets
        templates = list(template_groups.keys())
        val_size = int(len(templates) * self.val_split)
        train_size = len(templates) - val_size

        random.shuffle(templates)
        train_templates = templates[:train_size]
        val_templates = templates[train_size:]

        # Gather the corresponding images
        train_data = [idx for template in train_templates for idx in template_groups[template]]
        val_data = [idx for template in val_templates for idx in template_groups[template]]

        subset_train = Subset(full_dataset, train_data)
        subset_train_reset_indices = ResetIndicesDS(subset_train)
        self.train_data = subset_train_reset_indices
        # self.train_data = Subset(full_dataset, train_data)
        self.val_data = Subset(val_dataset, val_data)

        self.train_templates = train_templates
        self.val_templates = val_templates

        self.t_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        self.v_dataloader = DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)


    def train_dataloader(self):
        return self.t_dataloader


    def val_dataloader(self):
        return self.v_dataloader

    # For logging purposes
    def rollback_normalisation(self, tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        tensor = tensor.clone()
        for t, m, s in zip(tensor.unbind(dim=1), mean, std):
            t.mul_(s).add_(m)
        return tensor

    def get_train_size(self):
        return len(self.train_data)

    def get_val_size(self):
        return len(self.val_data)

    def get_train_templates_count(self):
        return len(self.train_templates)

    def get_val_templates_count(self):
        return len(self.val_templates)

