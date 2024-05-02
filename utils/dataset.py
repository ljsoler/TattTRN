from torchvision.datasets import ImageFolder
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from pathlib import Path

class SyntheticImagesDS(ImageFolder):
    def __init__(self, root_dir: str, images_dir: str, templates_dir: str, transform=None, transform_target = None):
        super().__init__(os.path.join(root_dir, images_dir), transform)
        self.template_dir = os.path.join(root_dir, templates_dir)
        self.transform_template_target = transform_target


    def __getitem__(self, idx):
        image, target = super().__getitem__(idx)
        image_path = self.imgs[idx][0]
        template_filename = Path(image_path).parent.name + '.png'
        template_path = os.path.join(self.template_dir, template_filename)
        template = Image.open(template_path)

        if self.transform_template_target is not None:
            template = self.transform_template_target(template)

        return image, template, target
    

    def get_template_name(self, idx):
        image_path = self.imgs[idx][0]
        return Path(image_path).parent.name


class ResetIndicesDS(Dataset):
    def __init__(self, subset):
        self.subset = subset
        unique_targets = sorted(set(target for _, _, target in subset))
        self.target_map = {old_target: torch.tensor(new_target, dtype=torch.long) for new_target, old_target in enumerate(unique_targets)}
        print(f'[INFO] Unique targets: {len(unique_targets)}')

    def __getitem__(self, index):
        image, template, old_target = self.subset[index]
        new_target = self.target_map[old_target]  # remap the target here
        return image, template, new_target
    def __len__(self):
        return len(self.subset)

