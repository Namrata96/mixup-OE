from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch

def mixup_image(img1, img2, label1, label2, lam):
    mixed_img = lam*np.asarray(img1) + (1-lam)*np.asarray(img2)
    mixed_label = lam*label1 + (1-lam)*label2
    return Image.fromarray(mixed_img.clip(0, 255).astype('uint8'), 'RGB'), mixed_label

class MixupInputDataset(Dataset):
    def __init__(self, dataset, transform=None, lam_mag=0.5, lam_random=False, lam_direction='random'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset = dataset
        self.transform = transform
        self.lam_mag = 1
        self.lam_random = lam_random
        self.lam_direction = lam_direction

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img, label = self.dataset[idx]
        idx2 = np.random.randint(0, len(self.dataset))
        img2, label2 = self.dataset[idx2]
        
        if self.lam_random:
            lam_mag = self.lam_mag * (0.5 + np.random.rand()/2)
        else:
            lam_mag = self.lam_mag
        
        if self.lam_direction == 'neg':
            mixed_img, mixed_label = mixup_image(img, img2, label, label2, -lam_mag)
        elif self.lam_direction == 'pos':
            mixed_img, mixed_label = mixup_image(img, img2, label, label2, 1+lam_mag)
        elif self.lam_direction == 'inter':
            mixed_img, mixed_label = mixup_image(img, img2, label, label2, lam_mag)
        else:
            if np.random.rand() > 0.5:
                mixed_img, mixed_label = mixup_image(img, img2, label, label2, -lam_mag)
            else:
                mixed_img, mixed_label = mixup_image(img, img2, label, label2, 1+lam_mag)

        if self.transform:
            mixed_img = self.transform(mixed_img)

        return mixed_img, mixed_label
