import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from skimage.io import imread
from torch.utils.data import DataLoader


class Neu_Metal_Defect_dataset(torch.utils.data.Dataset):
    def __init__(self, dir):
        self.root = "/Data4/student_zhihan_data/data/NEU Metal Surface Defects Data"
        self.dir = os.path.join(self.root, dir)
        self.image_list = []
        categories = os.listdir(self.dir)
        for i in categories:
            self.image_list.extend([os.path.join(self.dir, i, j) for j in os.listdir(os.path.join(self.dir, i))])
        self.image_list.sort()
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        path = self.image_list[idx]
        image = imread(path) / 255.0 
        image_tensor = torch.from_numpy(image)
        return image_tensor


if __name__ == "__main__":
    dataset = Neu_Metal_Defect_dataset("train")
    dataloader = DataLoader(dataset, batch_size=32, num_workers=12, shuffle=True)
    for idx, data in enumerate(dataloader):
        pass



        