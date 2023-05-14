import torch
from torch.utils.data import Dataset
import os
from  PIL import Image
from torchvision import transforms

class Cat_breads(Dataset):
    def __init__(self, root_dir = 'cats-breads',transform = False):
        self.root_dir = root_dir
        self.labels = self.__getlabels__()
        self.transform = transform if transform else transforms.Compose([
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                transforms.Resize((240,240)),
                ])

    def __getlabels__(self):
        sub_dirs = [sub_dir for sub_dir in os.listdir(self.root_dir)]
        return dict(zip(sub_dirs,torch.arange(len(sub_dirs))))

    def __files__(self):
        sub_dirs = [sub_dir for sub_dir in os.listdir(self.root_dir)]
        img_names = []
        for sub_dir in sub_dirs:
            img_names.extend([f'{self.root_dir}/{sub_dir}/{img_name}' for img_name in os.listdir(f'{self.root_dir}/{sub_dir}')])
        return img_names

    def __len__(self):
        return len(self.__files__())

    def __getitem__(self,idx):
        file_name = self.__files__()[idx]
        img = self.transform(Image.open(file_name))
        label = file_name.split('/')[1]
        return img,self.labels[label]
