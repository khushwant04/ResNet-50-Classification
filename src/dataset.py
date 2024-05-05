import os
import torch 
from torch import nn
from torch.utils.data import Dataset,random_split,DataLoader
from PIL import Image
from pathlib import Path 
from src.transforms import make_transforms



class Animals(Dataset):
    def __init__(self,root,transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        
        for i, label_dir in enumerate(sorted(os.listdir(root))):
            class_dir = os.path.join(root,label_dir)
            self.class_to_idx[label_dir] = i
            for image_file in sorted(os.listdir(class_dir)):
                self.images.append(os.path.join(class_dir,image_file))
                self.labels.append(i)
                
    
    def get_class_to_idx(self):
        return self.class_to_idx
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        return image, label
    
    
    def train_test_val_data(root_dir='src/data/animal_data',transforms=make_transforms()):
        full_dataset = Animals(root=root_dir)
        
        train_size = int(0.7* len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        train_data, val_data,test_data = random_split  (full_dataset,[train_size,val_size,test_size])
        train_data.dataset.transform = transforms
        test_data.dataset.transform = transforms
        val_data.dataset.transform = transforms
        return train_data,val_data,test_data

        
            
        
        
        