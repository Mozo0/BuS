import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torchvision import transforms


class BusDataset(Dataset):
    def __init__(self, root_dir, transform=None, device='cpu'):
        self.root_dir = root_dir
        self.transform = transform
        self.device = device

        self.images = []
        self.labels = []
        self.filenames = []

        for folder in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, folder)):
                label = 1 if folder == 'malignant' else 0
                for filename in os.listdir(os.path.join(root_dir, folder)):
                    self.images.append(os.path.join(root_dir, folder, filename))
                    self.labels.append(label)
                    self.filenames.append(filename)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = self.labels[idx]
        filename = self.filenames[idx]

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label)
        image = image.to(self.device)
        label = label.to(self.device)
        return image, label

    def get_file_name(self, idx):
        return self.filenames[idx]

class RGBtoGray(object):
    def __call__(self, img):
        gray_img = img.convert('L')
        return gray_img

class GraytoRGB(object):
    def __call__(self, img):
        gray_img = img.convert('RGB')
        return gray_img

def get_data(device, bs=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        GraytoRGB(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = BusDataset('data/BUS_dataset/train', transform=transform, device=device)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_dataset = BusDataset('data/BUS_dataset/test', transform=transform, device=device)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=True)
    val_dataset = BusDataset('data/BUS_dataset/test', transform=transform, device=device)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=True)
    return train_dataset, test_dataset, val_dataset, train_dataloader, test_dataloader, val_dataloader


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = BusDataset('./BUS_dataset/train', transform=transform,  device='cuda')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

    print("Loaded {} train sample".format(len(train_dataset)))
