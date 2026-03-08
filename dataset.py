import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
#Transforms
train_transforms = transforms.Compose ([
    transforms.Resize ((224,224)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
])

# Dataset
dataset = ImageFolder("data", transform=train_transforms )
print(dataset.classes)
print(len(dataset))

#Split
train_size = int(0.8 * len(dataset))  # 80% training
val_size = len(dataset) - train_size  # 20% validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

#Batch Recuperation
images, labels= next(iter(train_loader))
print(images.shape)
print(labels.shape)
