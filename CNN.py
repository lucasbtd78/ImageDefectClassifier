import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import train_loader, val_loader

nb_epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DefectCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.dropout(x)

        x = self.fc1(x)
        return x

model = DefectCNN().to(device)

images, labels = next(iter(train_loader))
images = images.to(device)
outputs = model(images)
print(outputs.shape)

# Loss & Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(nb_epochs):
    #Training
    model.train()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}/{nb_epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}")
            
    #Validation
    model.eval()
    
    total_correct= 0
    total_samples = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    val_accuracy = total_correct / total_samples
    val_loss = val_loss / total_samples
    print(f"Epoch {epoch+1}/{nb_epochs} | Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy*100:.2f}%")

#Model Save  
torch.save(model.state_dict(), "defect_cnn.pth")
print("Model Save : defect_cnn.pth")


