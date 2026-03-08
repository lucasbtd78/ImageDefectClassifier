import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

classes = ["class_A", "class_B"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DefectCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3,16,3)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16,32,3)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(32,2)

    def forward(self,x):

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        x = self.pool(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)

        x = self.fc1(x)

        return x


# Load model
model = DefectCNN().to(device)
model.load_state_dict(torch.load("defect_cnn.pth", map_location=device))
model.eval()


# Transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


# Picture Load
image_path = "test.png"
image = Image.open(image_path).convert("RGB")

image = transform(image)
image = image.unsqueeze(0)
image = image.to(device)

# Prediction
with torch.no_grad():

    outputs = model(image)

    pred = torch.argmax(outputs, dim=1)

    predicted_class = classes[pred.item()]

print("Prediction :", predicted_class)

#------------------or------------------------
#--------------Multi_picture_with_folder------------------
'''
image_folder = "predict_images"


for img_name in os.listdir(image_folder):

    img_path = os.path.join(image_folder, img_name)

    image = Image.open(img_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)
    '''



'''
print(f"{img_name} → {predicted_class}")
'''