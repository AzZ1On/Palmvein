import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import os
import pandas as pd
from PIL import Image

# Определение класса Dataset для загрузки изображений и меток
class HandDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file, delimiter=";")
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = self.annotations.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = 1 if self.annotations.iloc[idx, 2] == "Right" else 0  # 1 для правой руки, 0 для левой

        if self.transform:
            image = self.transform(image)

        return image, label

# Пути к папкам с данными
train_dir = "C:/Users/youss/.ssh/train/images"
test_dir = "C:/Users/youss/.ssh/test/images"

# Преобразование изображений
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Загрузка данных
train_data = HandDataset(csv_file="C:/Users/youss/.ssh/train/label.csv", root_dir=train_dir, transform=train_transform)
test_data = HandDataset(csv_file="C:/Users/youss/.ssh/test/label.csv", root_dir=test_dir, transform=test_transform)
test_hard_data = HandDataset(csv_file="C:/Users/youss/.ssh/test_hard/label.csv", root_dir="C:/Users/youss/.ssh/test_hard/images", transform=test_transform)

# DataLoader для обучающего и тестового наборов данных
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
test_hard_loader = DataLoader(test_hard_data, batch_size=batch_size, shuffle=False)

# Использование предобученной модели ResNet
class HandClassifier(nn.Module):
    def __init__(self):
        super(HandClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        x = self.resnet(x)
        return x

# Инициализация модели, функции потерь и оптимизатора
model = HandClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Обучение модели
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # Точность на обучающем наборе данных
    model.eval()
    correct_train = 0
    total_train = 0
    with torch.no_grad():
        for images, labels in train_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
    
    print(f"Accuracy on train images: {(100 * correct_train / total_train):.2f}%")

# Оценка модели
model.eval()
correct_test = 0
total_test = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()
        
    for images, labels in test_hard_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()    

print(f"Accuracy on test images: {(100 * correct_test / total_test):.2f}%")

# PS C:\Users\youss\OneDrive\Документы\GitHub\Palmvein> & C:/Users/youss/AppData/Local/Programs/Python/Python312/python.exe c:/Users/youss/OneDrive/Документы/GitHub/Palmvein/classificator.py
# C:\Users\youss\AppData\Local\Programs\Python\Python312\Lib\site-packages\torchvision\models\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
#   warnings.warn(
# C:\Users\youss\AppData\Local\Programs\Python\Python312\Lib\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
#   warnings.warn(msg)
# Epoch [1/20], Loss: 0.1049
# Accuracy on train images: 91.55%
# Epoch [2/20], Loss: 0.0238
# Accuracy on train images: 99.55%
# Epoch [3/20], Loss: 0.0180
# Accuracy on train images: 99.60%
# Epoch [4/20], Loss: 0.0094
# Accuracy on train images: 99.80%
# Epoch [5/20], Loss: 0.0095
# Accuracy on train images: 99.50%
# Epoch [6/20], Loss: 0.0134
# Accuracy on train images: 99.75%
# Epoch [7/20], Loss: 0.0143
# Accuracy on train images: 99.60%
# Epoch [8/20], Loss: 0.0052
# Accuracy on train images: 99.55%
# Epoch [9/20], Loss: 0.0114
# Accuracy on train images: 99.80%
# Epoch [10/20], Loss: 0.0086
# Accuracy on train images: 99.80%
# Epoch [11/20], Loss: 0.0165
# Accuracy on train images: 99.35%
# Epoch [12/20], Loss: 0.0113
# Accuracy on train images: 99.55%
# Epoch [13/20], Loss: 0.0056
# Accuracy on train images: 99.80%
# Epoch [14/20], Loss: 0.0063
# Accuracy on train images: 99.60%
# Epoch [15/20], Loss: 0.0061
# Accuracy on train images: 99.90%
# Epoch [16/20], Loss: 0.0031
# Accuracy on train images: 99.95%
# Epoch [17/20], Loss: 0.0054
# Accuracy on train images: 99.80%
# Epoch [18/20], Loss: 0.0025
# Accuracy on train images: 99.90%
# Epoch [19/20], Loss: 0.0041
# Accuracy on train images: 99.85%
# Epoch [20/20], Loss: 0.0018
# Accuracy on train images: 100.00%
# Accuracy on test images: 93.32%