import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
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
transform = transforms.Compose([
    transforms.Resize((420, 380)),
    transforms.ToTensor()
])

# Загрузка данных
train_data = HandDataset(csv_file="C:/Users/youss/.ssh/train/label.csv", root_dir=train_dir, transform=transform)
test_data = HandDataset(csv_file="C:/Users/youss/.ssh/test/label.csv", root_dir=test_dir, transform=transform)
test_hard_data = HandDataset(csv_file="C:/Users/youss/.ssh/test_hard/label.csv", root_dir="C:/Users/youss/.ssh/test_hard/images", transform=transform)

# DataLoader для обучающего и тестового наборов данных
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
test_hard_loader = DataLoader(test_hard_data, batch_size=batch_size, shuffle=False)

# Определение модели
class HandClassifier(nn.Module):
    def __init__(self):
        super(HandClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 105 * 95, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 класса: Правая/Левая

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 105 * 95)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Инициализация модели, функции потерь и оптимизатора
model = HandClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
num_epochs = 10
for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


# Оценка модели
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    for images, labels in test_hard_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()    

print(f"Accuracy on test images: {(100 * correct / total):.2f}%")
