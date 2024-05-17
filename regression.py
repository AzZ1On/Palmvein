import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import pandas as pd
from PIL import Image

# Определение класса Dataset для загрузки изображений и меток
class HandAngleDataset(Dataset):
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
        angle = float(self.annotations.iloc[idx, 1])  # Угол ладони

        if self.transform:
            image = self.transform(image)

        return image, angle

# Пути к папкам с данными
train_dir = "C:/Users/youss/.ssh/train/images"
test_dir = "C:/Users/youss/.ssh/test/images"
test_hard_dir = "C:/Users/youss/.ssh/test_hard/images"

# Преобразование изображений
transform = transforms.Compose([
    transforms.Resize((420, 380)),
    transforms.ToTensor()
])

# Загрузка данных
train_data = HandAngleDataset(csv_file="C:/Users/youss/.ssh/train/label.csv", root_dir=train_dir, transform=transform)
test_data = HandAngleDataset(csv_file="C:/Users/youss/.ssh/test/label.csv", root_dir=test_dir, transform=transform)
test_hard_data = HandAngleDataset(csv_file="C:/Users/youss/.ssh/test_hard/label.csv", root_dir=test_hard_dir, transform=transform)

# DataLoader для обучающего и тестового наборов данных
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
test_hard_loader = DataLoader(test_hard_data, batch_size=batch_size, shuffle=False)

# Определение модели для регрессии
class HandAngleRegressor(nn.Module):
    def __init__(self):
        super(HandAngleRegressor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 105 * 95, 128)
        self.fc2 = nn.Linear(128, 1)  # Выводит одно числовое значение - угол

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 105 * 95)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Инициализация модели, функции потерь и оптимизатора
model = HandAngleRegressor()
criterion = nn.L1Loss()  # Использование MAE (Mean Absolute Error)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for images, angles in train_loader:
        optimizer.zero_grad()
        outputs = model(images.float())  # Приведение к типу Float
        loss = criterion(outputs.squeeze(), angles.float())  # Приведение к типу Float
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    mean_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {mean_loss:.4f}")

# Оценка модели на тестовом наборе данных
model.eval()
total_loss = 0
with torch.no_grad():
    for images, angles in test_loader:
        outputs = model(images.float())  # Приведение к типу Float
        loss = criterion(outputs.squeeze(), angles.float())  # Приведение к типу Float
        total_loss += loss.item() * images.size(0)

mean_loss = total_loss / len(test_loader.dataset)
print(f"Mean Absolute Error on test images: {mean_loss:.4f}")

# Оценка модели на новом тестовом наборе данных test_hard
model.eval()
total_loss_hard = 0
with torch.no_grad():
    for images_hard, angles_hard in test_hard_loader:
        outputs_hard = model(images_hard.float())  # Приведение к типу Float
        loss_hard = criterion(outputs_hard.squeeze(), angles_hard.float())  # Приведение к типу Float
        total_loss_hard += loss_hard.item() * images_hard.size(0)

mean_loss_hard = total_loss_hard / len(test_hard_loader.dataset)
print(f"Mean Absolute Error on hard test images: {mean_loss_hard:.4f}")

# PS C:\Users\youss\OneDrive\Документы\GitHub\Palmvein> & C:/Users/youss/AppData/Local/Programs/Python/Python312/python.exe c:/Users/youss/OneDrive/Документы/GitHub/Palmvein/regression.py
# Epoch [1/10], Loss: 28.9238
# Epoch [2/10], Loss: 6.8257
# Epoch [3/10], Loss: 6.1452
# Epoch [4/10], Loss: 5.4594
# Epoch [5/10], Loss: 3.9266
# Epoch [6/10], Loss: 3.5086
# Epoch [7/10], Loss: 4.5958
# Epoch [8/10], Loss: 3.1887
# Epoch [9/10], Loss: 5.1954
# Epoch [10/10], Loss: 2.9482
# Mean Absolute Error on test images: 2.5967
# Mean Absolute Error on hard test images: 46.0138