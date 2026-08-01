import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import json
import time

device = torch.device('cpu')
print(f"Using device: {device}")

DATA_DIR = 'dataset'
IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 25

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

base_dataset = datasets.ImageFolder(DATA_DIR)
class_names = base_dataset.classes
num_classes = len(class_names)
print(f"Classes ({num_classes}): {class_names}")

with open('class_names.json', 'w') as f:
    json.dump(class_names, f)

# Split indices first, then apply different transforms via Subset wrapper
n = len(base_dataset)
val_size = int(0.15 * n)
train_size = n - val_size
generator = torch.Generator().manual_seed(42)
train_subset, val_subset = random_split(base_dataset, [train_size, val_size], generator=generator)

train_ds = datasets.ImageFolder(DATA_DIR, transform=train_transform)
val_ds = datasets.ImageFolder(DATA_DIR, transform=val_transform)

train_subset.dataset = train_ds
val_subset.dataset = val_ds

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

print(f"Train: {train_size}, Val: {val_size}")


class PlantCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),   # 96->48
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 48->24
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2), # 24->12
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2), # 12->6
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = PlantCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

best_val_acc = 0.0

for epoch in range(EPOCHS):
    start = time.time()
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += images.size(0)

    scheduler.step()
    train_loss = running_loss / total
    train_acc = running_correct / total

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += images.size(0)

    val_acc = val_correct / val_total
    elapsed = time.time() - start
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | Time: {elapsed:.1f}s")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'plant_disease_model.pth')

print(f"\nBest validation accuracy: {best_val_acc*100:.2f}%")
print("Model saved as plant_disease_model.pth")
