import torch
from torch import nn, optim
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from sklearn.metrics import f1_score

# 1. Data Augmentation & Transforms
train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. Loading and Splitting Dataset
# Note: Update these paths if running locally vs Kaggle
data_path = '/kaggle/input/datasets/imsparsh/flowers-dataset/train'
full_train_dataset = datasets.ImageFolder(data_path, transform=train_transforms)

train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_data, val_data = random_split(full_train_dataset, [train_size, val_size])

# Re-apply strict transforms to validation set
val_data.dataset.transform = test_transforms

trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
valloader = DataLoader(val_data, batch_size=64)

# 3. Custom Class for Unlabeled Test Images
class FlowerTestDataset(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = [f for f in os.listdir(main_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        return self.transform(image), self.all_imgs[idx]

test_path = '/kaggle/input/datasets/imsparsh/flowers-dataset/test'
test_data = FlowerTestDataset(test_path, transform=test_transforms)
testloader = DataLoader(test_data, batch_size=64, shuffle=False)

# 4. Model Architecture
class FlowerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 5) 
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = FlowerClassifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 5. Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Training Loop
epochs = 10 
for e in range(epochs):
    running_loss = 0
    model.train()
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Validation step
    val_accuracy = 0
    model.eval()
    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
    print(f"Epoch {e+1}.. Train Loss: {running_loss/len(trainloader):.3f}.. "
          f"Val Accuracy: {val_accuracy/len(valloader):.3f}")

# 7. Final Evaluation & Submission Generation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in valloader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

final_f1 = f1_score(all_labels, all_preds, average='macro')
print(f"Final F1 Macro Score: {final_f1:.4f}")

# Save Submission CSV
test_results = []
class_names = full_train_dataset.classes 
with torch.no_grad():
    for images, filenames in testloader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        for i in range(len(filenames)):
            test_results.append({
                'id': filenames[i],
                'flower_class': class_names[preds[i]]
            })

df = pd.DataFrame(test_results)
df.to_csv('submission.csv', index=False)
print("submission.csv created!")

# 8. Save Model
torch.save(model.state_dict(), 'checkpoint.pth')
print("Model saved to checkpoint.pth")
