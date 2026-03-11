# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-03-11T12:00:48.994316Z","iopub.execute_input":"2026-03-11T12:00:48.994524Z","iopub.status.idle":"2026-03-11T12:00:52.762406Z","shell.execute_reply.started":"2026-03-11T12:00:48.994503Z","shell.execute_reply":"2026-03-11T12:00:52.761569Z"}}
## Importing packages
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import pandas as pd
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from sklearn.metrics import f1_score



# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# STARTS HERE

# %% [code] {"execution":{"iopub.status.busy":"2026-03-11T12:00:52.763711Z","iopub.execute_input":"2026-03-11T12:00:52.764159Z","iopub.status.idle":"2026-03-11T12:00:52.769426Z","shell.execute_reply.started":"2026-03-11T12:00:52.764132Z","shell.execute_reply":"2026-03-11T12:00:52.768744Z"},"jupyter":{"outputs_hidden":false}}
# Data Augmentation 
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

# %% [code] {"execution":{"iopub.status.busy":"2026-03-11T12:00:52.770496Z","iopub.execute_input":"2026-03-11T12:00:52.770796Z","iopub.status.idle":"2026-03-11T12:00:56.884419Z","shell.execute_reply.started":"2026-03-11T12:00:52.770767Z","shell.execute_reply":"2026-03-11T12:00:56.883867Z"},"jupyter":{"outputs_hidden":false}}
# Loading the labeled data and splitting it 
full_train_dataset = datasets.ImageFolder('/kaggle/input/datasets/imsparsh/flowers-dataset/train', transform=train_transforms)

train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_data, val_data = random_split(full_train_dataset, [train_size, val_size])

# Re-apply strict transforms to validation set
val_data.dataset.transform = test_transforms

trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
valloader = DataLoader(val_data, batch_size=64)

# Custom Class for the "Bunch of Images" (Test Folder)
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

# Loading the unlabeled test images
test_data = FlowerTestDataset('/kaggle/input/datasets/imsparsh/flowers-dataset/test', transform=test_transforms)
testloader = DataLoader(test_data, batch_size=64, shuffle=False)

# %% [code] {"execution":{"iopub.status.busy":"2026-03-11T12:00:56.886155Z","iopub.execute_input":"2026-03-11T12:00:56.886878Z","iopub.status.idle":"2026-03-11T12:00:57.079131Z","shell.execute_reply.started":"2026-03-11T12:00:56.886849Z","shell.execute_reply":"2026-03-11T12:00:57.078524Z"},"jupyter":{"outputs_hidden":false}}

class FlowerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers 
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers 
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 5) 
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Adding sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flattening image for the linear layers
        x = x.view(-1, 64 * 28 * 28)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = FlowerClassifier()

# %% [code] {"execution":{"iopub.status.busy":"2026-03-11T12:00:57.080117Z","iopub.execute_input":"2026-03-11T12:00:57.080395Z","iopub.status.idle":"2026-03-11T12:00:57.575714Z","shell.execute_reply.started":"2026-03-11T12:00:57.080370Z","shell.execute_reply":"2026-03-11T12:00:57.575155Z"},"jupyter":{"outputs_hidden":false}}
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# %% [code] {"execution":{"iopub.status.busy":"2026-03-11T12:00:57.576625Z","iopub.execute_input":"2026-03-11T12:00:57.576949Z","iopub.status.idle":"2026-03-11T12:03:35.371224Z","shell.execute_reply.started":"2026-03-11T12:00:57.576914Z","shell.execute_reply":"2026-03-11T12:03:35.370517Z"},"jupyter":{"outputs_hidden":false}}
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
    
    # Validation step to get score
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

# %% [code] {"execution":{"iopub.status.busy":"2026-03-11T12:03:35.372215Z","iopub.execute_input":"2026-03-11T12:03:35.372521Z","iopub.status.idle":"2026-03-11T12:03:44.026965Z","shell.execute_reply.started":"2026-03-11T12:03:35.372495Z","shell.execute_reply":"2026-03-11T12:03:44.026323Z"},"jupyter":{"outputs_hidden":false}}
#Calculating F1 Macro on Validation Set 
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

# Calculating the F1 Macro
final_f1 = f1_score(all_labels, all_preds, average='macro')
print(f"Final F1 Macro Score: {final_f1:.4f}")

# Generating submission.csv for Test Images 
test_results = []
# Get the class names from the underlying dataset
class_names = train_data.dataset.classes 

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

# Saving the CSV file for GitHub
df = pd.DataFrame(test_results)
#df.to_csv('submission.csv', index=False)
print("submission.csv created!")

# %% [code] {"execution":{"iopub.status.busy":"2026-03-11T12:15:09.441123Z","iopub.execute_input":"2026-03-11T12:15:09.441932Z","iopub.status.idle":"2026-03-11T12:15:09.447493Z","shell.execute_reply.started":"2026-03-11T12:15:09.441882Z","shell.execute_reply":"2026-03-11T12:15:09.446866Z"}}
df.to_csv('2_submission.csv', index=False)

# %% [code] {"execution":{"iopub.status.busy":"2026-03-11T12:03:44.027865Z","iopub.execute_input":"2026-03-11T12:03:44.028178Z","iopub.status.idle":"2026-03-11T12:03:44.032289Z","shell.execute_reply.started":"2026-03-11T12:03:44.028145Z","shell.execute_reply":"2026-03-11T12:03:44.031699Z"},"jupyter":{"outputs_hidden":false}}
# This saves the trained weights into a file named checkpoint.pth
#torch.save(model.state_dict(), 'checkpoint.pth')
print("Model saved to /kaggle/working/checkpoint.pth")

# %% [code] {"execution":{"iopub.status.busy":"2026-03-11T12:03:44.033190Z","iopub.execute_input":"2026-03-11T12:03:44.033480Z","iopub.status.idle":"2026-03-11T12:03:44.042718Z","shell.execute_reply.started":"2026-03-11T12:03:44.033447Z","shell.execute_reply":"2026-03-11T12:03:44.042070Z"},"jupyter":{"outputs_hidden":false}}
# This converts your current notebook to a script file in the /kaggle/working directory
#!jupyter nbconvert --to script __notebook__.ipynb --output train# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-03-11T12:00:48.994316Z","iopub.execute_input":"2026-03-11T12:00:48.994524Z","iopub.status.idle":"2026-03-11T12:00:52.762406Z","shell.execute_reply.started":"2026-03-11T12:00:48.994503Z","shell.execute_reply":"2026-03-11T12:00:52.761569Z"}}
## Importing packages
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import pandas as pd
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from sklearn.metrics import f1_score



# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# STARTS HERE

# %% [code] {"execution":{"iopub.status.busy":"2026-03-11T12:00:52.763711Z","iopub.execute_input":"2026-03-11T12:00:52.764159Z","iopub.status.idle":"2026-03-11T12:00:52.769426Z","shell.execute_reply.started":"2026-03-11T12:00:52.764132Z","shell.execute_reply":"2026-03-11T12:00:52.768744Z"},"jupyter":{"outputs_hidden":false}}
# Data Augmentation 
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

# %% [code] {"execution":{"iopub.status.busy":"2026-03-11T12:00:52.770496Z","iopub.execute_input":"2026-03-11T12:00:52.770796Z","iopub.status.idle":"2026-03-11T12:00:56.884419Z","shell.execute_reply.started":"2026-03-11T12:00:52.770767Z","shell.execute_reply":"2026-03-11T12:00:56.883867Z"},"jupyter":{"outputs_hidden":false}}
# Loading the labeled data and splitting it 
full_train_dataset = datasets.ImageFolder('/kaggle/input/datasets/imsparsh/flowers-dataset/train', transform=train_transforms)

train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_data, val_data = random_split(full_train_dataset, [train_size, val_size])

# Re-apply strict transforms to validation set
val_data.dataset.transform = test_transforms

trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
valloader = DataLoader(val_data, batch_size=64)

# Custom Class for the "Bunch of Images" (Test Folder)
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

# Loading the unlabeled test images
test_data = FlowerTestDataset('/kaggle/input/datasets/imsparsh/flowers-dataset/test', transform=test_transforms)
testloader = DataLoader(test_data, batch_size=64, shuffle=False)

# %% [code] {"execution":{"iopub.status.busy":"2026-03-11T12:00:56.886155Z","iopub.execute_input":"2026-03-11T12:00:56.886878Z","iopub.status.idle":"2026-03-11T12:00:57.079131Z","shell.execute_reply.started":"2026-03-11T12:00:56.886849Z","shell.execute_reply":"2026-03-11T12:00:57.078524Z"},"jupyter":{"outputs_hidden":false}}

class FlowerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers 
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers 
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 5) 
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Adding sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flattening image for the linear layers
        x = x.view(-1, 64 * 28 * 28)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = FlowerClassifier()

# %% [code] {"execution":{"iopub.status.busy":"2026-03-11T12:00:57.080117Z","iopub.execute_input":"2026-03-11T12:00:57.080395Z","iopub.status.idle":"2026-03-11T12:00:57.575714Z","shell.execute_reply.started":"2026-03-11T12:00:57.080370Z","shell.execute_reply":"2026-03-11T12:00:57.575155Z"},"jupyter":{"outputs_hidden":false}}
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# %% [code] {"execution":{"iopub.status.busy":"2026-03-11T12:00:57.576625Z","iopub.execute_input":"2026-03-11T12:00:57.576949Z","iopub.status.idle":"2026-03-11T12:03:35.371224Z","shell.execute_reply.started":"2026-03-11T12:00:57.576914Z","shell.execute_reply":"2026-03-11T12:03:35.370517Z"},"jupyter":{"outputs_hidden":false}}
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
    
    # Validation step to get score
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

# %% [code] {"execution":{"iopub.status.busy":"2026-03-11T12:03:35.372215Z","iopub.execute_input":"2026-03-11T12:03:35.372521Z","iopub.status.idle":"2026-03-11T12:03:44.026965Z","shell.execute_reply.started":"2026-03-11T12:03:35.372495Z","shell.execute_reply":"2026-03-11T12:03:44.026323Z"},"jupyter":{"outputs_hidden":false}}
#Calculating F1 Macro on Validation Set 
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

# Calculating the F1 Macro
final_f1 = f1_score(all_labels, all_preds, average='macro')
print(f"Final F1 Macro Score: {final_f1:.4f}")

# Generating submission.csv for Test Images 
test_results = []
# Get the class names from the underlying dataset
class_names = train_data.dataset.classes 

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

# Saving the CSV file for GitHub
df = pd.DataFrame(test_results)
df.to_csv('2_submission.csv', index=False)
print("submission.csv created!")



# %% [code] {"execution":{"iopub.status.busy":"2026-03-11T12:03:44.027865Z","iopub.execute_input":"2026-03-11T12:03:44.028178Z","iopub.status.idle":"2026-03-11T12:03:44.032289Z","shell.execute_reply.started":"2026-03-11T12:03:44.028145Z","shell.execute_reply":"2026-03-11T12:03:44.031699Z"},"jupyter":{"outputs_hidden":false}}
# This saves the trained weights into a file named checkpoint.pth
torch.save(model.state_dict(), 'checkpoint.pth')
print("Model saved to /kaggle/working/checkpoint.pth")

# %% [code] {"execution":{"iopub.status.busy":"2026-03-11T12:03:44.033190Z","iopub.execute_input":"2026-03-11T12:03:44.033480Z","iopub.status.idle":"2026-03-11T12:03:44.042718Z","shell.execute_reply.started":"2026-03-11T12:03:44.033447Z","shell.execute_reply":"2026-03-11T12:03:44.042070Z"},"jupyter":{"outputs_hidden":false}}
# This converts your current notebook to a script file in the /kaggle/working directory
!jupyter nbconvert --to script __notebook__.ipynb --output train
