import boto3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from botocore.client import Config
from sklearn.metrics import f1_score, accuracy_score
from pathlib import Path
from tqdm import tqdm
import pandas as pd

BASE_DIR = '/mnt/d/tracking_homework_04/ai-vs-human-generated-dataset-hw'
BUCKET = 'models'
MODEL_IN = 'model_v1.pth'
MODEL_OUT = 'model_v2.pth'
BATCH_SIZE = 32
NUM_EPOCHS = 5
LR = 0.0001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

s3 = boto3.client(
    's3',
    endpoint_url='http://localhost:9000',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin',
    config=Config(signature_version='s3v4')
)

print('Скачивание model_v1.pth из MinIO')
s3.download_file(BUCKET, MODEL_IN, 'model_v1.pth')
print('Скачали!')

class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name = self.data.iloc[idx]['file_name']
        img_path = self.root_dir / file_name
        image = Image.open(img_path).convert('RGB')
        label = int(self.data.iloc[idx]['label'])
        if self.transform:
            image = self.transform(image)
        return image, label

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageDataset(
    csv_file=f'{BASE_DIR}/Train_2/train.csv',
    root_dir=f'{BASE_DIR}/Train_2',
    transform=train_transform
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
print(f'Train_2 dataset size: {len(train_dataset)}')

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('model_v1.pth', map_location=device))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

writer = SummaryWriter(log_dir='runs/finetune')
writer.add_text('hyperparameters',
    f'lr={LR}, batch_size={BATCH_SIZE}, epochs={NUM_EPOCHS}, optimizer=Adam')

print('Дообучение')
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

    writer.add_scalar('Loss/finetune', epoch_loss, epoch)
    writer.add_scalar('Accuracy/finetune', epoch_acc, epoch)
    writer.add_scalar('F1/finetune', epoch_f1, epoch)

    print(f'Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}, F1={epoch_f1:.4f}')

writer.close()

torch.save(model.state_dict(), 'model_v2.pth')
s3.upload_file('model_v2.pth', BUCKET, MODEL_OUT)
print('model_v2.pth загружена в MinIO')
