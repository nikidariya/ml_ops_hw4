import boto3
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from botocore.client import Config
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json

BASE_DIR = '/mnt/d/tracking_homework_04/ai-vs-human-generated-dataset-hw'
BUCKET = 'models'
MODEL_IN = 'model_v2.pth'
BATCH_SIZE = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

s3 = boto3.client(
    's3',
    endpoint_url='http://localhost:9000',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin',
    config=Config(signature_version='s3v4')
)

print('Скачивание model_v2.pth из MinIO')
s3.download_file(BUCKET, MODEL_IN, 'model_v2.pth')

class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, folder_override=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.folder_override = folder_override

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name = self.data.iloc[idx]['file_name']
        if self.folder_override:
            file_name = self.folder_override + '/' + file_name.split('/')[-1]
        img_path = self.root_dir / file_name
        image = Image.open(img_path).convert('RGB')
        label = int(self.data.iloc[idx]['label'])
        if self.transform:
            image = self.transform(image)
        return image, label

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = ImageDataset(
    csv_file=f'{BASE_DIR}/Test_2/test.csv',
    root_dir=f'{BASE_DIR}/Test_2',
    transform=test_transform,
    folder_override='test_data'
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
print(f'Test_2 dataset size: {len(test_dataset)}')

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('model_v2.pth', map_location=device))
model = model.to(device)

model.eval()
running_loss = 0.0
all_preds = []
all_labels = []
criterion = nn.CrossEntropyLoss()

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc='Testing'):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss = running_loss / len(test_loader.dataset)
test_acc = accuracy_score(all_labels, all_preds)
test_f1 = f1_score(all_labels, all_preds, average='weighted')
test_precision = precision_score(all_labels, all_preds, average='weighted')
test_recall = recall_score(all_labels, all_preds, average='weighted')

print(f'Test Loss:      {test_loss:.4f}')
print(f'Test Accuracy:  {test_acc:.4f}')
print(f'Test F1:        {test_f1:.4f}')
print(f'Test Precision: {test_precision:.4f}')
print(f'Test Recall:    {test_recall:.4f}')

writer = SummaryWriter(log_dir='runs/test_v2')
writer.add_scalar('Loss/test_v2', test_loss, 0)
writer.add_scalar('Accuracy/test_v2', test_acc, 0)
writer.add_scalar('F1/test_v2', test_f1, 0)
writer.add_scalar('Precision/test_v2', test_precision, 0)
writer.add_scalar('Recall/test_v2', test_recall, 0)
writer.close()

metrics = {
    'test_loss': test_loss,
    'test_acc': test_acc,
    'test_f1': test_f1,
    'test_precision': test_precision,
    'test_recall': test_recall
}
with open('metrics_v2.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print('Метрики сохранены в metrics_v2.json')
