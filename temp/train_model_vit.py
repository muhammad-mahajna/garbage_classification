import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import DistilBertModel, DistilBertTokenizer, ViTModel, ViTFeatureExtractor
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Device configuration with preference order: CUDA, MPS (for Apple Silicon), otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {device}')

# Define class labels and data directory
label_map = {"Black": 0, "Blue": 1, "Green": 2, "TTR": 3}
print("Class labels:", label_map)

# Initialize transformers for ViT and DistilBERT
print("Loading Vision Transformer and DistilBERT models...")
vit_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
print("Models loaded successfully.")

# Dataset class to handle image and text data
class MultimodalDataset(Dataset):
    def __init__(self, image_paths, texts, labels, transform=None, tokenizer=None, max_len=24):
        self.image_paths = image_paths
        self.texts = texts
        self.labels = labels
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        label = self.labels[idx]
        return {
            'image': image,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Function to load data
def load_data(data_dir, label_map):
    print(f"Loading data from {data_dir}...")
    image_paths, texts, labels = [], [], []
    for label_name, label_idx in label_map.items():
        folder_path = os.path.join(data_dir, label_name)
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(folder_path, filename))
                texts.append(filename.split('_')[0])  # Simple text extraction from filename
                labels.append(label_idx)
    print(f"Loaded {len(image_paths)} samples from {data_dir}.")
    return np.array(image_paths), np.array(texts), np.array(labels)

# Data transformation function
def get_vit_image_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=vit_feature_extractor.image_mean, std=vit_feature_extractor.image_std)
    ])

# Function to detect the base directory based on environment
def detect_base_directory():
    possible_dirs = [
        r"/work/TALC/enel645_2024f/garbage_data",
        r"../../data/enel645_2024f/garbage_data"
    ]
    for base_dir in possible_dirs:
        if os.path.exists(base_dir):
            print(f"Using base directory: {base_dir}")
            return base_dir
    raise ValueError("No valid base directory found.")

# Detect and set base directory
base_dir = detect_base_directory()

# Define paths for training, validation, and testing datasets
train_dir = os.path.join(base_dir, "CVPR_2024_dataset_Train")
val_dir = os.path.join(base_dir, "CVPR_2024_dataset_Val")
test_dir = os.path.join(base_dir, "CVPR_2024_dataset_Test")

# Load data
print("Preparing data loaders...")
train_image_paths, train_texts, train_labels = load_data(train_dir, label_map)
val_image_paths, val_texts, val_labels = load_data(val_dir, label_map)

# Create datasets and dataloaders
image_transform = get_vit_image_transforms()
train_dataset = MultimodalDataset(train_image_paths, train_texts, train_labels, transform=image_transform, tokenizer=distilbert_tokenizer)
val_dataset = MultimodalDataset(val_image_paths, val_texts, val_labels, transform=image_transform, tokenizer=distilbert_tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True)
print("Data loaders are ready.")

# Define the multimodal model using ViT for images and DistilBERT for text
class ViTMultimodalModel(nn.Module):
    def __init__(self, num_classes):
        super(ViTMultimodalModel, self).__init__()
        self.vit = vit_model
        self.vit_feature_dim = self.vit.config.hidden_size
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.distilbert_feature_dim = 768

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.vit_feature_dim + self.distilbert_feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        vit_outputs = self.vit(pixel_values=images).last_hidden_state[:, 0, :]
        text_features = self.distilbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        combined_features = torch.cat((vit_outputs, text_features), dim=1)
        output = self.fc(combined_features)
        return output

# Instantiate the model
num_classes = len(label_map)
model = ViTMultimodalModel(num_classes=num_classes).to(device)
print("Model instantiated.")

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

# Training function
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("Training...")
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch [{batch_idx + 1}/{len(train_loader)}]: Loss = {loss.item():.4f}")

        val_loss = 0.0
        model.eval()
        y_true, y_pred = [], []
        print("Validating...")
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(images, input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy = accuracy_score(y_true, y_pred)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_vit_multimodal_model.pth')
            print("Best model saved with validation loss:", val_loss)

# Train the model
print("Starting training...")
train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=5)

# Load the best model for testing
print("Loading best model for testing...")
model.load_state_dict(torch.load('best_vit_multimodal_model.pth'))
model.eval()

# Testing
print("Evaluating on test set...")
test_image_paths, test_texts, test_labels = load_data(test_dir, label_map)
test_dataset = MultimodalDataset(test_image_paths, test_texts, test_labels, transform=image_transform, tokenizer=distilbert_tokenizer)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

y_true, y_pred = [], []
with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(images, input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

        if (batch_idx + 1) % 10 == 0:
            print(f"  Test Batch [{batch_idx + 1}/{len(test_loader)}]")

# Performance metrics
test_accuracy = accuracy_score(y_true, y_pred)
test_precision = precision_score(y_true, y_pred, average='weighted')
test_recall = recall_score(y_true, y_pred, average='weighted')
test_f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Test Results - Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, "
      f"Recall: {test_recall:.4f}, F1 Score: {test_f1:.4f}")

# Confusion Matrix
print("Generating confusion matrix...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_map.keys(), yticklabels=label_map.keys())
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
