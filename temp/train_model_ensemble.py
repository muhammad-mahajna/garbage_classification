import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from transformers import ViTModel, DistilBertModel, ViTFeatureExtractor, DistilBertTokenizer
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {device}')

# Define class labels
label_map = {"Black": 0, "Blue": 1, "Green": 2, "TTR": 3}
print("Class labels:", label_map)

# Initialize transformers for ViT and DistilBERT
print("Loading Vision Transformer and DistilBERT models...")
vit_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
print("Models loaded successfully.")

# Function to detect base directory
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

base_dir = detect_base_directory()

# Define paths for training, validation, and testing datasets
train_dir = os.path.join(base_dir, "CVPR_2024_dataset_Train")
val_dir = os.path.join(base_dir, "CVPR_2024_dataset_Val")
test_dir = os.path.join(base_dir, "CVPR_2024_dataset_Test")

# Function to load data
def load_data(data_dir, label_map):
    print(f"Loading data from {data_dir}...")
    image_paths, texts, labels = [], [], []
    for label_name, label_idx in label_map.items():
        folder_path = os.path.join(data_dir, label_name)
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(folder_path, filename))
                texts.append(filename.split('_')[0])  # Extract text from filename
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

# Load training and validation data
print("Preparing data loaders...")
train_image_paths, train_texts, train_labels = load_data(train_dir, label_map)
val_image_paths, val_texts, val_labels = load_data(val_dir, label_map)

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

# DataLoader setup
image_transform = get_vit_image_transforms()
train_dataset = MultimodalDataset(train_image_paths, train_texts, train_labels, transform=image_transform, tokenizer=distilbert_tokenizer)
val_dataset = MultimodalDataset(val_image_paths, val_texts, val_labels, transform=image_transform, tokenizer=distilbert_tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True)
print("Data loaders are ready.")

# Define ResNet-based model
class ResNetMultimodalModel(nn.Module):
    def __init__(self, num_classes):
        super(ResNetMultimodalModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()
        self.resnet_feature_dim = 2048
        
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.distilbert_feature_dim = 768

        self.fc = nn.Sequential(
            nn.Linear(self.resnet_feature_dim + self.distilbert_feature_dim, 1024),
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
        image_features = self.resnet(images)
        text_features = self.distilbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        combined_features = torch.cat((image_features, text_features), dim=1)
        output = self.fc(combined_features)
        return output

# Define ViT-based model
class ViTMultimodalModel(nn.Module):
    def __init__(self, num_classes):
        super(ViTMultimodalModel, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.vit_feature_dim = self.vit.config.hidden_size
        
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.distilbert_feature_dim = 768

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

# Training function
def train_model(model, criterion, optimizer, train_loader, val_loader, model_name, num_epochs=5):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        print(f"\nEpoch {epoch + 1}/{num_epochs} for {model_name}")
        for batch in train_loader:
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

        val_loss = 0.0
        model.eval()
        y_true, y_pred = [], []
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
        print(f"{model_name} Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_{model_name.lower()}_model.pth')
            print(f"Best {model_name} model saved with validation loss: {val_loss:.4f}")

# Instantiate and train ResNet model
num_classes = len(label_map)
resnet_model = ResNetMultimodalModel(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(resnet_model.parameters(), lr=1e-5)
train_model(resnet_model, criterion, optimizer, train_loader, val_loader, model_name="ResNet", num_epochs=5)

# Instantiate and train ViT model
vit_model = ViTMultimodalModel(num_classes).to(device)
optimizer = optim.AdamW(vit_model.parameters(), lr=1e-5)
train_model(vit_model, criterion, optimizer, train_loader, val_loader, model_name="ViT", num_epochs=5)

# Evaluation function for individual models
def evaluate_model(model, dataloader, model_name):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images, input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"\n{model_name} Model - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# Load best models for evaluation
resnet_model.load_state_dict(torch.load('best_resnet_model.pth'))
vit_model.load_state_dict(torch.load('best_vit_model.pth'))

# Evaluate individual models
evaluate_model(resnet_model, val_loader, model_name="ResNet")
evaluate_model(vit_model, val_loader, model_name="ViT")

# Define the Ensemble Model
class EnsembleModel(nn.Module):
    def __init__(self, model1, model2, num_classes):
        super(EnsembleModel, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, images, input_ids, attention_mask, weight1, weight2):
        outputs1 = self.model1(images, input_ids, attention_mask)
        outputs2 = self.model2(images, input_ids, attention_mask)
        combined_outputs = weight1 * outputs1 + weight2 * outputs2
        return combined_outputs

# Grid search for best weights
def grid_search_weights(model1, model2, dataloader):
    weights = np.linspace(0, 1, 11)
    best_f1, best_weight1, best_weight2 = 0, 0, 0

    print("Starting grid search for optimal weights...")
    for weight1 in weights:
        weight2 = 1 - weight1
        y_true, y_pred = [], []

        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs1 = model1(images, input_ids, attention_mask)
                outputs2 = model2(images, input_ids, attention_mask)
                combined_outputs = weight1 * outputs1 + weight2 * outputs2
                _, preds = torch.max(combined_outputs, dim=1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        f1 = f1_score(y_true, y_pred, average='weighted')
        if f1 > best_f1:
            best_f1 = f1
            best_weight1, best_weight2 = weight1, weight2
        print(f"Weights - ResNet: {weight1:.2f}, ViT: {weight2:.2f} | F1 Score: {f1:.4f}")

    print(f"\nBest weights - ResNet: {best_weight1:.2f}, ViT: {best_weight2:.2f} | Best F1 Score: {best_f1:.4f}")
    return best_weight1, best_weight2

best_weight1, best_weight2 = grid_search_weights(resnet_model, vit_model, val_loader)

# Load test data and evaluate ensemble model with best weights
test_image_paths, test_texts, test_labels = load_data(test_dir, label_map)
test_dataset = MultimodalDataset(test_image_paths, test_texts, test_labels, transform=image_transform, tokenizer=distilbert_tokenizer)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=4)

# Ensemble evaluation function with best weights
def evaluate_best_ensemble(model1, model2, dataloader, weight1, weight2):
    model1.eval()
    model2.eval()
    y_true, y_pred = [], []
    print("Evaluating ensemble model on test set...")
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs1 = model1(images, input_ids, attention_mask)
            outputs2 = model2(images, input_ids, attention_mask)
            combined_outputs = weight1 * outputs1 + weight2 * outputs2
            _, preds = torch.max(combined_outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"\nEnsemble Model - Optimal Weights | "
          f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# Run final evaluation on test set with best weights
evaluate_best_ensemble(resnet_model, vit_model, test_loader, best_weight1, best_weight2)
