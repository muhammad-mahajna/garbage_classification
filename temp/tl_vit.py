# %%
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor

# Set up directories
# Load data
train_dir = '/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Train'
val_dir = '/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Val'
test_dir = '/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Test'

# Image transformations: Resize, normalize, and augmentations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to ViT input size
    transforms.ToTensor(),          # Convert image to Tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize the images
])

# Load the datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load pre-trained ViT model from Hugging Face and fine-tune for 4-class classification
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=4)

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up loss function and optimizer
criterion = CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
epochs = 3  # You can adjust this depending on your data and time
for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct_predictions = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels)
        running_loss += loss.item()
    
    # Calculate average loss and accuracy over the training dataset
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct_predictions.double() / len(train_dataset)
    
    print(f'Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}')
    
    # Validation loop
    model.eval()  # Set model to evaluation mode
    correct_val_predictions = 0
    val_running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images).logits
            val_loss = criterion(outputs, labels)
            val_running_loss += val_loss.item()
            
            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            correct_val_predictions += torch.sum(preds == labels)
    
    # Calculate average loss and accuracy over the validation dataset
    val_loss = val_running_loss / len(val_loader)
    val_acc = correct_val_predictions.double() / len(val_dataset)
    
    print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

# Save the model after training
model.save_pretrained('./garbage_classifier_model')

# Test the model (optional)
model.eval()  # Set model to evaluation mode
correct_test_predictions = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images).logits
        _, preds = torch.max(outputs, 1)
        correct_test_predictions += torch.sum(preds == labels)

test_acc = correct_test_predictions.double() / len(test_dataset)
print(f'Test Accuracy: {test_acc:.4f}')



