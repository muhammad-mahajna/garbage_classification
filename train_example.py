# %%
# Imports and Env 
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import re
from torch.nn.utils.rnn import pad_sequence
import torch.nn.utils.rnn as rnn_utils
import numpy as np
# Check if CUDA or MPS (for local training on laptop) is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use CUDA for NVIDIA GPU
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Use Apple's MPS
    print("Using Apple MPS (Metal Performance Shaders)")
else:
    device = torch.device("cpu")  # Fallback to CPU
    print("Using CPU")


# %%
# Preprocessing
# Preprocessing for the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize all images to 128x128
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Vocabulary and tokenization for text processing
class Vocabulary:
    def __init__(self):
        self.word2idx = {"<unk>": 0}  # Initialize with "<unk>" token
        self.idx2word = {0: "<unk>"}
        self.idx = 1  # Start indexing from 1 since 0 is for "<unk>"

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx["<unk>"])  # Return "<unk>" if word not found

    def __len__(self):
        return len(self.word2idx)

# Tokenize text from filenames
def tokenize_text(text):
    return text.lower().split()


def collate_fn(batch):
    # Separate images, texts, and labels from the batch
    images, texts, labels = zip(*batch)

    # Stack images and labels as usual
    images = torch.stack(images, 0)
    labels = torch.tensor(labels)

    # Pad the text sequences to the same length
    lengths = [len(text) for text in texts]
    padded_texts = rnn_utils.pad_sequence(texts, batch_first=True, padding_value=0)  # Pad with 0 (can use <PAD> token)
    
    return images, padded_texts, labels

# %%
# Custom Dataset class for both image and text input
class CustomImageTextDataset(Dataset):
    def __init__(self, root_dir, transform=None, vocab=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = ['Black', 'Blue', 'Green', 'TTR']
        self.vocab = vocab
        
        # Collect all image file paths and corresponding labels from subfolders
        self.image_paths = []
        self.labels = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(self.root_dir, class_name)
            for img_file in os.listdir(class_dir):
                if img_file.endswith('.jpg') or img_file.endswith('.jpeg') or img_file.endswith('.png'):
                    self.image_paths.append(os.path.join(class_dir, img_file))
                    self.labels.append(self.class_names.index(class_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load the image
        image = Image.open(img_path).convert('RGB')

        # Extract the text from the filename (if needed)
        img_name = os.path.basename(img_path)
        label_text = re.sub(r'\d+', '', img_name.split('.')[0]).strip().lower()
        text_tokens = tokenize_text(label_text)

        # Convert text tokens to indices using the vocabulary
        text_indices = [self.vocab(token) for token in text_tokens]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(text_indices), label

# Define paths for datasets
train_data_path = '../../data/enel645_2024f/garbage_data/CVPR_2024_dataset_Train'
val_data_path = '../../data/enel645_2024f/garbage_data/CVPR_2024_dataset_Val'
test_data_path = '../../data/enel645_2024f/garbage_data/CVPR_2024_dataset_Test'

# Define vocabulary and build from text labels
vocab = Vocabulary()

# Iterate through dataset directories and add words to the vocabulary
for dataset_dir in [train_data_path, val_data_path, test_data_path]:
    for file_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, file_name)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                label_text = re.sub(r'\d+', '', img_file.split('.')[0]).strip().lower()
                for token in tokenize_text(label_text):
                    vocab.add_word(token)


# Load datasets with custom dataset class
train_dataset = CustomImageTextDataset(root_dir=train_data_path, transform=transform, vocab=vocab)
val_dataset = CustomImageTextDataset(root_dir=val_data_path, transform=transform, vocab=vocab)
test_dataset = CustomImageTextDataset(root_dir=test_data_path, transform=transform, vocab=vocab)

# Check if dataset contains valid data
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

# Create DataLoaders with custom collate_fn
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Print class to index mapping
print(f'Class to index mapping: {train_dataset.class_names}')


# %%
# Deep learning model that processes both images and text
class ImageTextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(ImageTextCNN, self).__init__()
        
        # CNN for images
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layer for image features
        self.fc_img = nn.Linear(64 * 32 * 32, 128)
        
        # Text model (Embedding + LSTM)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc_text = nn.Linear(hidden_dim, 128)

        # Combined layer
        self.fc_combined = nn.Linear(256, num_classes)

    def forward(self, image, text):
        # Process the image through CNN
        x_img = self.pool(torch.relu(self.conv1(image)))
        x_img = self.pool(torch.relu(self.conv2(x_img)))
        x_img = x_img.view(-1, 64 * 32 * 32)  # Flatten image features
        x_img = torch.relu(self.fc_img(x_img))

        # Process the text through Embedding and LSTM
        embedded_text = self.embedding(text)
        packed_embedded = rnn_utils.pack_padded_sequence(embedded_text, lengths=[len(t) for t in text], batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_embedded)
        lstm_out, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)
        x_text = torch.relu(self.fc_text(lstm_out[:, -1, :]))  # Use the last output of LSTM

        # Combine image and text features
        x_combined = torch.cat((x_img, x_text), dim=1)
        output = self.fc_combined(x_combined)
        return output
    

# %%
# Set parameters
embedding_dim = 50
hidden_dim = 64
num_classes = 4
vocab_size = len(vocab)

# Instantiate the model
model = ImageTextCNN(vocab_size, embedding_dim, hidden_dim, num_classes)

# Move the model to the selected device
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training, Validation, and Testing loop
num_epochs = 5

for epoch in range(num_epochs):
    # Training phase
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for images, text, labels in train_loader:
        images, text, labels = images.to(device), text.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, text)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss/len(train_loader):.4f}')

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation for validation
        for images, text, labels in val_loader:
            images, text, labels = images.to(device), text.to(device), labels.to(device)
            outputs = model(images, text)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f'Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%')

# Testing phase
model.eval()  # Set the model to evaluation mode
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():  # Disable gradient calculation for testing
    for images, text, labels in test_loader:
        images, text, labels = images.to(device), text.to(device), labels.to(device)
        outputs = model(images, text)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%')


# %%



