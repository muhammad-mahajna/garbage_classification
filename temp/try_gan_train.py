import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os

# Function to detect the correct base directory for the dataset
def detect_base_directory():
    possible_dirs = [
        r"/work/TALC/enel645_2024f/garbage_data",  # Directory on TALC cluster
        r"../../data/enel645_2024f/garbage_data"   # Directory on LAPTOP - relative path
    ]

    for base_dir in possible_dirs:
        if os.path.exists(base_dir):
            print(f"Using base directory: {base_dir}")
            return base_dir

    # Raise an error if no valid data directory is found
    raise ValueError("No valid base directory found.")

# Detect the base directory
base_dir = detect_base_directory()

# Define paths for training, validation, and testing datasets
train_dir = os.path.join(base_dir, "CVPR_2024_dataset_Train")

# Hyperparameters
image_size = 128  # Set resolution to 128x128
batch_size = 64  # Reduce batch size for larger images
z_dim = 100
learning_rate = 2e-4
num_epochs = 100
checkpoint_path = "checkpoint.pth"  # Path to save/load checkpoint

# Device selection with MPS support
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {device}')

# Create a directory to save generated images
os.makedirs("generated_images", exist_ok=True)

# Image transformations (resize, normalize)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Scale images to [-1, 1]
])

# Dataset and DataLoader
dataset = datasets.ImageFolder(root=train_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the Generator with increased capacity
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 1024, 4, 1, 0, bias=False),  # Larger model capacity
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Define the Discriminator with increased capacity
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Initialize the models
gen = Generator().to(device)
disc = Discriminator().to(device)

# Loss and Optimizers
criterion = nn.BCELoss()
optimizer_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Helper function to create noise vectors
def generate_noise(batch_size, z_dim, device):
    return torch.randn(batch_size, z_dim, 1, 1).to(device)

# Load checkpoint if it exists
start_epoch = 0
if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    gen.load_state_dict(checkpoint["gen_state_dict"])
    disc.load_state_dict(checkpoint["disc_state_dict"])
    optimizer_gen.load_state_dict(checkpoint["optimizer_gen_state_dict"])
    optimizer_disc.load_state_dict(checkpoint["optimizer_disc_state_dict"])
    start_epoch = checkpoint["epoch"] + 1  # Continue from the next epoch
    print(f"Resuming from epoch {start_epoch}")

# Training loop
for epoch in range(start_epoch, num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        # Create labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        optimizer_disc.zero_grad()

        # Forward real images
        real_output = disc(real_images).view(-1, 1)
        real_loss = criterion(real_output, real_labels)

        # Generate fake images
        noise = generate_noise(batch_size, z_dim, device)
        fake_images = gen(noise)
        fake_output = disc(fake_images.detach()).view(-1, 1)
        fake_loss = criterion(fake_output, fake_labels)

        # Backward pass and update
        disc_loss = real_loss + fake_loss
        disc_loss.backward()
        optimizer_disc.step()

        # Train Generator
        optimizer_gen.zero_grad()

        # Forward fake images
        output = disc(fake_images).view(-1, 1)
        gen_loss = criterion(output, real_labels)

        # Backward pass and update
        gen_loss.backward()
        optimizer_gen.step()

    print(f"Epoch [{epoch}/{num_epochs}] | D Loss: {disc_loss.item()} | G Loss: {gen_loss.item()}")

    # Save generated images periodically
    if epoch % 10 == 0:
        save_image(fake_images[:25], f"generated_images/generated_epoch_{epoch}.png", nrow=5, normalize=True)
    
    # Save checkpoint
    checkpoint = {
        "epoch": epoch,
        "gen_state_dict": gen.state_dict(),
        "disc_state_dict": disc.state_dict(),
        "optimizer_gen_state_dict": optimizer_gen.state_dict(),
        "optimizer_disc_state_dict": optimizer_disc.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)

# Save the final models
torch.save(gen.state_dict(), 'generator_final.pth')
torch.save(disc.state_dict(), 'discriminator_final.pth')
