import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {device}')

# Possible paths for train, val, and test directories
possible_paths = {
    "train": [
        r"../../data/enel645_2024f/garbage_data/CVPR_2024_dataset_Train",
        r"/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Train"
    ],
    "val": [
        r"../../data/enel645_2024f/garbage_data/CVPR_2024_dataset_Val",
        r"/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Val"
    ],
    "test": [
        r"../../data/enel645_2024f/garbage_data/CVPR_2024_dataset_Test",
        r"/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Test"
    ]
}

# Function to automatically detect and return the correct directory path
def get_data_directory(data_type):
    for path in possible_paths[data_type]:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"None of the paths for {data_type} directory exist!")

# Get the correct paths
train_dir = get_data_directory("train")
val_dir = get_data_directory("val")
test_dir = get_data_directory("test")

# List all images in a directory
def list_images_in_dir(directory, valid_extensions=(".png", ".jpg", ".jpeg")):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Load the model and processor
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

# Function to get captions for a batch of images
def get_batch_captions(image_paths):
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        images.append(image_np)

    prompt = "<CAPTION>"
    inputs = processor(text=[prompt]*len(images), images=images, return_tensors="pt", padding=True).to(device, torch_dtype)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    return generated_texts

# Function to process images from a directory and save results in chunks
def process_and_save_in_batches(directory, output_file, batch_size=4):
    image_paths = list_images_in_dir(directory)
    
    # Initialize the CSV if it doesn't exist
    if not os.path.exists(output_file):
        df = pd.DataFrame(columns=["image", "description"])
        df.to_csv(output_file, index=False)
    
    # Loop through images in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        try:
            captions = get_batch_captions(batch_paths)
            print(f"Processed batch {i // batch_size + 1}/{(len(image_paths) + batch_size - 1) // batch_size}")
            
            # Prepare new entries to be saved
            new_entries = [{"image": path, "description": caption} for path, caption in zip(batch_paths, captions)]
            
            # Save batch to CSV
            new_df = pd.DataFrame(new_entries)
            new_df.to_csv(output_file, mode='a', header=False, index=False)
            
        except Exception as e:
            print(f"Error processing batch starting with {batch_paths[0]}: {e}")

# Define output files
train_output = 'train_image_descriptions.csv'
val_output = 'val_image_descriptions.csv'
test_output = 'test_image_descriptions.csv'

# Process each directory and save the results in different CSV files
process_and_save_in_batches(train_dir, train_output)
process_and_save_in_batches(val_dir, val_output)
process_and_save_in_batches(test_dir, test_output)
