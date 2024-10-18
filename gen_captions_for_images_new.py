import os
import torch
from PIL import Image
import pandas as pd
from transformers import BlipProcessor, BlipForConditionalGeneration

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {device}')

# Directories
train_dir = '/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Train'

# List all images in the train directory
def list_images_in_dir(directory, valid_extensions=(".png", ".jpg", ".jpeg")):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths

image_paths = list_images_in_dir(train_dir)

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Caption generation function
def get_image_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Output CSV file
output_file = 'image_descriptions.csv'

# Initialize the CSV if it doesn't exist
if not os.path.exists(output_file):
    df = pd.DataFrame(columns=["image", "description"])
    df.to_csv(output_file, index=False)

# Loop through images and get captions, saving as we go
for image_path in image_paths:
    try:
        caption = get_image_caption(image_path)
        print(f"Image: {image_path} - Caption: {caption}")
        
        # Save to CSV immediately
        new_entry = pd.DataFrame([{"image": image_path, "description": caption}])
        new_entry.to_csv(output_file, mode='a', header=False, index=False)
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
