import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from transformers import BertModel, BertTokenizer
from PIL import Image
import numpy as np
import re
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoProcessor, AutoModelForCausalLM
import requests

def list_images_in_dir(directory, valid_extensions=(".png", ".jpg", ".jpeg")):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {device}')


train_dir = '/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Train'
val_dir = '/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Val'
test_dir = '/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Test'

# List all images in the train directory
image_paths = list_images_in_dir(train_dir)

from transformers import AutoProcessor, AutoModelForCausalLM
import requests

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

def get_image_caption(image):
    image = np.array(image.convert("RGB"))
    prompt = "<CAPTION>"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(image.shape[1], image.shape[0]))

    return parsed_answer

descriptions = []

for image_path in image_paths:
    
    parsed_answer = get_image_caption(Image.open(image_path))
    descriptions.append({
        "image": image_path,
        "description": parsed_answer
    })
    print(parsed_answer)
    
# Save descriptions to CSV
import pandas as pd
df = pd.DataFrame(descriptions)
df.to_csv('image_descriptions.csv', index=False)
