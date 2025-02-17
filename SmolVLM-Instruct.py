import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import csv

# Set the GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load image from your device
image_path = "/home/alexsanlenz/fias/subsample100_seed=42/images/oai_DE-MUS-048017_703_johannes-verspronck-portrait-woman-armchair-703--thumb-xl.jpg"  # Replace with the path to your image
image1 = Image.open(image_path)

# Extract image filename without extension
image_filename = os.path.splitext(os.path.basename(image_path))[0]

# Initialize processor and model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",
    torch_dtype=torch.float32,
).to(DEVICE)

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe the image with details and give key words to describe the motifs."} #, then the picture elements, then the association, then the atmosphere, then the emtions?"}
        ]
    },
]

# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image1], return_tensors="pt")
inputs = inputs.to(DEVICE)

# Generate outputs
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

# Capture the output text
output_text = generated_texts[0]

# Debugging print statement to ensure the text is captured
print("Generated Text:", output_text)

# Save to CSV file named after the image filename with "output" added
csv_file_path = f"{image_filename}_output.csv"  # Create the CSV filename
try:
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Description"])
        writer.writerow([output_text])
    print(f"Output successfully saved to {csv_file_path}")
except Exception as e:
    print(f"Error saving output to CSV file: {e}")
