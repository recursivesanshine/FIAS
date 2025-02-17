import sys
import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import csv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Initialize processor and model
try:
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
    model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceTB/SmolVLM-Instruct",
        torch_dtype=torch.float32,
    ).to(device)
except Exception as e:
    logging.error(f"Error loading model or processor: {e}")
    exit()

# Function to process a single image
def process_image(image_path):
    try:
        image = Image.open(image_path)
    except Exception as e:
        logging.error(f"Error loading image: {e}")
        return

    # Extract image filename without extension
    image_filename = os.path.splitext(os.path.basename(image_path))[0]

    # Create input messages
    user_prompt = "Provide a detailed description of the image using exactly 128 characters. Keywords for Atmosphere: [List of relevant keywords here], Keywords for Emotion: [List of relevant keywords here], Picture Elements: [List of relevant elements here]"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt}
            ]
        },
    ]

    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = inputs.to(device)

    # Generate outputs
    try:
        generated_ids = model.generate(**inputs, max_new_tokens=500)
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        output_text = generated_texts[0]
        logging.info("Generated Text for %s: %s", image_filename, output_text)
    except Exception as e:
        logging.error(f"Error generating text for %s: %s", image_filename, e)
        return

    # Save to CSV file named after the image filename with "output" added
    csv_file_path = f"{image_filename}_output.csv"
    try:
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Description"])
            writer.writerow([output_text])
        logging.info(f"Output successfully saved to {csv_file_path}")
    except Exception as e:
        logging.error(f"Error saving output to CSV file for %s: %s", image_filename, e)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error("Usage: python script_name.py image_path_or_folder")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    if os.path.isdir(input_path):
        # Process all image files in the specified folder
        image_paths = [os.path.join(input_path, file) for file in os.listdir(input_path) if file.endswith(('.jpg', '.jpeg', '.png'))]
        for image_path in image_paths:
            process_image(image_path)
    elif os.path.isfile(input_path) and input_path.endswith(('.jpg', '.jpeg', '.png')):
        # Process a single image file
        process_image(input_path)
    else:
        logging.error("Invalid input path. Please provide a path to a directory or an image file.")
