import torch
import os
from PIL import Image
from transformers import AutoModelForCausalLM

# Clear CUDA cache
torch.cuda.empty_cache()

# Set environment variable to avoid memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Load model on the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "AIDC-AI/Ovis1.6-Gemma2-9B",
    torch_dtype=torch.bfloat16,
    multimodal_max_length=8192,
    trust_remote_code=True
).to(device)
print("Model loaded.")

text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()

# Function to preprocess and generate output for a batch
def process_batch(image_paths, text):
    batch_input_ids = []
    batch_pixel_values = []
    for image_path in image_paths:
        try:
            image = Image.open(image_path)
            query = f'<image>\n{text}'
            prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image])
            batch_input_ids.append(input_ids.unsqueeze(0))
            batch_pixel_values.append(pixel_values.to(device))
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue

    if not batch_input_ids:
        return "No valid images to process."

    input_ids = torch.cat(batch_input_ids).to(device)
    pixel_values = torch.cat(batch_pixel_values)
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id).to(device)

    # Generate output
    print("Generating output...")
    try:
        with torch.inference_mode(), torch.cuda.amp.autocast():
            gen_kwargs = dict(
                max_new_tokens=16,
                do_sample=False,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=model.generation_config.eos_token_id,
                pad_token_id=text_tokenizer.pad_token_id,
                use_cache=True
            )
            print("Calling model.generate()...")
            output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
            output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
            print(f'Output:\n{output}')
    except Exception as e:
        print(f"Error during generation: {e}")

# Example image paths and text prompt
image_paths = [
    "/fias/subsample100_seed=42/images/oai_DE-MUS-048017_703_johannes-verspronck-portrait-woman-armchair-703--thumb-xl.jpg",
    # Add more image paths as needed
]
text = "Please describe the sentiment image."

# Batch size
batch_size = 1  # Adjust this based on your GPU memory capacity

# Split data into batches
num_samples = len(image_paths)
num_batches = num_samples // batch_size + (num_samples % batch_size > 0)

for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, num_samples)
    batch_image_paths = image_paths[start_idx:end_idx]
    process_batch(batch_image_paths, text)

# Clear CUDA cache again after processing
torch.cuda.empty_cache()
