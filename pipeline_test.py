from transformers import pipeline
from PIL import Image
import torch
torch.cuda.empty_cache()


# Check if a GPU is available and set the device index to GPU 1
device = 0 if torch.cuda.is_available() else -1

# Load the image from your device
image_path = "/home/alexsanlenz/fias/subsample100_seed=42/images/oai_DE-MUS-048017_703_johannes-verspronck-portrait-woman-armchair-703--thumb-xl.jpg"
image = Image.open(image_path)

pipe = pipeline("image-text-to-text", model="llava-hf/llava-1.5-7b-hf", device=device, use_fp16=True)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What is the sentiment of the image?"},
        ],
    },
]

out = pipe(text=messages, padding=True, max_new_tokens=100)
print(out)
