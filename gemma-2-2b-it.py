import torch
from transformers import pipeline

# Set the device to CPU
device = torch.device('cpu')
pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=-1  # Use CPU
)

messages = [
    {"role": "user", "content": "Tell me who you are in 125 characters."},
]

# Adjusted code to handle the outputs correctly
outputs = pipe(messages, max_new_tokens=256)
assistant_response = outputs[0]["generated_text"]
print(assistant_response)
