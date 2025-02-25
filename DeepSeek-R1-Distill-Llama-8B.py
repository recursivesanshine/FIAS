from transformers import pipeline

# Define the pipeline
pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

# Define the messages
messages = [
    {"role": "user", "content": "Who are you?"},
]

# Use the pipeline to generate a response with a larger max_length
response = pipe(messages, max_length=100)

# Print the generated response
print(response)
