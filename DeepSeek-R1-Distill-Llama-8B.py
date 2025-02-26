from transformers import pipeline
import pandas as pd
import os

# Define the pipeline
pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

# Check if the file exists
file_path = "oai_DE-MUS-048017_703_johannes-verspronck-portrait-woman-armchair-703--thumb-xl_output.csv"
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Define the base message
    base_message = {"role": "user", "content": "Write me a 125 character summary out of the csv input text file."}

    # Use the pipeline to generate a response with max_new_tokens
    for index, row in df.iterrows():
        # Example content from the CSV
        content = row['Description']
        messages = [base_message, {"role": "user", "content": content}]
        
        response = pipe(messages, max_new_tokens=200)

        # Print the generated response
        summary = response[0]["generated_text"]
        print(f"Processed Text: {content}\nSummary: {summary}\n")
