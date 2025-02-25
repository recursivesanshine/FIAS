import sys
import torch
from transformers import pipeline
import pandas as pd

def main(csv_file_path):
    # Set the device to CPU
    device = torch.device('cpu')
    pipe = pipeline(
        "text-generation",
        model="google/gemma-2-2b-it",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=-1  # Use CPU
    )

    # Define messages
    messages = [
        {"role": "user", "content": "Summarize the input csv text to a 125 characters length summary."},
    ]

    # Load CSV file
    df = pd.read_csv(csv_file_path)

    # Function to process each row based on the user prompt
    def process_text(text):
        input_prompt = f"{messages[0]['content']} {text}"
        output = pipe(input_prompt, max_new_tokens=50)[0]["generated_text"]
        return output.strip()[:125]  # Ensure the output is properly formatted and within 125 characters

    # Apply the function to each row in the 'Description' column
    if 'Description' in df.columns:
        df['result'] = df['Description'].apply(process_text)
    else:
        print("The CSV file does not have a 'Description' column.")
        return

    # Save the results to a new CSV file named "gemma_output.csv"
    output_file_path = 'gemma_output.csv'
    df.to_csv(output_file_path, index=False)

    # Example of generating text
    assistant_response = pipe(messages[0]['content'], max_new_tokens=50)[0]["generated_text"].strip()
    print(assistant_response)

    # Print the full name of the new file
    print(f"Output file saved as: {output_file_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python gemma-2-2b-it.py <csv_file_path>")
    else:
        csv_file_path = sys.argv[1]
        main(csv_file_path)
