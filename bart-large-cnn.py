from transformers import pipeline
import pandas as pd
import os
import csv

# Define the summarization pipeline
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    print("Summarization pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

# Specify your input CSV file
input_file = "oai_DE-MUS-048017_703_johannes-verspronck-portrait-woman-armchair-703--thumb-xl_output.csv"

# Check if the file exists
if not os.path.exists(input_file):
    print(f"File not found: {input_file}")
    exit()

# Read the CSV file
try:
    df = pd.read_csv(input_file)
    if 'Description' not in df.columns:
        print("Error: 'Description' column not found.")
        exit()
    print("CSV file loaded successfully.")
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit()

def truncate_at_sentence(text, max_chars):
    text = text.strip()
    if len(text) <= max_chars:
        return text.rstrip('.') + '.'  # Ensure a single trailing period

    truncated = text[:max_chars]
    last_boundary = max(
        truncated.rfind('.'),
        truncated.rfind('?'),
        truncated.rfind('!')
    )

    if last_boundary == -1:
        return truncated.strip() + '.'
    return truncated[:last_boundary + 1].strip()

formatted_summaries = []

for index, row in df.iterrows():
    content = row['Description']
    content_truncated = truncate_at_sentence(content, 1024)

    try:
        # Generate the summary
        summary_text = summarizer(
            content_truncated,
            max_length=150,
            min_length=100,
            do_sample=False
        )[0]['summary_text']
        # Truncate the summary at the last sentence within 125 characters
        summary_truncated = truncate_at_sentence(summary_text, 125)
    except Exception as e:
        print(f"Error summarizing row {index}: {e}")
        summary_truncated = "Error: Summary generation failed."

    # Format the description and summary without adding extra quotation marks
    formatted_description = content_truncated.replace('\n', ' ').replace('\r', ' ')
    formatted_summary = f'Summary: {summary_truncated}'.replace('\n', ' ').replace('\r', ' ')

    formatted_summaries.append((formatted_description, formatted_summary))
    print(f"Processed Text:\n{formatted_description}\n{formatted_summary}\n")

# Create a new DataFrame with the formatted outputs
output_df = pd.DataFrame(formatted_summaries, columns=['Description', 'Formatted Summary'])

# Generate the output file name
output_file = f"{os.path.splitext(input_file)[0]}_summary.csv"

# Save to CSV with minimal quoting
try:
    output_df.to_csv(
        output_file,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        escapechar='\\'
    )
    print(f"Summaries saved to {output_file}")
except Exception as e:
    print(f"Error saving CSV: {e}")
