from transformers import pipeline
import pandas as pd
import os

# Define the summarization pipeline
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

# Check if the file exists
file_path = "oai_DE-MUS-048017_703_johannes-verspronck-portrait-woman-armchair-703--thumb-xl_output.csv"
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit()

# Read the CSV file
try:
    df = pd.read_csv(file_path)
    if 'Description' not in df.columns:
        print("Error: 'Description' column not found in the CSV file.")
        exit()
except Exception as e:
    print(f"Error reading the CSV file: {e}")
    exit()

# Process each row in the CSV file
summaries = []
for index, row in df.iterrows():
    content = row['Description']
    
    try:
        # Truncate input text if it exceeds the model's token limit
        max_input_length = 1024  # Adjust based on the model's token limit
        if len(content) > max_input_length:
            content = content[:max_input_length]
        
        # Generate the summary
        summary = summarizer(content, max_length=125, min_length=50, do_sample=False)[0]['summary_text']
        
        # Ensure the summary is no more than 125 characters
        summary = summary[:125]  # Truncate to 125 characters
        summaries.append(summary)
        print(f"Processed Text: {content}\nSummary: {summary}\n")
    except Exception as e:
        print(f"Error generating summary for row {index}: {e}")
        summaries.append("Error: Summary generation failed.")

# Save the summaries to a new CSV file
output_file = "summaries_output.csv"
try:
    df['Summary'] = summaries
    df.to_csv(output_file, index=False)
    print(f"Summaries saved to {output_file}")
except Exception as e:
    print(f"Error saving summaries to CSV: {e}")