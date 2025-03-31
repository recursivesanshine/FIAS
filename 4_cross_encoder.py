import os
import csv
from sentence_transformers.cross_encoder import CrossEncoder
import sys

def extract_keywords_from_sections(file_path, sections):
    extracted_data = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for section in sections:
            keywords = []
            section_found = False
            for i, line in enumerate(lines):
                if f"**{section}:**" in line:
                    section_found = True
                    headline = line.strip().strip('**').strip(':')
                    for keyword_line in lines[i+1:]:
                        if keyword_line.strip() == "" or keyword_line.startswith('**'):
                            break
                        keywords.append(keyword_line.strip().strip('-').strip())
                    extracted_data[headline] = keywords
                    break
            if not section_found:
                print(f"Section '{section}' not found in the file.")
    return extracted_data

def extract_corpus_from_csv(file_path, keyword_category):
    corpus = []
    keyword_category = keyword_category.lower()
    if keyword_category == "atmosphere":
        keyword_category = "atmosphäre"
    elif keyword_category == "elements":
        keyword_category = "bildelement"
    
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for row in reader:
            if row[0].lower() == keyword_category:
                corpus.append(row[-1])
    return corpus

if len(sys.argv) < 3:
    print("Usage: python cross_encoder.py path/to/your/text_folder_or_file path/to/your/csv_folder")
    sys.exit(1)

text_folder_or_file = sys.argv[1]
csv_folder_path = sys.argv[2]

model = CrossEncoder("cross-encoder/stsb-distilroberta-base")

sections = ["Keywords for Atmosphere", "Keywords for Emotion", "Picture Elements"]

# Check if the input is a single file or a directory
if os.path.isfile(text_folder_or_file):
    text_files = [text_folder_or_file]  # Single file
elif os.path.isdir(text_folder_or_file):
    text_files = [os.path.join(text_folder_or_file, f) for f in os.listdir(text_folder_or_file) if f.endswith('.txt')]
else:
    print(f"Invalid path: {text_folder_or_file}. It must be a file or directory.")
    sys.exit(1)

for text_file_path in text_files:
    file_name = os.path.basename(text_file_path)
    csv_file_path = os.path.join(csv_folder_path, file_name.replace('.txt', '.csv'))

    if not os.path.exists(csv_file_path):
        print(f"CSV file for '{file_name}' not found. Skipping this file.")
        continue

    output_file = f"{os.path.splitext(file_name)[0]}_results.csv"
    print(f"Processing: {output_file}")  # Print the output file name
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Matching Results'])
        writer.writerow([f'Text file: {text_file_path}'])
        writer.writerow([f'CSV file: {csv_file_path}'])
        writer.writerow([])

    extracted_data = extract_keywords_from_sections(text_file_path, sections)

    for headline, queries in extracted_data.items():
        keyword_category = headline.split()[-1].lower()
        corpus = extract_corpus_from_csv(csv_file_path, keyword_category)
        
        if keyword_category == "elements":
            keyword_category = "bildelemente"
        print(f"Corpus for '{keyword_category}': {corpus}")

        if not corpus:
            print(f"No corpus found for keyword category '{keyword_category}'. Skipping this section.")
            continue

        with open(output_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            for query in queries:
                writer.writerow([f"== {headline}: {query} =="])
                print(f"== {headline}: {query} ==")
                
                sentence_combinations = [[query, sentence] for sentence in corpus]
                if not sentence_combinations:
                    print(f"No sentence combinations for query '{query}'. Skipping this query.")
                    continue
                
                scores = model.predict(sentence_combinations)
                sorted_sentences = sorted(zip(scores, corpus), reverse=True, key=lambda x: x[0])
                
                for score, sentence in sorted_sentences:
                    result = f"{score:.2f}\t{sentence}"
                    print(result)
                    writer.writerow([result])
                
                writer.writerow([])  # Add a blank line for separation
                print("\n")
